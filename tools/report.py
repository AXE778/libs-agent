r"""
tools.report —— 把「质量检查」和「元素匹配」的结论汇总成一份可交付文档（Step 8）。

============================ 为什么单独有这一层 ============================

前面几步各自解决了一个问题：

    Step 3  这条谱**读得进来吗**
    Step 4  这条谱 / 这组谱**能不能用**（饱和、噪声、尖刺、轴一致）
    Step 7  这些峰**支持哪些元素**（候选，不是检出）

但它们吐出来的都是「结构化 dict」—— 适合程序读，不适合交给同事、导师或客户看。
这一步做的事很窄：**把已有的结构化结果，按固定的骨架拼成一份人看的文档。**

它**自己不产生任何新结论、不做任何新的信号处理、更不调用大模型。**
报告里的每一句话都能在 check_quality / analyze_elements 的返回值里找到出处。
允许「概括」，不允许「推断」。

---------------- 三条硬规矩（和 tools/ 的其它模块一致） ----------------

1. 纯计算：不 print、不调大模型、不 import agent/。
   （脚本想显示进度，把回调函数传进 progress 参数即可，本模块自己不 print。）
2. 不产生新结论：见上。
3. **不许出现含量 / 浓度**：这批数据没有浓度标定（没标样、没定标曲线），
   所以报告里只能出现强度、信噪比、峰位、证据强弱 —— 一个浓度数字都不能有。
   报告开头就直接把这条写给读者看。

---------------- 两处「实测改出来」的设计决定 ----------------

① 元素匹配只对**少数几条最干净的谱**做，不是全组做。

   元素匹配比质量检查慢一个量级（要对全体峰跑 200 次置换检验），
   而同一组（同材料、同参数、同一台仪器）的谱线分布本来就高度相似，
   全组跑一遍得到的是同一个结论重复 N 次。

   挑法：**优先没饱和的，同样没饱和时优先信噪比高的。**
   但本机实测 316 那组 25 条**全部打饱和** —— 一条干净的都没有。
   这种情况下退化成「信噪比最高的 3 条」，并且把 fallback 标记为 True，
   报告里**明写**这一点，免得读者以为元素结论建立在好数据上。

② 报告默认落到 `outputs/`，文件名带时间戳。

   报告是「一次检测的存档」。默认覆盖上一次的文件名，就没法拿两次结果做对比了。
   想要固定文件名，显式传 out_name。
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Callable, Sequence

from config.paths import DATA_ROOT, OUTPUT_DIR
from tools import element_analysis, spectrum_catalog, spectrum_loader, spectrum_quality

__all__ = [
    "ReportError",
    "collect_spectrum",
    "collect_group",
    "build_headline",
    "render_markdown",
    "generate_report",
]


class ReportError(Exception):
    """汇总失败时抛出，消息是给人看的中文。"""


# ---------------------------------------------------------------------------
# 0) 小工具
# ---------------------------------------------------------------------------
_DASH = "—"


def _now_stamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _num(v: Any, nd: int = 3) -> str:
    """数字转字符串，None / NaN / 无穷 一律给破折号，绝不给 nan 或 inf 让人误读。"""
    if v is None:
        return _DASH
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if f != f or f in (float("inf"), float("-inf")):
        return _DASH
    if nd == 0:
        return f"{f:,.0f}"
    return f"{f:,.{nd}f}"


def _cell(v: Any) -> str:
    """表格单元：把竖线换掉，否则会把 Markdown 表格撑散。"""
    return str(v).replace("|", "/").replace("\n", " ")


def _table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> list[str]:
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(_cell(c) for c in r) + " |")
    return out


def _bullets(items: Sequence[str], indent: str = "") -> list[str]:
    return [f"{indent}- {it}" for it in items]


def _el_brief(rows: Sequence[dict[str, Any]]) -> str:
    """把元素行拼成 `Cr(43 个峰)、Ni(31 个峰)` 这种紧凑写法。"""
    return "、".join(f"{r['element']}（{r['n_supporting_peaks']} 个峰）" for r in rows)


# ---------------------------------------------------------------------------
# 1) 收集：把已有的结构化结果拼成一份「报告数据」
# ---------------------------------------------------------------------------
def collect_spectrum(
    path: str,
    *,
    with_elements: bool = True,
    tol_nm: float | None = None,
    max_elements: int = 12,
) -> dict[str, Any]:
    """单条谱的报告数据：摘要 + 质量检查 + 元素匹配。只读一次文件。"""
    spec = spectrum_loader.load_spectrum(path)
    out: dict[str, Any] = {
        "kind": "single",
        "summary": spectrum_loader.summarize(spec),
        "quality": spectrum_quality.check_quality(spec),
        "elements": None,
        "element_error": None,
    }
    if with_elements:
        out["elements"], out["element_error"] = _try_elements(
            spec, tol_nm=tol_nm, max_elements=max_elements)
    return out


def _try_elements(spec: Any, *, tol_nm: float | None, max_elements: int
                  ) -> tuple[dict[str, Any] | None, str | None]:
    """元素匹配，失败不抛。

    为什么吃掉异常：元素匹配是整个流程里最重、最容易因为库/数据怪状失败的一步。
    它失败时，质量检查那部分仍然是有效结论 —— 不能因为一个模块挂了
    就让整份报告什么都没有。错误原文如实写进报告，让读者知道缺了什么。
    """
    try:
        return element_analysis.analyze_elements(
            spec, tol_nm=tol_nm, max_elements=max_elements), None
    except Exception as exc:  # noqa: BLE001 —— 故意兜住一切
        return None, f"{type(exc).__name__}: {exc}"


def _pick_element_files(per_file: Sequence[dict[str, Any]], n: int
                        ) -> tuple[list[dict[str, Any]], bool]:
    """挑最值得做元素匹配的 n 条谱，返回 (挑中的行, 是否退化了)。

    为什么是这个顺序：
      · 先排掉打饱和的 —— 峰顶被削平，**峰位也会被拉偏**，
        峰位一偏，元素匹配的第一步（峰 → 候选谱线）就错了；
      · 都没饱和时取信噪比高的 —— 弱谱找出来的「峰」一半是噪声，
        拿噪声去撞谱线库，撞出来的当然是瞎猜水平。
    """
    if not per_file:
        return [], False
    clean = [r for r in per_file if int(r.get("n_saturated") or 0) == 0]
    fallback = not clean
    pool = clean or list(per_file)
    pool = sorted(pool, key=lambda r: -float(r.get("snr") or 0.0))
    return pool[:max(1, int(n))], fallback


def collect_group(
    paths: Sequence[str],
    *,
    material: str | None = None,
    max_files: int = 60,
    element_files: int = 3,
    with_elements: bool = True,
    tol_nm: float | None = None,
    max_elements: int = 12,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """一组谱的报告数据：整组质量检查 + 少数几条谱的元素匹配。

    progress: 可选回调，脚本用它显示「正在做哪一条」，本模块自己不 print。
    """
    if not paths:
        raise ReportError("没有给任何光谱路径")
    paths = [str(p) for p in paths][:max_files]

    # keep_per_file=True：顶层汇总只有数量，挑不出「具体哪一条干净」，
    # 而挑谱正是元素匹配那一步的前置条件。
    quality = spectrum_quality.check_many(
        paths, max_files=max_files, keep_per_file=True)
    per_file = quality.get("per_file") or []
    if not per_file:
        raise ReportError("这组谱一条都没读成功，出不了报告")

    picked: list[dict[str, Any]] = []
    fallback = False
    if with_elements and element_files > 0:
        picked, fallback = _pick_element_files(per_file, element_files)

    elements: list[dict[str, Any]] = []
    for row in picked:
        if progress:
            progress(f"元素匹配：{row['name']}")
        spec_or_err: dict[str, Any] | None = None
        err: str | None = None
        try:
            spec_or_err, err = _try_elements(
                spectrum_loader.load_spectrum(row["path"]),
                tol_nm=tol_nm, max_elements=max_elements)
        except Exception as exc:  # noqa: BLE001 —— 连读文件都失败也算进去
            err = f"{type(exc).__name__}: {exc}"
        elements.append({
            "name": row["name"],
            "path": row["path"],
            "quality_verdict": row.get("verdict"),
            "snr": row.get("snr"),
            "n_saturated": row.get("n_saturated"),
            "report": spec_or_err,
            "error": err,
        })

    dirs = sorted({os.path.dirname(p) for p in paths})
    label = material or (os.path.basename(dirs[0]) if len(dirs) == 1 else "多目录")
    return {
        "kind": "group",
        "material": label,
        "data_dirs": dirs,
        # 关键字可能一次命中多组（实测：material='7075' 会同时命中
        # data-7075-1 / data-7075-2 / data-7075-新 三组）。跨目录的统计量
        # 只有在「同材料、同参数」时才可比，所以这件事必须一路带到报告里说清楚。
        "n_data_dirs": len(dirs),
        "n_files": len(paths),
        "paths": list(paths),
        "quality": quality,
        "elements": elements,
        "element_files_requested": int(element_files),
        # ★ 这个标记必须一路带到报告里 —— 它决定「元素结论能不能当回事」
        "element_fallback_all_saturated": bool(fallback),
    }


# ---------------------------------------------------------------------------
# 2) 结论摘要（给模型的 headline，也给报告的第一节）
# ---------------------------------------------------------------------------
def build_headline(data: dict[str, Any]) -> list[str]:
    """把报告数据压成几条「结论先行」的话。

    ★ 为什么由 Python 拼、而不是让模型自己总结：
      模型总结一遍就等于把数字重打一遍，那正是编造数字的入口。
      这里拼好的句子是**确定性**的，模型只能照抄。
    """
    if data.get("kind") == "single":
        return _headline_single(data)
    return _headline_group(data)


def _element_headline(rep: dict[str, Any]) -> list[str]:
    out = [f"元素谱线匹配：{rep.get('verdict')}（候选，不是检出）"]
    rows = rep.get("elements") or []
    claimed = [r for r in rows if r.get("verdict") == "有支持"]
    if claimed:
        # 空气元素必须单独说 —— 它们本来就飘在空气里，不能当样品成分报
        solid = [r for r in claimed if not r.get("in_air")]
        air = [r for r in claimed if r.get("in_air")]
        if solid:
            out.append(f"　· 证据超出偶然水平：{_el_brief(solid)}")
        if air:
            out.append(f"　· 证据超出偶然水平（**空气元素**，反映气氛不代表样品）："
                       f"{_el_brief(air)}")
    else:
        out.append("　· 没有任何元素的证据明显超出偶然水平。")
    if rep.get("n_peaks_near_miss"):
        out.append(
            f"　· ⚠ {rep['n_peaks_near_miss']} 个峰近失配（差一点点对上强线），"
            "提示这段波长标定可能有偏差，先核对标定，别当成『没有该元素』。")
    return out


def _headline_single(data: dict[str, Any]) -> list[str]:
    q = data["quality"]
    s = data["summary"]
    out = [
        f"这条谱 {s['n_points']:,} 个点，波长 {_num(s['wavelength_min_nm'])}~"
        f"{_num(s['wavelength_max_nm'])} nm，最强峰在 "
        f"{_num(s['peak_wavelength_nm'])} nm（原始强度 {_num(s['peak_intensity'], 0)}）。",
        f"采集质量判级 **{q['verdict']}** —— {q['summary']}",
    ]
    for it in q["issues"][:3]:
        out.append(f"　· {it}")
    if data.get("element_error"):
        out.append(f"元素匹配没跑成：{data['element_error']}")
    elif data.get("elements"):
        out.extend(_element_headline(data["elements"]))
    if q.get("advice"):
        out.append("建议：" + "；".join(q["advice"][:2]))
    return out


def _headline_group(data: dict[str, Any]) -> list[str]:
    q = data["quality"]
    n = q["n_checked"]
    vc = q["verdict_counts"]
    out: list[str] = []

    n_fail = int(vc.get("FAIL", 0))
    out.append(
        f"{n} 条谱的判级分布：OK {vc.get('OK', 0)} / WARN {vc.get('WARN', 0)} / "
        f"FAIL {n_fail}"
        + ("。" if n_fail == 0 else "（FAIL = 打饱和，峰高不再正比于信号强度）。"))

    if int(data.get("n_data_dirs") or 1) > 1:
        dirs = data.get("data_dirs") or []
        out.append(
            f"⚠ 这份报告跨了 {len(dirs)} 个目录（"
            + "、".join(os.path.basename(d) for d in dirs[:6])
            + "）。跨目录汇总要求它们同材料、同采集参数；"
            "如果它们是不同批次，下面这些比例和结论不能直接混着看。"
            "想只针对一组，把 material 换成完整子目录名（如 data-7075-1）重出一份。")

    if q["saturated"]["n_files"]:
        top = q.get("peak_wavelength_nm_top") or []
        where = f"，且最强峰集中在 {top[0][0]} nm 附近（{top[0][1]}/{n} 条）" if top else ""
        out.append(
            f"⚠ {q['saturated']['n_files']}/{n} 条打饱和{where} —— "
            "这一组不适合直接比强度、做定量或做元素成像。")
    if q["weak_signal"]["n_files"]:
        out.append(f"⚠ {q['weak_signal']['n_files']}/{n} 条信号弱"
                   f"（信噪比低于判定线）。")
    if q["high_background"]["n_files"]:
        out.append(f"⚠ {q['high_background']['n_files']}/{n} 条背景偏高。")
    if q["suspected_spikes"]["n_files"]:
        sr = q.get("spike_recurrence") or {}
        out.append(
            f"⚠ {q['suspected_spikes']['n_files']}/{n} 条孤立尖刺偏多"
            f"（已先剔掉 {sr.get('n_recurrent', 0)} 处会重复出现的真谱线）。")

    ax = q.get("axis_consistency") or {}
    if ax:
        # conclusion 本身已经是完整一句话（「波长轴一致，可以直接逐点比较。」），
        # 前面再加「波长轴：」就成了「波长轴：波长轴一致」。
        out.append(str(ax.get("conclusion", _DASH)))

    snr = q.get("snr") or {}
    if snr.get("median") is not None:
        out.append(f"信噪比（中位）约 {_num(snr.get('median'), 0)}，"
                   f"最低 {_num(snr.get('min'), 0)}。")

    els = data.get("elements") or []
    if data.get("element_fallback_all_saturated") and els:
        out.append(
            f"⚠ 元素匹配是在**全部饱和的谱**上做的（这组没有一条不饱和），"
            "峰位可能被削平影响，结论只能当参考。")
    for e in els:
        if e.get("error"):
            out.append(f"{e['name']} 元素匹配没跑成：{e['error']}")
        elif e.get("report"):
            out.append(f"{e['name']}：{e['report'].get('verdict')}")

    return out


# ---------------------------------------------------------------------------
# 3) 渲染成 Markdown
# ---------------------------------------------------------------------------
_WARNINGS = [
    "本报告**不含元素含量 / 浓度**：这批数据没有做浓度标定。",
    "判为『有支持』的元素是**候选**，不是「检出」，最终以人工核对谱线形状与强度比为准。",
    "空气元素（N / O / Ar）在空气里本来就有很强的线，它们『有支持』不代表样品含这些元素。",
    "打饱和（判级 FAIL）的谱，峰顶被削平，峰高不可直接比较或用于定量。",
]


def _extra_warnings(data: dict[str, Any]) -> list[str]:
    """这次报告**特有**的边界（通用四条的补充）。

    为什么要单独拎出来：通用四条每次都一样，模型容易读成套话而跳过；
    这里的两条是「这一次的数据本身有坑」，必须跟着 headline 一起回给模型。
    """
    out: list[str] = []
    if data.get("kind") == "group" and int(data.get("n_data_dirs") or 1) > 1:
        dirs = data.get("data_dirs") or []
        out.append(
            f"本报告跨了 {len(dirs)} 个数据目录（"
            + "、".join(f"`{d}`" for d in dirs[:6])
            + "）：跨目录汇总要求同材料、同采集参数，不同批次不能直接混着看。")
    if data.get("element_fallback_all_saturated"):
        out.append(
            "元素匹配是在**全部饱和的谱**上做的（这一组没有一条不饱和），"
            "峰位可能被削平影响，元素那部分的结论只能当参考。")
    return out


def render_markdown(
    data: dict[str, Any],
    *,
    generated_at: str | None = None,
    title: str | None = None,
) -> str:
    """把报告数据渲染成一份 Markdown 文档（纯函数，不写盘）。"""
    at = generated_at or _now_stamp()
    kind = data.get("kind", "group")
    headline = build_headline(data)

    L: list[str] = []
    L.append(f"# {title or 'LIBS 光谱检测报告'}")
    L.append("")

    # ---- 抬头 ----
    if kind == "single":
        s = data["summary"]
        L += _table(["项", "内容"], [
            ["生成时间", at],
            ["报告类型", "单条光谱"],
            ["数据来源", f"`{s.get('source', _DASH)}`"],
            ["点数 / 波长范围",
             f"{s['n_points']:,} 点，{_num(s['wavelength_min_nm'])} ~ "
             f"{_num(s['wavelength_max_nm'])} nm"],
            ["生成方式", "libs_agent Step 8 自动汇总：数字全部来自 tools/ 的纯计算，未经大模型改写"],
        ])
    else:
        q = data["quality"]
        L += _table(["项", "内容"], [
            ["生成时间", at],
            ["报告类型", "整组测量（二维扫描）"],
            ["数据目录", "、".join(f"`{d}`" for d in data.get("data_dirs", []))],
            ["检查条数", f"{q['n_checked']} 条"],
            ["判级分布",
             f"OK {q['verdict_counts'].get('OK', 0)} / "
             f"WARN {q['verdict_counts'].get('WARN', 0)} / "
             f"FAIL {q['verdict_counts'].get('FAIL', 0)}"],
            ["元素匹配", f"对 {len(data.get('elements') or [])} 条谱做了（见第四节）"],
            ["生成方式", "libs_agent Step 8 自动汇总：数字全部来自 tools/ 的纯计算，未经大模型改写"],
        ])
    L.append("")

    # ---- 必须在最前面说的边界 ----
    L.append("> **先看这几条边界，否则会误读整份报告**")
    L.append(">")
    for w in _WARNINGS + _extra_warnings(data):
        L.append(f"> - {w}")
    L.append("")

    # ---- 一、结论摘要 ----
    L.append("## 一、结论摘要")
    L.append("")
    for i, line in enumerate(headline, 1):
        L.append(f"{i}. {line}")
    L.append("")

    # ---- 二、数据概况 ----
    L.append("## 二、数据概况")
    L.append("")
    if kind == "single":
        L += _render_single_profile(data["summary"])
    else:
        L += _render_group_profile(data)
    L.append("")

    # ---- 三、采集质量 ----
    L.append("## 三、采集质量")
    L.append("")
    if kind == "single":
        L += _render_single_quality(data["quality"])
    else:
        L += _render_group_quality(data["quality"])
    L.append("")

    # ---- 四、元素谱线匹配 ----
    L.append("## 四、元素谱线匹配（候选，非检出）")
    L.append("")
    L += _render_elements(data)
    L.append("")

    # ---- 五、局限 ----
    L.append("## 五、本报告的局限")
    L.append("")
    L += _render_limits(data)
    L.append("")

    # ---- 附录 ----
    L.append("## 附录 A：检查过的文件")
    L.append("")
    L += _render_file_list(data)
    L.append("")

    L.append("## 附录 B：判据与阈值")
    L.append("")
    L += _render_thresholds(data)
    L.append("")

    return "\n".join(L) + "\n"


def _render_single_profile(s: dict[str, Any]) -> list[str]:
    return _table(["项", "值"], [
        ["点数", f"{s['n_points']:,}"],
        ["波长范围", f"{_num(s['wavelength_min_nm'])} ~ {_num(s['wavelength_max_nm'])} nm"],
        ["步长（均值 / 最小 / 最大）",
         f"{_num(s['step_mean_nm'], 4)} / {_num(s['step_min_nm'], 4)} / "
         f"{_num(s['step_max_nm'], 4)} nm"
         + ("（等间距）" if s.get("is_uniform") else "（不等间距，是 CCD 像素到波长的天然映射）")],
        ["强度（最小 / 中位 / 最大）",
         f"{_num(s['intensity_min'], 0)} / {_num(s['intensity_median'], 0)} / "
         f"{_num(s['intensity_max'], 0)}"],
        ["最强峰", f"{_num(s['peak_wavelength_nm'])} nm，强度 {_num(s['peak_intensity'], 0)}"],
        ["读取来源", f"编码 {s.get('encoding')}，分隔符 {s.get('delimiter')!r}，"
                     f"表头 {'有' if s.get('has_header') else '无'}"],
    ])


def _render_group_profile(data: dict[str, Any]) -> list[str]:
    q = data["quality"]
    ax = q.get("axis_consistency") or {}
    rows = ax.get("spectra") or []
    n_points = sorted({r["n_points"] for r in rows}) if rows else []
    rng = rows[0].get("range_nm") if rows else None
    snr = q.get("snr") or {}
    base = q.get("baseline_ratio") or {}
    top = q.get("peak_wavelength_nm_top") or []

    return _table(["项", "值"], [
        ["检查条数", f"{q['n_checked']} 条（请求 {q['n_requested']} 条）"],
        ["每条点数", "、".join(f"{n:,}" for n in n_points) if n_points else _DASH],
        ["波长范围", f"{_num(rng[0])} ~ {_num(rng[1])} nm" if rng else _DASH],
        ["信噪比（最低 / 中位 / 最高）",
         f"{_num(snr.get('min'), 0)} / {_num(snr.get('median'), 0)} / "
         f"{_num(snr.get('max'), 0)}"],
        ["基线占采集上限（中位）",
         f"{_num((base.get('median') or 0) * 100, 2)}%"],
        ["最强峰位置分布（前 3）",
         "、".join(f"{w} nm×{c} 条" for w, c in top[:3]) if top else _DASH],
        ["读取失败", f"{q['n_failed_to_read']} 条"
                     + (f"（例：{q['failures'][0].get('path')}）" if q.get("failures") else "")],
    ])


def _render_single_quality(q: dict[str, Any]) -> list[str]:
    L = [f"**判级 `{q['verdict']}`** —— {q['summary']}", ""]
    L += _table(["检查项", "结果"], [
        ["打饱和", f"{q['saturation']['n_saturated']} 个点"
                   + (f"，最宽的一段 {q['saturation']['saturated_ranges'][0]['from_nm']:.2f}~"
                      f"{q['saturation']['saturated_ranges'][0]['to_nm']:.2f} nm"
                      if q["saturation"].get("saturated_ranges") else "")],
        ["近饱和点", f"{q['saturation']['n_near_saturated']} 个"],
        ["信噪比", f"{_num(q['peak']['snr'], 1)}（主峰净高 {_num(q['peak']['net_height'], 0)}）"],
        ["基线水平", f"{_num(q['baseline']['level'], 0)}"
                     f"（占上限 {_num(q['baseline']['level_ratio_of_ceiling'] * 100, 2)}%）"],
        ["疑似孤立尖峰", f"{q['spikes']['n_suspected']} 处（单条谱分不开窄真谱线与宇宙射线）"],
    ])
    if q.get("issues"):
        L.append("")
        L.append("**发现的问题**")
        L.append("")
        L += _bullets(q["issues"])
    if q.get("advice"):
        L.append("")
        L.append("**建议**")
        L.append("")
        L += _bullets(q["advice"])
    return L


def _render_group_quality(q: dict[str, Any]) -> list[str]:
    n = q["n_checked"]
    L: list[str] = ["**总述**：" + str(q.get("conclusion", _DASH)), ""]

    sr = q.get("spike_recurrence") or {}
    ax = q.get("axis_consistency") or {}
    L += _table(["检查项", "结果"], [
        ["打饱和", f"{q['saturated']['n_files']}/{n} 条"
                   + (f"；单条最多 {q['saturated']['max_saturated_points_in_one_file']} 个点"
                      if q['saturated']['n_files'] else "")],
        ["信号弱", f"{q['weak_signal']['n_files']}/{n} 条"],
        ["背景偏高", f"{q['high_background']['n_files']}/{n} 条"],
        ["孤立尖刺偏多",
         f"{q['suspected_spikes']['n_files']}/{n} 条"
         f"（先剔掉了 {sr.get('n_recurrent', 0)} 处会重复出现的真谱线）"],
        ["波长轴一致", ("是" if ax.get("all_consistent") else "否") if ax else _DASH],
    ])
    if ax and not ax.get("all_consistent"):
        L.append("")
        L.append(f"> {ax.get('conclusion')}")
    if sr.get("recurrent_peaks_nm"):
        L.append("")
        L.append("按重复性认定为**真谱线**的波长（前 10 处）："
                 + "、".join(f"{w} nm" for w in sr["recurrent_peaks_nm"]))

    worst = q.get("worst_files") or []
    if worst:
        all_ok = all(w.get("verdict") == "OK" for w in worst)
        L.append("")
        # 全组都 OK 的时候，「需要优先看的谱」这个标题是误导的 ——
        # 这里给出的只是「信噪比最低的几条」，没有一条有问题。
        L.append("### 3.1 " + ("信噪比最低的几条（全组均为 OK，仅作参考）"
                              if all_ok else "需要优先看的谱"))
        L.append("")
        L += _table(["文件", "判级", "信噪比", "首要问题"], [
            [w.get("name"), w.get("verdict"), _num(w.get("snr"), 0),
             (w.get("issues") or [_DASH])[0]]
            for w in worst
        ])
    return L


def _render_elements(data: dict[str, Any]) -> list[str]:
    L: list[str] = []
    L.append("### 4.1 这个方法能说什么、不能说什么")
    L.append("")
    L += _bullets([
        "谱线库在仪器波段（180.6~954.0 nm）内保留 **17,888 条线 / 86 个元素**，"
        "平均每 nm 23 条 —— 所以**单个峰对上一两条线几乎不构成证据**，"
        "本工具因此只做三件事：报候选、数支持峰、用置换检验算「瞎猜也能撞上这么多」的概率。",
        "判定分三档：**有支持**（值得人工复核）/ **不可采信**（不比瞎猜强）/ "
        "**证据不足**（支持峰少于 2 个）。",
        "即使判为『有支持』也只是**候选**。要定一个元素，得看多条谱线之间的强度比，"
        "那件事本工具没做。",
        "**Fe 基本无法下结论**：它在库里有 3,569 条保留线，随便给个波长撞上它的概率约 46%。",
    ])
    L.append("")

    els = data.get("elements")
    if els is None:
        L.append("### 4.2 结果")
        L.append("")
        L.append("本次**没有做**元素谱线匹配（可按需开启）。")
        return L
    if not els:
        L.append("### 4.2 结果")
        L.append("")
        L.append("没有筛出可做元素匹配的谱。")
        return L

    L.append("### 4.2 参与匹配的谱")
    L.append("")
    L += _table(["文件", "采集判级", "信噪比", "饱和点数"], [
        [e["name"], e.get("quality_verdict"), _num(e.get("snr"), 0),
         e.get("n_saturated")]
        for e in els
    ])
    L.append("")
    if data.get("element_fallback_all_saturated"):
        L.append("> ⚠ **注意**：这一组里**没有一条不饱和的谱**，"
                 "所以上面的元素匹配是在打饱和的谱上做的。"
                 "峰顶被削平会拉偏峰位，第四节的所有结论只能当参考，不能当定论。")
        L.append("")

    L.append("### 4.3 各谱结论")
    L.append("")
    for i, e in enumerate(els, 1):
        L.append(f"#### 4.3.{i} `{e['name']}`")
        L.append("")
        if e.get("error") or not e.get("report"):
            L.append(f"元素匹配没跑成：{e.get('error') or _DASH}")
            L.append("")
            continue
        rep = e["report"]
        L.append(f"**判定**：{rep.get('verdict')}")
        L.append("")
        rows = rep.get("elements") or []
        if rows:
            L += _table(
                ["元素", "支持峰数", "偶然水平（均值）", "尾部概率", "判定", "备注"],
                [[r["element"], r["n_supporting_peaks"], _num(r.get("null_mean"), 2),
                  _num(r.get("p_value"), 4), r["verdict"], _el_note(r)]
                 for r in rows],
            )
        else:
            L.append("没有元素进入元素表。")
        L.append("")
        L.append(f"- 参与统计的峰：{rep.get('n_peaks')} 个"
                 f"（列出 {rep.get('n_peaks_listed')} 个，其中 "
                 f"{rep.get('n_peaks_ambiguous')} 个的容差窗口里不止一个候选）")
        if rep.get("n_peaks_near_miss"):
            L.append(f"- ⚠ **近失配 {rep['n_peaks_near_miss']} 处**："
                     "有强峰在容差里只撞到些弱线，但 0.5 nm 内存在明显更强的线。"
                     "这通常意味着**这段波长标定有偏差**，请先核对标定。")
            for nm in (rep.get("near_miss_peaks") or [])[:5]:
                # 注意 `state` 已经是「元素 + 电离态」的合并写法，例如 Cd II，
                # 不要再单独取元素字段，否则会渲染成 None。
                L.append(f"  - 实测峰 {_num(nm.get('peak_wavelength_nm'))} nm ↔ "
                         f"{nm.get('state')} {_num(nm.get('line_wavelength_nm'))} nm"
                         f"（差 {_num(nm.get('delta_nm'))} nm，"
                         f"超出容差 {_num(nm.get('beyond_tol_by_nm'))} nm）")
        L.append("")

    # 汇总：整份报告里所有「有支持」的元素
    claimed: list[tuple[str, str, str]] = []
    for e in els:
        rep = e.get("report") or {}
        for r in (rep.get("elements") or []):
            if r.get("verdict") == "有支持":
                claimed.append((r["element"], e["name"],
                                "空气元素" if r.get("in_air") else "样品侧"))
    L.append("### 4.4 汇总：全报告里判为『有支持』的元素")
    L.append("")
    if claimed:
        L += _table(["元素", "来自哪条谱", "归类"], [list(c) for c in claimed])
        L.append("")
        L.append("> 只有上表里的元素值得人工复核；没出现在表里的元素是「证据不够」，"
                 "不是「不存在」。")
    else:
        L.append("本次没有任何元素的证据明显超出偶然水平。")
    return L


def _el_note(r: dict[str, Any]) -> str:
    bits: list[str] = []
    if r.get("in_air"):
        bits.append("**空气元素**（不代表样品）")
    if r.get("on_sensitive_line") and r.get("sensitive_hits_nm"):
        bits.append("命中灵敏线 " + "、".join(
            f"{_num(w)} nm" for w in r["sensitive_hits_nm"][:3]))
    if r.get("mean_abs_delta_nm") is not None:
        bits.append(f"平均偏差 {_num(r.get('mean_abs_delta_nm'), 3)} nm")
    return "；".join(bits) if bits else _DASH


def _render_limits(data: dict[str, Any]) -> list[str]:
    L: list[str] = []
    common = [
        "**没有浓度标定**：这批数据没有配套标样与定标曲线，"
        "所以本报告只谈强度、峰位、信噪比和证据强弱，任何含量数字都不可据此得出。",
        "**饱和的谱不能用来比强度**：峰顶被削平后，峰高与真实信号不再成正比。",
        "**单条谱分不开窄真谱线与宇宙射线**：两者在单条谱里形状一样，"
        "只有靠「在多条谱里是否重复出现」来甄别，本报告里的尖刺数已是甄别之后的结果。",
        "**元素结论是候选**：判为『有支持』只说明证据超出偶然水平，"
        "不等于检出；最终要人工核对谱线形状与强度比。",
        "**判据是工程经验值**，不是光谱学标准：阈值集中在 `config/quality.py` 与 "
        "`config/spectral_lines.py`，觉得太松/太严可直接改。",
    ]
    L += _bullets(common)

    # 元素匹配模块自己给出的 caveats（原文保留，才有出处）。
    # ★ 只留「方法层面」的那几条，**砍掉带本次具体数字的**（含 `（listed）` 的
    #   近失配条、含 `只分析到` 的样本量条）。
    #   原因：这些数字是**按谱**算的，三张谱就有三条不同的近失配说明，
    #   原文堆在一起既重复又互相矛盾（「9 个」和「8 个」并排出现，读者不知道信哪个）。
    #   它们该出现的地方是 4.3 各小节，那里每条谱有自己的上下文。
    seen: set[str] = set()
    extra: list[str] = []
    for e in (data.get("elements") or []):
        for c in ((e.get("report") or {}).get("caveats") or []):
            if _is_run_specific_caveat(c):
                continue
            if c not in seen:
                seen.add(c)
                extra.append(c)
    if extra:
        L.append("")
        L.append("元素匹配模块自己声明的方法性注意事项（原文保留；"
                 "每张谱的近失配处数与归属歧义规模已写在 4.3 各小节里）：")
        L.append("")
        L += _bullets(extra, indent="  ")
    return L


_RUN_SPECIFIC_CAVEAT_MARKERS = ("（listed）", "只分析到")


def _is_run_specific_caveat(c: str) -> bool:
    """这条 caveat 是不是「只对某一张谱成立」的？

    这类句子里带着本次运行的具体计数（近失配几处、归属歧义几比几、只分析到几个峰），
    放到「整份报告的局限」里就是错的 —— 它是 per-spectrum 的，不是 per-report 的。
    """
    return any(m in c for m in _RUN_SPECIFIC_CAVEAT_MARKERS)


def _render_file_list(data: dict[str, Any]) -> list[str]:
    if data.get("kind") == "single":
        return [f"- `{data['summary'].get('source', _DASH)}`"]
    lines = [f"共 {data['n_files']} 条：", ""]
    for p in data.get("paths", []):
        lines.append(f"- `{p}`")
    return lines


def _render_thresholds(data: dict[str, Any]) -> list[str]:
    L: list[str] = []
    q = data["quality"]
    th = q.get("thresholds_used") or {}
    if th:
        L.append("**采集质量（`config/quality.py`）**")
        L.append("")
        L += _table(["阈值", "值"], [
            ["ADC 采集上限", _num(th.get("adc_ceiling"), 0)],
            ["信噪比——可接受线", _num(th.get("snr_good"), 0)],
            ["信噪比——及格线", _num(th.get("snr_fair"), 0)],
            ["背景偏高提醒线", f"{_num((th.get('baseline_high_ratio') or 0) * 100, 1)}%"],
            ["尖刺数量提醒线", _num(th.get("max_spikes_warn"), 0)],
        ])
        if th.get("note"):
            L.append("")
            L.append(f"> {th['note']}")

    # 元素匹配的阈值取参与谱里的第一份（同一份配置，参数必然一致）
    eth = None
    for e in (data.get("elements") or []):
        eth = ((e.get("report") or {}).get("thresholds_used")) or None
        if eth:
            break
    if eth:
        L.append("")
        L.append("**元素谱线匹配（`config/spectral_lines.py`）**")
        L.append("")
        L += _table(["阈值", "值"], [
            ["匹配容差", str(eth.get("tol_nm"))],
            ["跃迁概率下限 Aki", f"{eth.get('aki_min'):.0e} /s"],
            ["保留的电离态", "、".join(eth.get("ion_states_kept") or [])],
            ["判定所需最少支持峰数", _num(eth.get("min_peaks_to_claim"), 0)],
            ["尾部概率门槛", _num(eth.get("p_value_max"), 4)],
            ["置换检验次数", _num(eth.get("n_null_shifts"), 0)],
        ])
        if eth.get("note"):
            L.append("")
            L.append(f"> {eth['note']}")
    return L


# ---------------------------------------------------------------------------
# 4) 主入口：生成 + 落盘
# ---------------------------------------------------------------------------
def _default_out_name(kind: str, label: str, at: str) -> str:
    stamp = at.replace("-", "").replace(":", "").replace(" ", "-")
    safe = "".join(ch for ch in str(label) if ch.isalnum() or ch in "-_") or "report"
    return f"report_{safe}_{stamp}.md" if kind == "group" else f"report_{safe}_{stamp}_single.md"


def generate_report(
    *,
    material: str | None = None,
    paths: Sequence[str] | None = None,
    data_root: str | None = None,
    max_files: int = 60,
    element_files: int = 3,
    with_elements: bool = True,
    tol_nm: float | None = None,
    max_elements: int = 12,
    out_name: str | None = None,
    out_dir: str | None = None,
    title: str | None = None,
    generated_at: str | None = None,
    write: bool = True,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """汇总一份报告，默认写进 outputs/ 并把要点回给调用方。

    参数
    ----
    material    : 一组数据的关键字（如 `data-316-1` / `316`）。与 paths 二选一。
    paths       : 直接给文件路径列表。给了就优先用它。**一条**时出「单条谱报告」。
    element_files: 对几条谱做元素匹配。0 = 不做（快很多）。
    write       : False 时只返回内容、不落盘（测试用）。

    返回
    ----
    {ok, report_path, kind, headline, warnings, sections, markdown_chars, ...}
    注意返回里**不带正文全文** —— 正文在磁盘上，几万字塞进模型上下文没有意义。
    """
    at = generated_at or _now_stamp()
    resolved: list[str] = []

    if paths:
        resolved = [str(p) for p in paths]
    elif material:
        found = spectrum_catalog.find_spectra(
            data_root or DATA_ROOT, str(material), limit=max(1, int(max_files)))
        resolved = [f["path"] for f in found["files"]]
        if not resolved:
            raise ReportError(f"没有找到匹配『{material}』的光谱文件")
    else:
        raise ReportError("要么给 material（一组数据的关键字），要么给 paths（文件路径列表）")

    if len(resolved) == 1:
        data = collect_spectrum(
            resolved[0], with_elements=with_elements,
            tol_nm=tol_nm, max_elements=max_elements)
    else:
        data = collect_group(
            resolved, material=material, max_files=max_files,
            element_files=(element_files if with_elements else 0),
            with_elements=with_elements, tol_nm=tol_nm,
            max_elements=max_elements, progress=progress)

    headline = build_headline(data)
    md = render_markdown(data, generated_at=at, title=title)

    # ---- 落盘 ----
    label = data.get("material") or (
        data["summary"].get("name", "spectrum") if data.get("kind") == "single"
        else "group")
    name = out_name or _default_out_name(data.get("kind", "group"), label, at)
    if not name.lower().endswith(".md"):
        name += ".md"
    name = os.path.basename(name)          # 禁止路径穿越
    out_path = os.path.join(out_dir or OUTPUT_DIR, name)

    if write:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(md)

    return {
        "ok": True,
        "report_path": out_path if write else None,
        "written": bool(write),
        "kind": data.get("kind"),
        "title": title or "LIBS 光谱检测报告",
        "generated_at": at,
        "n_files": data.get("n_files", 1),
        "n_files_checked": (data["quality"]["n_checked"]
                            if data.get("kind") == "group" else 1),
        "n_elements_analyzed_files": len(data.get("elements") or []),
        "headline": headline,
        # 通用四条 + 这次特有的边界，模型至少要把相关的带上
        "warnings": list(_WARNINGS) + _extra_warnings(data),
        "sections": ["结论摘要", "数据概况", "采集质量", "元素谱线匹配",
                     "本报告的局限", "附录 A：检查过的文件", "附录 B：判据与阈值"],
        "markdown_chars": len(md),
        "markdown": md if not write else None,
        "note": (
            f"报告已写到 {out_path}。回答用户时：先把 report_path 给用户，"
            "再把 headline 里的要点讲一遍，并至少带上 warnings 里与你回答相关的边界；"
            "**不要自己重新组织或推算数字**，也**不要提含量/浓度**。"
        ),
    }
