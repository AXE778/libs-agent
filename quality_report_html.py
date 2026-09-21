# -*- coding: utf-8 -*-
r"""
quality_report_html.py —— 把「整批数据的质量检查结果」画成一张网页。

为什么要有这一页？
    step4_demo.py 打的是竖排文字，一屏只能看一行。数据一多（17 个组、
    每个组 25~150 条谱），你想知道的是「**哪几个组有问题、严重到什么程度**」，
    而不是把每一行读完。
    HTML 能把「判级比例」直接画成长条，一眼看出哪个组是红的。

跑法（在 <项目根目录> 下）：
    python quality_report_html.py            # 生成 outputs/step4_quality_report.html

它不依赖任何第三方库（不用 matplotlib），只拼 HTML + 内联 CSS，
所以不会出现「装不上包就出不了图」的问题。
"""

from __future__ import annotations

import html
import os
import sys
from datetime import datetime
from typing import Any

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from config.paths import DATA_ROOT, OUTPUT_DIR  # noqa: E402
from tools import spectrum_catalog, spectrum_quality  # noqa: E402

MAX_FILES_PER_GROUP = 60

# 判级配色：合格=绿，留意=琥珀，不能用=红
VERDICT_COLOR = {
    "OK": "#1a7f37",
    "WARN": "#9a6700",
    "FAIL": "#b42318",
}
VERDICT_LABEL = {"OK": "合格", "WARN": "留意", "FAIL": "不能用"}


def _esc(s: Any) -> str:
    return html.escape(str(s))


def _bar(rep: dict[str, Any]) -> str:
    """判级比例条：三段宽度按比例。"""
    vc = rep["verdict_counts"]
    n = max(1, sum(vc.values()))
    parts = []
    for key in ("OK", "WARN", "FAIL"):
        v = vc.get(key, 0)
        if not v:
            continue
        pct = v / n * 100
        parts.append(
            f'<span style="width:{pct:.3f}%;background:{VERDICT_COLOR[key]}" '
            f'title="{VERDICT_LABEL[key]} {v} 条">{v}</span>'
        )
    return f'<div class="bar">{"".join(parts)}</div>'


def _stat_card(label: str, value: str, hint: str = "") -> str:
    h = f'<div class="hint">{_esc(hint)}</div>' if hint else ""
    return (
        f'<div class="stat"><div class="stat-v">{value}</div>'
        f'<div class="stat-l">{_esc(label)}</div>{h}</div>'
    )


def _group_card(name: str, rep: dict[str, Any]) -> str:
    vc = rep["verdict_counts"]
    n = rep["n_checked"]
    snr = rep["snr"]

    sat_n = rep["saturated"]["n_files"]
    weak_n = rep["weak_signal"]["n_files"]
    bg_n = rep["high_background"]["n_files"]
    spike_n = rep["suspected_spikes"]["n_files"]

    ax = rep.get("axis_consistency") or {}
    if ax:
        ax_txt = "一致" if ax.get("all_consistent") else "不一致"
        ax_cls = "ok" if ax.get("all_consistent") else "bad"
        npts = ax.get("distinct_n_points") or []
        if not npts and ax.get("spectra"):
            npts = sorted({s["n_points"] for s in ax["spectra"]})
        ax_hint = f"{len(npts)} 种点数"
    else:
        ax_txt, ax_cls, ax_hint = "未比", "mut", ""

    sr = rep.get("spike_recurrence") or {}
    rec_n = sr.get("n_recurrent", 0)

    worst = rep.get("worst_files") or []
    worst_html = ""
    if worst:
        lis = []
        for w in worst:
            issues = w.get("issues") or ([w["issue"]] if w.get("issue") else [])
            iss = "".join(f"<li>{_esc(t)}</li>" for t in issues[:2])
            lis.append(
                f'<li><b>{_esc(w["name"])}</b> '
                f'<span class="tag {w["verdict"].lower()}">{w["verdict"]}</span> '
                f'SNR {w["snr"]:.0f}<ul>{iss}</ul></li>'
            )
        worst_html = (
            '<div class="sub">最该先看的几条</div><ul class="worst">'
            + "".join(lis) + "</ul>"
        )

    partial = f'<div class="partial">{_esc(rep["partial"])}</div>' if rep.get("partial") else ""

    return f"""
    <section class="card">
      <header>
        <h3>{_esc(name)}</h3>
        <div class="meta">抽查 {n} 条 · 中位 SNR {snr['median']:.0f}
          （{snr['min']:.0f} ~ {snr['max']:.0f}）
          · 波长轴 <span class="tag {ax_cls}">{ax_txt}</span> {_esc(ax_hint)}</div>
      </header>
      {_bar(rep)}
      <div class="stats">
        {_stat_card("打饱和", f"{sat_n}", "峰高不可比")}
        {_stat_card("弱信号", f"{weak_n}", f"SNR<20")}
        {_stat_card("背景偏高", f"{bg_n}", "")}
        {_stat_card("孤立尖刺超线", f"{spike_n}", f"已剔真谱线 {rec_n} 处")}
        {_stat_card("合格 / 留意 / 不能用",
                    f"{vc.get('OK',0)} / {vc.get('WARN',0)} / {vc.get('FAIL',0)}", "")}
      </div>
      <p class="concl">{_esc(rep["conclusion"])}</p>
      {partial}
      {worst_html}
    </section>"""


def build_html(rows: list[tuple[str, dict[str, Any]]],
               out_path: str,
               data_root: str = DATA_ROOT) -> str:
    tot_checked = sum(r["n_checked"] for _, r in rows)
    tot_sat = sum(r["saturated"]["n_files"] for _, r in rows)
    tot_fail = sum(r["verdict_counts"].get("FAIL", 0) for _, r in rows)
    bad_groups = [n for n, r in rows if r["saturated"]["n_files"] or r["verdict_counts"].get("FAIL", 0)]
    axis_bad = [n for n, r in rows
                if (r.get("axis_consistency") or {}).get("all_consistent") is False]

    verdict_line = (
        f"共有 <b>{len(bad_groups)}</b> 个组存在打饱和，"
        f"合计 <b>{tot_sat}</b> 条谱打饱和（占抽查 {tot_checked} 条的 "
        f"{tot_sat / max(1, tot_checked) * 100:.1f}%）。"
        if bad_groups else
        f"抽查的 {tot_checked} 条谱里没有发现打饱和。"
    )
    axis_line = (
        f"另有 <b>{len(axis_bad)}</b> 个组的波长轴不一致，跨组比较前必须先对齐。"
        if axis_bad else "所有组的波长轴一致，可以直接逐点比较。"
    )

    cards = "".join(_group_card(n, r) for n, r in rows)

    doc = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>光谱数据质量体检报告</title>
<style>
  :root {{
    --bg:#f6f7f9; --panel:#ffffff; --ink:#1f2328; --mut:#656d76;
    --line:#d8dee4; --ok:#1a7f37; --warn:#9a6700; --fail:#b42318;
  }}
  * {{ box-sizing:border-box; }}
  body {{
    margin:0; padding:32px 20px 64px; background:var(--bg); color:var(--ink);
    font-family:"Microsoft YaHei","PingFang SC","Segoe UI",system-ui,sans-serif;
    font-size:14px; line-height:1.65;
  }}
  .wrap {{ max-width:1000px; margin:0 auto; }}
  h1 {{ font-size:24px; margin:0 0 6px; }}
  .lead {{ color:var(--mut); margin:0 0 24px; }}
  .summary {{
    background:var(--panel); border:1px solid var(--line); border-radius:10px;
    padding:18px 20px; margin-bottom:24px;
  }}
  .summary p {{ margin:0 0 8px; }}
  .summary p:last-child {{ margin-bottom:0; }}
  .card {{
    background:var(--panel); border:1px solid var(--line); border-radius:10px;
    padding:18px 20px; margin-bottom:16px;
  }}
  .card header {{ display:flex; flex-wrap:wrap; gap:8px 16px;
    align-items:baseline; justify-content:space-between; }}
  .card h3 {{ font-size:16px; margin:0; }}
  .meta {{ color:var(--mut); font-size:13px; }}
  .bar {{ display:flex; height:26px; border-radius:5px; overflow:hidden;
    margin:14px 0 4px; background:#eef1f4; }}
  .bar span {{ display:flex; align-items:center; justify-content:center;
    color:#fff; font-size:12px; font-weight:600; min-width:0; }}
  .stats {{ display:grid; gap:10px; margin:16px 0 12px;
    grid-template-columns:repeat(auto-fit,minmax(140px,1fr)); }}
  .stat {{ border:1px solid var(--line); border-radius:8px; padding:10px 12px; }}
  .stat-v {{ font-size:20px; font-weight:700; }}
  .stat-l {{ font-size:12px; color:var(--mut); }}
  .hint {{ font-size:11px; color:var(--mut); opacity:.85; }}
  .concl {{ margin:8px 0 0; padding-left:10px; border-left:3px solid var(--line); }}
  .partial {{ margin-top:8px; font-size:13px; color:var(--warn); }}
  .sub {{ margin-top:14px; font-size:13px; color:var(--mut); font-weight:600; }}
  ul.worst {{ margin:6px 0 0; padding-left:18px; }}
  ul.worst > li {{ margin-bottom:6px; }}
  ul.worst ul {{ margin:4px 0 0; padding-left:18px; color:var(--mut);
    font-size:13px; }}
  ul.worst ul li {{ margin-bottom:2px; }}
  .tag {{ display:inline-block; padding:0 6px; border-radius:4px;
    font-size:12px; font-weight:600; border:1px solid transparent; }}
  .tag.ok, .tag.mut {{ color:var(--ok); border-color:#b7e0c2; background:#e8f5ec; }}
  .tag.mut {{ color:var(--mut); border-color:var(--line); background:#f2f4f6; }}
  .tag.bad {{ color:var(--fail); border-color:#f3c2bd; background:#fdecea; }}
  .footer {{ color:var(--mut); font-size:12px; margin-top:28px; }}
</style>
</head>
<body>
<div class="wrap">
  <h1>光谱数据质量体检报告</h1>
  <p class="lead">数据根目录：{_esc(data_root)}　·　生成时间：{datetime.now():%Y-%m-%d %H:%M}</p>

  <div class="summary">
    <p>{verdict_line}</p>
    <p>{axis_line}</p>
    <p style="color:var(--mut);font-size:13px">
      「孤立尖刺超线」是<b>剔除了重复出现的真谱线之后</b>的结果 ——
      单条谱分不开窄的真谱线与宇宙射线，只有靠「同一波长是否在多条谱上重复出现」才能区分。
      判级口径与阈值见 config/quality.py（工程经验阈值，不是行业标准）。
    </p>
  </div>

  {cards}

  <p class="footer">
    判级含义：合格 = 未发现问题；留意 = 有问题但未饱和；不能用 = 打饱和，峰高不再正比于信号强度。<br>
    抽查上限：每个组最多 {MAX_FILES_PER_GROUP} 条。未抽查到的谱未纳入统计。
  </p>
</div>
</body>
</html>
"""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(doc)
    return out_path


def collect(max_files: int = MAX_FILES_PER_GROUP) -> list[tuple[str, dict[str, Any]]]:
    """把每个二维扫描组跑一遍 check_many，返回 (组名, 报告)。"""
    rows: list[tuple[str, dict[str, Any]]] = []
    for m in spectrum_catalog.list_materials(DATA_ROOT):
        if not m["grid"]:
            continue
        found = spectrum_catalog.find_spectra(DATA_ROOT, m["name"], limit=max_files)
        paths = [f["path"] for f in found["files"]]
        if not paths:
            continue
        try:
            rep = spectrum_quality.check_many(paths, max_files=max_files)
        except Exception as exc:
            print(f"[跳过] {m['name']}: {type(exc).__name__}: {exc}")
            continue
        if found["n_matched"] > len(paths):
            rep["partial"] = (
                f"这组共 {found['n_matched']} 条，本次只抽查了前 {len(paths)} 条。"
            )
        rows.append((m["name"], rep))
    return rows


def main() -> int:
    out = os.path.join(OUTPUT_DIR, "step4_quality_report.html")
    rows = collect()
    build_html(rows, out)
    print(f"共检查 {len(rows)} 个组，报告已写出：{out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
