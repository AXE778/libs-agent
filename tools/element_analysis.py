r"""
tools.element_analysis —— 元素谱线匹配（Step 7）。

设计约束（见 tools/__init__.py）：
  纯计算，不 print、不调用大模型、不 import agent。
  输入是 Spectrum 对象 + 项目内的谱线库，输出是结构化 dict。

============================ 这一步要诚实到什么程度 ============================

先说结论：**靠波长对上谱线，来判断「这个峰属于哪个元素」，本质上是一件很弱的事。**

原因是这套库太密了（实测，2026-09-21）：

  · 仪器波段内 144,331 条线，平均每 nm 187 条；
  · 就算只留 I/II/III 且 Aki >= 1e7，也还有 17,888 条；
  · 不筛的话，±0.05 nm 窗口里挤着 **13 条线、10 个不同元素**；
  · 更麻烦的是「偶然命中概率」—— 在筛后的库里随便给一个波长，
    它在 ±0.05 nm 里碰上 **Fe** 某条线的概率仍有 **46%**
    （Fe 在波段内保留了 3,569 条线，是线最多的元素）。

也就是说：**「652.38 nm 这个峰对上了 Fe 的一条线」这句话几乎没有信息量。**

所以这个模块**不做「判一个元素出来」这种事**。它做三件事：

  1. 报出每个峰的**全部候选**（窗口里有多少条线、多少个元素）—— 把歧义摆到台面上；
  2. 报出每个元素的**支持峰数**，并用**置换检验**给出「瞎猜也能撞上这么多」
     的概率 —— 把整组峰挪到波段里随机位置重跑 200 次，看能不能撞出同样的数量；
  3. 明确区分「有多个独立峰支持」和「只有 1 个峰对上，什么都不算」。

判定只有三种，含义严格区分：

| 判定 | 意思 |
|---|---|
| **有支持** | 支持峰数 ≥ 2，且置换检验的尾部概率 < 0.01。**值得人工复核**，不等于检出。 |
| **不可采信** | 支持峰数够，但正好是「瞎猜也能撞上这么多」的量级。不构成证据。 |
| **证据不足** | 支持峰数 < 2。单个峰对上什么都不算。 |

宁可少报，不要把一个巧合说成检出。

============================ 两处实测改出来的设计 ============================

**（一）容差必须跟着像素走。** 原来写死 ±0.05 nm，结果短波端全对得上、
长波端全落空：O I 777.417 实测在 777.363（差 0.054）、K I 766.490 实测在
766.360（差 0.130），都因为「超容差 0.004 nm」而匹配不上。
查下来是这台仪器的像素步长从 190 nm 处的 0.065 nm 变到 780 nm 处的 0.133 nm ——
**一个峰的位置精度不可能好过它占的那个像素**。所以容差 =
`clip(1.0 × 局部像素步长, 0.05, 0.20)`。（三个光谱的 δ 中位数都在 +0.003 nm，
说明波长轴本身没有系统偏移，纯粹是容差没跟着色散变。）

**（二）峰门槛不能按「最强峰的比例」卡太狠。** 304 那条 652.38 nm 的线，
峰高 57,220，而第二强的线只有 647 —— 差 88 倍。按「最强峰的 2%」卡，
整条谱只剩 1 个峰。门槛的用处只是「别把噪声当峰」，改成 0.2%。

============================ 为什么用置换检验而不是公式 ============================

公式法（把谱线当均匀分布）会**系统性高估显著性**，因为它完全忽略了
「真实谱线在波长上不是均匀的」这件事：库里的线在哪些波段密、哪些波段疏，
是固定结构，而随机撒的峰照样会落在密集处。

实测证据：316 不锈钢谱用公式法算，Br 和 Pt 都判了「有支持」——
316 里根本没有 Pt，这是纯假阳性。换成置换检验后，这两个都会掉回「不可采信」。

置换检验在说什么，一句大白话：
**「把你这组峰整体搬到波段里另一个不相干的位置，还能不能撞出同样多的线？」
搬 200 次都撞不出来，才算有点意思。**

要注意：**Fe 是这套方法的天敌** —— 即使砍到 1e7，它还剩 3,569 条线，
偶然命中率 46%，几乎是「每个峰都能撞上一条」。所以本模块对 Fe 基本无法给结论，
这不是 bug，是这套数据本身的性质。想真正定 Fe，得靠谱线间的**强度比**，
不是靠波长。
"""

from __future__ import annotations

import csv
import json
import os
import re
import sys
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Sequence

import numpy as np

# 本模块要读 config/ 的路径与阈值，保证项目根在 sys.path 上
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from config import spectral_lines as SL  # noqa: E402
from .spectrum_loader import Spectrum, load_spectrum  # noqa: E402
from . import spectrum_quality as SQ  # noqa: E402

__all__ = [
    "ElementAnalysisError",
    "SpectralLine",
    "LineLibrary",
    "load_line_library",
    "load_sensitive_lines",
    "peak_tolerance_nm",
    "find_peaks",
    "match_peaks",
    "analyze_elements",
    "analyze_file",
]


class ElementAnalysisError(Exception):
    """谱线库缺失 / 读不动 / 参数不合理时抛，消息写成人话。"""


_SYMBOL_RE = re.compile(r"^[A-Z][a-z]?$")


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SpectralLine:
    element: str        # 基态元素符号，如 Fe
    ion: str            # 电离态，如 I / II（裸符号行统一记作 I）
    state: str          # 合起来，如 "Fe I"
    wavelength: float   # nm
    intensity: float    # 库里的相对线强（★跨元素不可比，只作展示）
    aki: float          # 跃迁概率，1/s（跨元素可比，筛选用它）

    @property
    def label(self) -> str:
        return f"{self.state} {self.wavelength:.3f} nm"


@dataclass
class LineLibrary:
    """筛好、排好序的谱线库。"""

    lines: list[SpectralLine]
    wavelengths: np.ndarray                     # 递增，用于二分查找
    by_element: dict[str, list[SpectralLine]]
    band: tuple[float, float]
    n_total_in_band: int                        # 波段内原始线数（未筛）
    n_kept: int
    n_kept_by_aki: int                          # 靠 Aki 门槛留下来的
    n_kept_only_by_sensitive: int               # Aki 不够、但因为它是灵敏线才留的
    aki_min: float
    ion_states_kept: tuple[str, ...]
    n_dropped_high_ion: int                     # 因电离态太高被删
    n_dropped_weak: int                         # 因 Aki 太低被删
    n_sensitive_total: int
    n_sensitive_kept: int
    n_elements_in_json: int                     # elements.json 里的元素数（仅作对照）

    def summary(self) -> dict[str, Any]:
        return {
            "n_lines_total_in_band": self.n_total_in_band,
            "n_lines_kept": self.n_kept,
            "band_nm": [self.band[0], self.band[1]],
            "n_elements": len(self.by_element),
            "density_lines_per_nm": round(
                self.n_kept / (self.band[1] - self.band[0]), 2),
            "filter": {
                "ion_states_kept": list(self.ion_states_kept),
                "aki_min": self.aki_min,
                "always_keep_sensitive_lines": SL.ALWAYS_KEEP_SENSITIVE_LINES,
                "n_lines_by_aki": self.n_kept_by_aki,
                "n_lines_extra_from_sensitive_table": self.n_kept_only_by_sensitive,
                "n_dropped_high_ion": self.n_dropped_high_ion,
                "n_dropped_low_aki": self.n_dropped_weak,
            },
            "sensitive_lines_kept": f"{self.n_sensitive_kept}/{self.n_sensitive_total}",
            "elements_json_symbols": self.n_elements_in_json,
            "note": (
                "只留 I/II/III 且 Aki 达标的线，灵敏线表里的线无条件保留。"
                "筛选用 Aki（跃迁概率，跨元素可比）而不是 intensity"
                "（那一列在不同元素之间量纲不一致，拿它当门槛会整片删掉元素，例如 Mg 与 Hα）。"
                "elements.json 只列 92 个符号，比库里的元素还少，所以没拿它当白名单。"
            ),
        }


# ---------------------------------------------------------------------------
# 加载谱线库（项目内副本，不碰任何外部目录）
# ---------------------------------------------------------------------------
def _read_elements_json(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8-sig") as fh:
        return list(json.load(fh).get("elements", []))


def _parse_state(cell: str) -> tuple[str, str] | None:
    """把库里的 'Fe II' / 'H' 解析成 (元素, 电离态)。

    裸符号行（库里 142 条，实测全是 H）按中性处理 ——
    氢的巴耳末线在库表里不写电离态，但物理上就是 H I。
    """
    parts = cell.split()
    if not parts:
        return None
    el = parts[0]
    if not _SYMBOL_RE.match(el):
        return None
    if len(parts) == 1:
        return el, "I"
    return el, parts[-1]


@lru_cache(maxsize=4)
def load_line_library(
    aki_min: float = SL.LINE_AKI_MIN,
    band_min: float = SL.BAND_MIN_NM,
    band_max: float = SL.BAND_MAX_NM,
) -> LineLibrary:
    """读项目内的谱线库并筛成可用形态（结果缓存，读一次约 0.4 秒）。

    不做的事：不去读 `<LIBS 软件安装目录>` 的任何文件。项目必须自给自足。
    """
    path = SL.INTENSITY_CSV
    if not os.path.exists(path):
        raise ElementAnalysisError(
            f"找不到谱线库：{path}\n"
            "本项目**不自带**谱线库 —— 它是仪器/软件自带的数据库文件，不便随代码分发。\n"
            "请自备一份 CSV 放到上面这个位置，至少要有这几列：\n"
            "    Element, Wavelength, intensity, Aki\n"
            "（Wavelength 单位 nm，Aki 是跃迁概率，筛选时用它而不是 intensity）\n"
            "获取途径：\n"
            "  ① 你的 LIBS 软件安装目录里通常就带一份谱线库，导出成 CSV 即可；\n"
            "  ② NIST Atomic Spectra Database —— https://physics.nist.gov/asd\n"
            "     公开可查，按元素导出后整理成上面的列。\n"
            "放好之后，Step 7 / Step 8 的元素匹配功能就能用；\n"
            "缺这个文件不影响其它功能（读谱、质量检查、预处理、绘图都能跑）。"
        )

    sensitive = load_sensitive_lines()
    ions_ok = set(SL.ION_STATES_KEPT)
    keep_sens = SL.ALWAYS_KEEP_SENSITIVE_LINES
    near = SL.SENSITIVE_NEAR_NM

    def _on_sensitive_line(el: str, wl: float) -> bool:
        """这条库线是不是（该元素）人工挑好的分析线？"""
        if not keep_sens:
            return False
        return any(abs(wl - s) <= near for s in sensitive.get(el, ()))

    lines: list[SpectralLine] = []
    n_total = 0
    n_by_aki = 0
    n_only_sens = 0
    n_drop_ion = 0
    n_drop_weak = 0

    try:
        with open(path, "r", encoding="utf-8-sig", errors="replace") as fh:
            rdr = csv.reader(fh)
            next(rdr, None)  # 表头 Element,Wavelength,intensity,Aki
            for row in rdr:
                if len(row) < 4:
                    continue
                try:
                    wl = float(row[1])
                    inten = float(row[2])
                    aki = float(row[3])
                except ValueError:
                    continue
                if not (band_min <= wl <= band_max):
                    continue
                n_total += 1

                parsed = _parse_state(row[0])
                if parsed is None:
                    continue
                el, ion = parsed

                rescued = _on_sensitive_line(el, wl)
                if ion not in ions_ok and not rescued:
                    n_drop_ion += 1
                    continue
                if aki < aki_min and not rescued:
                    n_drop_weak += 1
                    continue

                if aki >= aki_min:
                    n_by_aki += 1
                else:
                    n_only_sens += 1
                lines.append(SpectralLine(el, ion, f"{el} {ion}", wl, inten, aki))
    except OSError as exc:
        raise ElementAnalysisError(f"谱线库读取失败：{path}（{exc}）") from exc

    if not lines:
        raise ElementAnalysisError(
            f"谱线库筛完一条都没剩下。检查 LINE_AKI_MIN（当前 {aki_min}）是不是设得太高。"
        )

    lines.sort(key=lambda ln: ln.wavelength)
    wavelengths = np.asarray([ln.wavelength for ln in lines], dtype=float)

    by_element: dict[str, list[SpectralLine]] = {}
    for ln in lines:
        by_element.setdefault(ln.element, []).append(ln)

    # 灵敏线保留情况：277 条里有多少条在库里找到了落脚点
    n_sens_total = sum(len(v) for v in sensitive.values())
    n_sens_kept = 0
    for el, wls in sensitive.items():
        for s in wls:
            lo = int(np.searchsorted(wavelengths, s - near, side="left"))
            hi = int(np.searchsorted(wavelengths, s + near, side="right"))
            if hi > lo and any(ln.element == el for ln in lines[lo:hi]):
                n_sens_kept += 1

    try:
        n_json = len(_read_elements_json(SL.ELEMENTS_JSON))
    except Exception:
        n_json = 0

    return LineLibrary(
        lines=lines,
        wavelengths=wavelengths,
        by_element=by_element,
        band=(band_min, band_max),
        n_total_in_band=n_total,
        n_kept=len(lines),
        n_kept_by_aki=n_by_aki,
        n_kept_only_by_sensitive=n_only_sens,
        aki_min=aki_min,
        ion_states_kept=tuple(SL.ION_STATES_KEPT),
        n_dropped_high_ion=n_drop_ion,
        n_dropped_weak=n_drop_weak,
        n_sensitive_total=n_sens_total,
        n_sensitive_kept=n_sens_kept,
        n_elements_in_json=n_json,
    )


@lru_cache(maxsize=1)
def load_sensitive_lines() -> dict[str, list[float]]:
    """读分析灵敏线表：{元素: [灵敏线波长, ...]}，共 277 条 / 64 元素。

    这是人工挑好的「该用哪条线看这个元素」，比从库里乱撞可靠得多。
    注意这个文件**没有表头**，第一行就是数据（`Ag,328.068`）。
    """
    path = SL.SENSITIVE_LINE_CSV
    if not os.path.exists(path):
        return {}
    out: dict[str, list[float]] = {}
    with open(path, "r", encoding="utf-8-sig", errors="replace") as fh:
        for row in csv.reader(fh):
            if len(row) < 2:
                continue
            el = row[0].strip()
            if not _SYMBOL_RE.match(el):
                continue
            try:
                wl = float(row[1])
            except ValueError:
                continue
            out.setdefault(el, []).append(wl)
    for v in out.values():
        v.sort()
    return out


# ---------------------------------------------------------------------------
# 容差：跟着像素走
# ---------------------------------------------------------------------------
def peak_tolerance_nm(step_nm: float | None) -> float:
    """按「峰所在处的局部像素步长」算匹配容差。

    一个峰的位置精度不可能好过它占的那个像素（实测步长 0.065~0.142 nm）。
    写死 0.05 nm 会让长波端的强线全部匹配不上 —— 这是实测踩出来的，见模块说明。
    """
    if step_nm is None or not np.isfinite(step_nm) or step_nm <= 0:
        return SL.MATCH_TOL_FLOOR_NM
    return float(min(SL.MATCH_TOL_CAP_NM,
                     max(SL.MATCH_TOL_FLOOR_NM, SL.TOL_PIXELS * step_nm)))


def _resolve_tol(peak: dict[str, Any], tol_nm: float | None) -> float:
    """调用方给了固定容差就用固定的，否则按峰所在处的像素步长算。"""
    if tol_nm is None:
        return peak_tolerance_nm(peak.get("wl_step_nm"))
    return float(tol_nm)


# ---------------------------------------------------------------------------
# 寻峰（复用 Step 4 的基线与噪声估计，避免两套标准）
# ---------------------------------------------------------------------------
def find_peaks(
    spec: Spectrum,
    *,
    min_snr: float = SL.PEAK_MIN_SNR,
    min_height_ratio: float = SL.PEAK_MIN_RATIO_TO_STRONGEST,
    max_peaks: int = SL.MAX_PEAKS_ANALYZED,
    stratify_width_nm: float = SL.PEAK_STRATIFY_WIDTH_NM,
    stratify_min_per_segment: int = SL.PEAK_STRATIFY_MIN_PER_SEGMENT,
    exclude_suspected_spikes: bool = True,
) -> dict[str, Any]:
    """在谱里找峰。

    为什么复用 spectrum_quality 的估计：
      如果质量检查和寻峰各用一套基线和噪声标准，就会出现「质量模块说这条谱
      信噪比 5000，寻峰模块却按另一个噪声去找峰」这种内部打架的事。
      统一到一处，数字才自洽。

    为什么要剔掉疑似尖刺：
      宇宙射线会形成一个又高又窄的孤立峰，直接拿去匹配就是凭空造一个假元素。

    为什么取样要按波段分层：
      见 config.PEAK_STRATIFY_WIDTH_NM 的说明。一句话 ——
      LIBS 谱的强度分布极不均匀（长波端的空气线和连续背景最强），
      「全谱取最高 N 个」会把紫外段的金属灵敏线整个挤掉。
    """
    from scipy.signal import find_peaks as _fp

    y = np.asarray(spec.intensity, dtype=float)
    wl = np.asarray(spec.wavelength, dtype=float)
    if y.size < 16:
        raise ElementAnalysisError("光谱点太少，无法寻峰。")

    base = SQ.estimate_baseline(y)
    resid = y - base["curve"]
    noise = SQ.estimate_noise(y, base["curve"])
    sigma = float(noise["sigma_used"]) or 1.0

    peak_net_max = float(resid.max())
    thresh = max(min_snr * sigma, min_height_ratio * peak_net_max)

    idx, props = _fp(resid, height=thresh, width=1)
    # 注意 scipy 返回的键名是 peak_heights（不是 height），这里踩过一次。
    heights = props["peak_heights"]
    widths = props.get("widths", np.full(len(idx), np.nan))

    step_all = np.gradient(wl)   # 每个点处的局部像素步长（nm/像素）

    exclude_wl: list[float] = []
    n_spike_excluded = 0
    if exclude_suspected_spikes:
        sp = SQ.detect_spikes(y, wl, sigma, peak_net_max,
                              dominant_wl=float(wl[int(np.argmax(resid))]))
        exclude_wl = [float(s["wavelength_nm"]) for s in sp["spikes"]]

    peaks: list[dict[str, Any]] = []
    for k, i in enumerate(idx):
        w = float(wl[i])
        if exclude_wl and any(abs(w - e) <= SL.MATCH_TOL_FLOOR_NM for e in exclude_wl):
            n_spike_excluded += 1
            continue
        wpt = float(widths[k]) if np.isfinite(widths[k]) else None
        step_local = float(step_all[i])
        peaks.append({
            "index": int(i),
            "wavelength_nm": w,
            "raw_intensity": float(y[i]),
            "net_height": float(heights[k]),
            "snr": float(heights[k] / sigma),
            "width_points": wpt,
            "fwhm_nm": wpt * step_local if wpt is not None else None,
            "wl_step_nm": step_local,
            "tol_nm": peak_tolerance_nm(step_local),
        })

    peaks.sort(key=lambda p: -p["net_height"])
    total = len(peaks)

    # ★ 选峰：全局前 N 个 + 每段保底配额（只补，不砍）。
    # 见 config 里的说明：7075 铝合金谱最强的一批峰全在 766~940 nm 的空气线上，
    # 只取全局前 N 会让紫外段的金属线拿不到名额。
    lo_nm = float(wl[0])
    seg_w = float(stratify_width_nm)
    chosen: dict[int, dict[str, Any]] = {p["index"]: p for p in peaks[:int(max_peaks)]}
    per_seg: dict[int, int] = {}
    if seg_w > 0:
        buckets: dict[int, list[dict[str, Any]]] = {}
        for p in peaks:                      # peaks 已按峰高降序 → 每段天然取到最高的几个
            s = int((p["wavelength_nm"] - lo_nm) // seg_w)
            buckets.setdefault(s, []).append(p)
        quota = int(stratify_min_per_segment)
        for s, group in buckets.items():
            for p in group[:quota]:
                chosen.setdefault(p["index"], p)
        per_seg = {s: len(group) for s, group in sorted(buckets.items())}

    peaks = sorted(chosen.values(), key=lambda p: -p["net_height"])

    return {
        "peaks": peaks,
        "n_peaks_found": total,
        "n_peaks_reported": len(peaks),
        "n_peaks_limited": bool(total > len(peaks)),
        "n_excluded_as_spike": n_spike_excluded,
        "selection": (
            f"全局最高的 {int(max_peaks)} 个，另加每 {seg_w:.0f} nm 段的保底 "
            f"{int(stratify_min_per_segment)} 个（合计 {len(peaks)} 个）"
            if seg_w > 0 else f"全谱取最高的 {int(max_peaks)} 个"
        ),
        "peaks_per_segment_found": per_seg,
        "thresholds": {
            "min_snr": min_snr,
            "min_ratio_to_strongest": min_height_ratio,
            "min_net_height": thresh,
            "sigma_used": sigma,
            "max_peaks": max_peaks,
            "stratify_width_nm": seg_w,
            "stratify_min_per_segment": int(stratify_min_per_segment),
        },
        "note": "已按净峰高从大到小排序；疑似宇宙射线尖刺已剔除，不参与匹配。",
    }


# ---------------------------------------------------------------------------
# 把一个峰匹配到库里的候选谱线
# ---------------------------------------------------------------------------
def _window_items(library: LineLibrary, c: float, tol: float) -> list[dict[str, Any]]:
    """库中落在 [c-tol, c+tol] 的全部谱线（按波长贴近、Aki 次之排序）。"""
    arr = library.wavelengths
    lo = int(np.searchsorted(arr, c - tol, side="left"))
    hi = int(np.searchsorted(arr, c + tol, side="right"))
    items = [
        {
            "element": ln.element,
            "ion": ln.ion,
            "state": ln.state,
            "line_wavelength_nm": ln.wavelength,
            "delta_nm": abs(ln.wavelength - c),
            "line_intensity": ln.intensity,
            "aki": ln.aki,
            "peak_wavelength_nm": c,
        }
        for ln in library.lines[lo:hi]
    ]
    # 不用 intensity 排序 —— 那一列跨元素不可比，见 config 里的坑。
    items.sort(key=lambda d: (d["delta_nm"], -d["aki"]))
    return items


def _near_miss(library: LineLibrary, c: float, tol: float) -> dict[str, Any] | None:
    """容差之外、近失配范围之内，Aki 最强的那条线。

    用来回答：「这个峰差一点点就能对上一个强线吗？」
    对上了 → 要么波长标定偏了，要么库里缺线，两种情况都该让人看到。
    """
    span = SL.NEAR_MISS_SPAN_NM
    arr = library.wavelengths
    lo = int(np.searchsorted(arr, c - span, side="left"))
    hi = int(np.searchsorted(arr, c + span, side="right"))
    best_out = None          # 窗口外 Aki 最强的线
    best_in = 0.0            # 窗口内 Aki 最强的线
    for ln in library.lines[lo:hi]:
        d = abs(ln.wavelength - c)
        if d <= tol:                       # 窗口内的已经匹配上了，不算「近失配」
            best_in = max(best_in, ln.aki)
            continue
        if ln.aki < SL.NEAR_MISS_MIN_AKI:
            continue
        if best_out is None or ln.aki > best_out[0]:
            best_out = (ln.aki, ln, d)
    if best_out is None:
        return None
    # ★ 必须「外面明显比里面强」才算数。否则在这么密的库里几乎每个峰都会命中，
    #   这条诊断就没有筛选力了（实测踩过：39/40 个峰全命中）。
    if best_out[0] < SL.NEAR_MISS_AKI_RATIO * max(best_in, 1.0):
        return None
    _, ln, d = best_out
    return {"state": ln.state, "line_wavelength_nm": ln.wavelength,
            "delta_nm": round(d, 4), "aki": ln.aki,
            "beyond_tol_by_nm": round(d - tol, 4),
            "aki_vs_best_inside": round(best_out[0] / max(best_in, 1.0), 2)}


def _match_full(peaks: Sequence[dict[str, Any]], library: LineLibrary,
                tol_nm: float | None, with_near_miss: bool = False
                ) -> list[dict[str, Any]]:
    """内部用：每个峰 -> 窗口内的**全部**候选（不截断）。

    match_peaks() 的展示版和 analyze_elements() 的统计都从这里取，
    保证「报出来的候选」和「算进统计的候选」是同一份数据。
    """
    out = []
    for p in peaks:
        c = float(p["wavelength_nm"])
        t = _resolve_tol(p, tol_nm)
        rec: dict[str, Any] = {
            "peak": p, "tol_nm": t,
            "items": _window_items(library, c, t),
        }
        if with_near_miss:
            rec["near_miss"] = _near_miss(library, c, t)
        out.append(rec)
    return out


def match_peaks(peaks: Sequence[dict[str, Any]], library: LineLibrary,
                tol_nm: float | None = None,
                max_candidates: int = 6) -> list[dict[str, Any]]:
    """给每个峰找出容差窗口内的全部候选谱线（展示版，候选列表会截断）。

    ★ 关键：这里**不挑一个赢家**。
      挑赢家等于把「窗口里有 10 个元素」这件重要的事藏起来。
      把候选全部报出来，由后面的统计和人来判断。
    """
    out: list[dict[str, Any]] = []
    for m in _match_full(peaks, library, tol_nm, with_near_miss=True):
        p, items = m["peak"], m["items"]
        out.append({
            "wavelength_nm": float(p["wavelength_nm"]),
            "net_height": float(p["net_height"]),
            "snr": float(p["snr"]),
            "tol_nm": round(float(m["tol_nm"]), 4),
            "n_candidates": len(items),
            "n_element_states": len({d["state"] for d in items}),
            "n_base_elements": len({d["element"] for d in items}),
            "ambiguous": len(items) > 1,
            "candidates": items[:max_candidates],
            "n_candidates_hidden": max(0, len(items) - max_candidates),
            "near_miss": m.get("near_miss"),
        })
    return out


# ---------------------------------------------------------------------------
# 置换检验：把整组峰挪到波段里的随机位置，看还能撞出多少
# ---------------------------------------------------------------------------
def _support_counts(wls: np.ndarray, tols: np.ndarray, library: LineLibrary,
                    shift: float) -> dict[str, int]:
    """在给定偏移下，每个元素被多少个不同的峰支持（一个峰只记一次）。"""
    arr = library.wavelengths
    lines = library.lines
    b0, b1 = library.band
    width = b1 - b0
    counts: dict[str, int] = {}
    for c, t in zip(wls, tols):
        c2 = (float(c) - b0 + shift) % width + b0
        lo = int(np.searchsorted(arr, c2 - t, side="left"))
        hi = int(np.searchsorted(arr, c2 + t, side="right"))
        if hi <= lo:
            continue
        for el in {ln.element for ln in lines[lo:hi]}:
            counts[el] = counts.get(el, 0) + 1
    return counts


def _null_distribution(wls: np.ndarray, tols: np.ndarray, library: LineLibrary,
                       n_shift: int, seed: int) -> dict[str, np.ndarray]:
    """跑 n_shift 次随机整体平移，得到每个元素「瞎猜能撞上几个」的分布。

    为什么要整体平移而不是每个峰各自随机：整体平移保留了这组峰彼此的间距
    （那是样品真实成分决定的），只破坏「它们和库里某元素谱线的位置对齐」。
    这正是我们要检验的东西。
    """
    rng = np.random.default_rng(seed)
    b0, b1 = library.band
    shifts = rng.uniform(0.0, b1 - b0, size=int(n_shift))
    acc: dict[str, list[int]] = {}
    for s in shifts:
        for el, k in _support_counts(wls, tols, library, float(s)).items():
            acc.setdefault(el, []).append(k)
    return {el: np.asarray(v, dtype=float) for el, v in acc.items()}


# ---------------------------------------------------------------------------
# 汇总：哪些元素「真的」有多个峰支持
# ---------------------------------------------------------------------------
def _element_verdict(observed: int, null_mean: float, p_value: float,
                     on_sensitive: bool, min_peaks: int) -> tuple[str, str]:
    """给一个元素的证据强度下判定（这是全模块最需要谨慎的地方）。

    判定只用两件事：够不够「多个独立峰」，以及置换检验的尾部概率够不够小。
    """
    if observed < min_peaks:
        return (
            "证据不足",
            f"只有 {observed} 个峰对上，不足以说明任何问题。"
            "在这么密的库里，单个峰对上属于常事。",
        )
    if p_value >= SL.VERDICT_PVALUE_MAX:
        return (
            "不可采信",
            f"{observed} 个峰支持，但把这组峰整体挪到波段里别的位置，"
            f"平均也能撞上 {null_mean:.1f} 个（尾部概率 {p_value:.3f}）。"
            "这个数量级不构成证据。",
        )
    extra = "，其中有峰踩在该元素人工挑好的灵敏线上" if on_sensitive else ""
    return (
        "有支持",
        f"{observed} 个峰支持，而把峰整体挪到别处平均只撞上 {null_mean:.1f} 个"
        f"（尾部概率 {p_value:.3f}）{extra}。"
        "**这只是「值得人工复核」**：请对着谱图核对谱线形状与强度比，再下结论。",
    )


def analyze_elements(
    spec: Spectrum,
    *,
    tol_nm: float | None = None,
    library: LineLibrary | None = None,
    do_peak_finding: bool = True,
    peaks: list[dict[str, Any]] | None = None,
    max_elements: int = 12,
    n_shift: int = SL.NULL_SHIFTS,
) -> dict[str, Any]:
    """一条光谱的元素匹配分析（Step 7 主入口）。

    参数
    ----
    tol_nm : 固定容差（nm）。**默认 None = 按峰所在处的像素步长自动算**，
             这是实测下来唯一站得住的做法。给了就用给的（不超过上限）。
    n_shift: 置换检验跑多少次。默认 200，p 的分辨率是 1/201 ≈ 0.005。

    返回的结构里，**只有 elements 里判为「有支持」的才值得往下看**；
    peaks 里每个峰的候选数和歧义情况是给人工复核用的。
    """
    if tol_nm is not None and not (0 < tol_nm <= SL.MATCH_TOL_CAP_NM):
        raise ElementAnalysisError(
            f"固定容差必须在 0~{SL.MATCH_TOL_CAP_NM} nm 之间，收到 {tol_nm}。"
            "不传这个参数就是按像素步长自动算。"
        )
    lib = library or load_line_library()

    if do_peak_finding:
        pk = find_peaks(spec)
        peak_list = pk["peaks"]
        peak_meta: dict[str, Any] = {
            "n_peaks_found": pk["n_peaks_found"],
            "n_peaks_analyzed": pk["n_peaks_reported"],
            "n_peaks_limited": pk["n_peaks_limited"],
            "n_excluded_as_spike": pk["n_excluded_as_spike"],
            "selection": pk["selection"],
            "thresholds": pk["thresholds"],
        }
    else:
        peak_list = list(peaks or [])
        peak_meta = {"source": "调用方直接传入的峰"}

    base_out: dict[str, Any] = {
        "ok": True,
        "name": spec.meta.get("name", "?"),
        "source": spec.meta.get("path", "?"),
        "library": lib.summary(),
        "peak_finding": peak_meta,
        "thresholds_used": {
            "tol_nm": ("自动：1.0×局部像素步长，夹在 "
                       f"{SL.MATCH_TOL_FLOOR_NM}~{SL.MATCH_TOL_CAP_NM} nm"
                       if tol_nm is None else tol_nm),
            "min_peaks_to_claim": SL.MIN_PEAKS_TO_CLAIM,
            "p_value_max": SL.VERDICT_PVALUE_MAX,
            "n_null_shifts": int(n_shift),
            "aki_min": lib.aki_min,
            "ion_states_kept": list(lib.ion_states_kept),
            "note": "均为工程经验设定（config/spectral_lines.py 可改），不是光谱学标准。",
        },
    }

    if not peak_list:
        base_out.update({
            "n_peaks": 0,
            "verdict": "没有找到可用的峰",
            "elements": [],
            "n_elements_reported": 0,
            "peaks": [],
            "caveats": ["这条谱里没找到高于噪声门槛的峰，谈不上元素匹配。"],
        })
        return base_out

    full = _match_full(peak_list, lib, tol_nm)
    n_peaks = len(full)
    sensitive = load_sensitive_lines()

    # ---- 观测到的支持峰数 + 细节 ----
    per_el: dict[str, dict[str, Any]] = {}
    for m in full:
        # 同一元素在一个峰上只记一次（窗口里它可能有好几条线）
        best_here: dict[str, dict[str, Any]] = {}
        for d in m["items"]:
            el = d["element"]
            if el not in best_here or d["delta_nm"] < best_here[el]["delta_nm"]:
                best_here[el] = d
        for el, d in best_here.items():
            slot = per_el.setdefault(el, {"peaks": [], "sensitive_hits": []})
            slot["peaks"].append({
                "peak_wavelength_nm": d["peak_wavelength_nm"],
                "line_wavelength_nm": d["line_wavelength_nm"],
                "delta_nm": d["delta_nm"],
                "ion": d["ion"],
            })
            sens = sensitive.get(el, [])
            if sens and any(abs(d["line_wavelength_nm"] - s) <= SL.SENSITIVE_NEAR_NM
                            for s in sens):
                slot["sensitive_hits"].append(d["line_wavelength_nm"])

    # ---- 置换检验的零分布 ----
    wls = np.asarray([m["peak"]["wavelength_nm"] for m in full], dtype=float)
    tols = np.asarray([m["tol_nm"] for m in full], dtype=float)
    null = _null_distribution(wls, tols, lib, n_shift, SL.NULL_SEED)
    zeros = np.zeros(int(n_shift), dtype=float)

    rows = []
    for el, slot in per_el.items():
        observed = len(slot["peaks"])
        dist = null.get(el, zeros)
        null_mean = float(dist.mean())
        p_value = (float(np.count_nonzero(dist >= observed)) + 1.0) / (len(dist) + 1.0)
        on_sens = bool(slot["sensitive_hits"])
        verdict, reason = _element_verdict(
            observed, null_mean, p_value, on_sens, SL.MIN_PEAKS_TO_CLAIM)
        rows.append({
            "element": el,
            "n_supporting_peaks": observed,
            "null_mean": round(null_mean, 2),
            "null_p95": round(float(np.percentile(dist, 95)), 2),
            "p_value": round(p_value, 4),
            "in_air": el in SL.AIR_LINE_ELEMENTS,
            "on_sensitive_line": on_sens,
            "sensitive_hits_nm": sorted(set(slot["sensitive_hits"]))[:5],
            "mean_abs_delta_nm": round(
                float(np.mean([p["delta_nm"] for p in slot["peaks"]])), 4),
            "supporting_peaks_nm": [p["peak_wavelength_nm"] for p in slot["peaks"]][:8],
            "verdict": verdict,
            "reason": reason,
        })

    order = {"有支持": 0, "不可采信": 1, "证据不足": 2}
    rows.sort(key=lambda r: (order.get(r["verdict"], 3), r["p_value"]))
    claimed = [r for r in rows if r["verdict"] == "有支持"]
    claimed_air = [r for r in claimed if r["in_air"]]

    n_listed = min(n_peaks, SL.MAX_PEAKS_REPORT)
    matched = match_peaks(peak_list[:n_listed], lib, tol_nm=tol_nm)
    n_amb = sum(1 for m in matched if m["ambiguous"])

    # 近失配：容差内只撞到弱线，但再往外一点就有强线 —— 疑似波长标定偏差或库里缺线
    near = [{"peak_wavelength_nm": m["wavelength_nm"], **m["near_miss"]}
            for m in matched if m.get("near_miss")]
    near.sort(key=lambda d: -d["aki"])

    caveats = [
        "只报「有几个独立峰支持某元素」，不报「这个峰属于谁」——"
        "在每 nm 20 多条线的库里，单峰归属本质上不可判定。",
        f"本次 {n_amb}/{n_listed} 个（listed）峰的容差窗口里不止一个候选，"
        "这就是归属歧义的实际规模。",
        "尾部概率来自置换检验（把整组峰整体平移到波段里的随机位置重跑 "
        f"{int(n_shift)} 次），不是精确的统计结论，只是「比瞎猜强多少」的度量。",
        "★ 空气元素提示：本系统在空气里打谱，N / O / Ar 的线一定很强"
        "（777.4 nm 的 O I 三线组尤其显眼）。它们被判「有支持」"
        "**不代表样品里含这些元素**。",
        "★ Fe 在库里保留线数最多（3,569 条），偶然命中率约 46%，"
        "所以本方法**基本无法对 Fe 给出结论** —— 要定 Fe 得靠谱线间的强度比。",
        "输出的元素列表是「值得人工复核的候选」，不是「检出的元素」。"
        "最终结论必须以人工核对谱线形状、强度比为准。",
        "本系统数据未经浓度标定，不得据此说含量。",
    ]
    if n_peaks < 5:
        caveats.insert(1, f"本次只分析到 {n_peaks} 个峰，样本太少，置换检验很不稳。")
    if near:
        caveats.insert(
            1,
            f"⚠ 有 {len(near)} 个（listed）峰是「近失配」：容差里只撞到些弱线，"
            f"但 {SL.NEAR_MISS_SPAN_NM} nm 内存在明显更强的线（例如 "
            f"{near[0]['state']} {near[0]['line_wavelength_nm']:.3f} nm，"
            f"差 {near[0]['delta_nm']:.3f} nm 没进窗口）。"
            "这通常意味着**波长标定在这段有偏差**，或者库里缺了那条线 —— "
            "请优先核对这几个峰，别急着当成「没有该元素」。",
        )

    base_out.update({
        "n_peaks": n_peaks,
        "n_peaks_listed": n_listed,
        "n_peaks_ambiguous": n_amb,
        "verdict": (
            f"在 {n_peaks} 个峰里，有 {len(claimed)} 个元素的证据明显超出偶然水平"
            + (f"（其中 {len(claimed_air)} 个是空气元素，不算样品成分）"
               if claimed_air else "")
            if claimed else
            f"{n_peaks} 个峰里没有任何元素的证据明显超出偶然水平"
        ),
        "elements": rows[:max_elements],
        "n_elements_reported": len(rows),
        "peaks": matched,
        "n_peaks_near_miss": len(near),
        "near_miss_peaks": near[:8],
        "caveats": caveats,
    })
    return base_out


def analyze_file(path: str, **kwargs: Any) -> dict[str, Any]:
    """便捷入口：直接给文件路径。"""
    return analyze_elements(load_spectrum(path), **kwargs)
