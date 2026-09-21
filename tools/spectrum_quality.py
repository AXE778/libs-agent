r"""
tools.spectrum_quality —— 单条光谱的采集质量检查（Step 4）。

设计约束（见 tools/__init__.py）：
  纯计算，不 print、不调用大模型、不 import agent。
  输入是 Spectrum 对象（或路径），输出是结构化 dict。

============================ 为什么这一步值得单独做 ============================

前面 Step 3 解决了「能不能把谱读进来」。但**读进来 ≠ 能用**。
真实实验里最常毁掉一批数据的几件事：

  1. **打饱和**：峰值顶到采集卡上限被削平 → 峰高不再正比于信号强度 → 定量必偏。
     而且它很隐蔽：谱线看起来依然是个漂亮的峰，只有看数值才发现顶平了。
  2. **弱信号**：信噪比太低 → 谱线在哪都看不清，更别提比强度。
  3. **宇宙射线尖刺**：1~3 个像素的孤立尖峰，不是谱线，但会被寻峰当谱线。
  4. **背景抬高**：连续辐射/杂散光把基线托起来 → 弱峰被淹。
  5. **波长轴不一致**：换了光谱仪或波段，两条谱的点数和波长对不上，
     直接逐点比较就是错的（必须先把轴对齐）。

前 4 项是「单条谱自身的问题」，第 5 项是「多条谱之间的问题」，所以分两个入口。

============================ 一处必须说清的局限 ============================
尖刺检测是**启发式**的，不是判决。窄的真实谱线和宇宙射线在形状上很像，
所以结果里一律写「疑似」，需要人确认。宁可漏报，不要把它当成确定结论。
"""

from __future__ import annotations

import os
import sys
from typing import Any

import numpy as np

# 本模块要读 config/quality.py 的阈值，所以保证项目根目录在 sys.path 上，
# 这样无论从哪个目录导入都能用。
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from config import quality as Q  # noqa: E402
from .spectrum_loader import Spectrum, load_spectrum  # noqa: E402

__all__ = [
    "rolling_percentile",
    "estimate_noise",
    "estimate_baseline",
    "detect_saturation",
    "detect_spikes",
    "resolve_spikes_by_recurrence",
    "check_quality",
    "check_file",
    "check_many",
    "slim_group_report",
    "compare_axes",
    "check_axis_consistency",
]


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------
def rolling_percentile(x: np.ndarray, window: int, q: float) -> np.ndarray:
    """滚动第 q 百分位（用于估计随波长缓慢变化的基线）。

    为什么不用「减一个常数基线」？
      真实 LIBS 谱的基线不是水平的：短波端和长波端常常差一截。
      用一个滚动窗口内的低百分位数当基线，能跟着基线一起起伏，
      这样在任意波长上算出的「净峰高」才可信。
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    w = int(window)
    if w % 2 == 0:
        w += 1
    if w < 3 or w >= n:
        return np.full(n, float(np.percentile(x, q)))
    pad = w // 2
    padded = np.pad(x, pad, mode="edge")
    win = np.lib.stride_tricks.sliding_window_view(padded, w)
    return np.percentile(win, q, axis=1)


def _robust_sigma(values: np.ndarray) -> float:
    """用 MAD 估标准差：1.4826 * median(|x - median(x)|)。

    为什么不直接用 std？因为光谱里有尖峰（真实谱线 + 宇宙射线），
    它们会把普通标准差拉得很大，算出来的噪声底偏高、信噪比偏低。
    MAD 只看「中位数附近有多散」，对少量极端值不敏感。
    """
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return 0.0
    med = float(np.median(v))
    return float(1.4826 * np.median(np.abs(v - med)))


def estimate_baseline(intensity: np.ndarray) -> dict[str, Any]:
    """估计基线（随波长缓慢变化的那部分）。"""
    y = np.asarray(intensity, dtype=float)
    base = rolling_percentile(y, Q.BASELINE_WINDOW, Q.BASELINE_Q)
    level = float(np.median(base))
    return {
        "level": level,
        "level_ratio_of_ceiling": float(level / Q.ADC_CEILING),
        "curve": base,  # 内部用，报告里不直接输出
    }


def estimate_noise(intensity: np.ndarray, baseline_curve: np.ndarray) -> dict[str, Any]:
    """估计噪声标准差（两路独立估计，互相印证）。

    路 1「安静区残差」：把基线扣掉后，取最安静的那 30% 点算 MAD。
        这是主用值 —— 它直接回答了「没有谱线的地方有多吵」。
    路 2「相邻差分」：一阶差分的 MAD / √2。白噪声下相邻两点独立，
        差的方差是 2σ²。这个估计跟有没有谱线关系不大，适合交叉验证。

    两路差很多时，说明基线估计有问题（比如谱线太密），报告里会标出来。
    """
    y = np.asarray(intensity, dtype=float)
    resid = y - baseline_curve

    quiet_cut = float(np.percentile(resid, 30))
    quiet = resid[resid <= quiet_cut]
    sigma_quiet = _robust_sigma(quiet)

    d = np.diff(y)
    sigma_diff = float(_robust_sigma(d) / np.sqrt(2.0))

    sigma = sigma_quiet if sigma_quiet > 0 else sigma_diff
    agree = None
    if sigma_quiet > 0 and sigma_diff > 0:
        agree = float(max(sigma_quiet, sigma_diff) / min(sigma_quiet, sigma_diff))

    return {
        "sigma_quiet": sigma_quiet,
        "sigma_diff": sigma_diff,
        "sigma_used": float(sigma),
        "agreement_ratio": agree,
        "disagree": bool(agree is not None and agree > 3.0),
    }


def detect_saturation(intensity: np.ndarray, wavelength: np.ndarray,
                      ceiling: float = Q.ADC_CEILING) -> dict[str, Any]:
    """找饱和点，并把连续的饱和点合并成「波段」报出来。

    为什么要报波段而不是只报个数？
      因为「776.5~777.9 nm 有 63 个点顶平了」能直接告诉你**哪条谱线被削掉了**；
      只报「63 个点饱和」你没法判断严重性。
    """
    y = np.asarray(intensity, dtype=float)
    wl = np.asarray(wavelength, dtype=float)

    sat_mask = y >= ceiling
    near_mask = (y >= ceiling * Q.NEAR_SATURATION_RATIO) & ~sat_mask

    ranges = []
    if sat_mask.any():
        idx = np.flatnonzero(np.diff(np.r_[False, sat_mask, False]))
        for start, stop in zip(idx[0::2], idx[1::2]):
            ranges.append({
                "from_nm": float(wl[start]),
                "to_nm": float(wl[stop - 1]),
                "n_points": int(stop - start),
            })

    return {
        "ceiling": float(ceiling),
        "n_saturated": int(sat_mask.sum()),
        "ratio_saturated": float(sat_mask.mean()),
        "n_near_saturated": int(near_mask.sum()),
        "saturated_ranges": ranges,
        "max_intensity": float(y.max()),
        "headroom_ratio": float(y.max() / ceiling),
    }


def detect_spikes(intensity: np.ndarray, wavelength: np.ndarray,
                  sigma: float, peak_height: float,
                  dominant_wl: float | None = None) -> dict[str, Any]:
    """找疑似宇宙射线尖刺。

    ★ 局限（必须说清）：这是启发式，不是判决。
      判据是「偏离局部中位数很多」+「宽度不超过 SPIKE_MAX_WIDTH 个点」。
      **只看单条谱，窄的真实谱线和宇宙射线在形状上就是分不开的** ——
      两者都是「又高又窄的孤立峰」。所以结果里一律写「疑似」，由人确认。

      能分的那一类，我们单独挑出来：落在这条谱主峰波长上的高窄峰，
      判为「就是主峰自己」，计入 dominant_overlaps，不参与「尖刺过多」判定。
      依据是重复性：全组各条谱的主峰都稳定落在同一波长，而宇宙射线不会。
      （更彻底的重复性判定在 check_many 里做。）

    参数 dominant_wl：这条谱主峰的波长（nm）。给了就启用上面那条排除。
    """
    y = np.asarray(intensity, dtype=float)
    wl = np.asarray(wavelength, dtype=float)
    n = y.size

    local_med = rolling_percentile(y, Q.SPIKE_WINDOW, 50.0)
    resid = y - local_med

    thresh = max(Q.SPIKE_SIGMA_K * sigma, Q.SPIKE_MIN_HEIGHT_RATIO * peak_height)
    mask = resid > thresh

    spikes: list[dict[str, Any]] = []
    dominant_overlaps: list[dict[str, Any]] = []
    n_rejected_broad = 0
    if mask.any():
        idx = np.flatnonzero(np.diff(np.r_[False, mask, False]))
        for start, stop in zip(idx[0::2], idx[1::2]):
            width = int(stop - start)
            if width > Q.SPIKE_MAX_WIDTH:
                n_rejected_broad += 1  # 太宽，更像真实谱线，不报
                continue
            j = start + int(np.argmax(y[start:stop]))
            item = {
                "wavelength_nm": float(wl[j]),
                "height": float(y[j]),
                "net_height": float(resid[j]),
                "width_points": width,
            }
            if dominant_wl is not None and abs(item["wavelength_nm"] - dominant_wl) <= Q.SPIKE_DOMINANT_TOL_NM:
                dominant_overlaps.append(item)  # 落在主峰上，判为真谱线
            else:
                spikes.append(item)

    return {
        "n_suspected": len(spikes),
        "threshold": float(thresh),
        "spikes": spikes,
        "n_dominant_overlaps": len(dominant_overlaps),
        "dominant_overlaps": dominant_overlaps,
        "n_rejected_broad": n_rejected_broad,
        "caveat": "疑似而非判决：单条谱里窄的真谱线与宇宙射线形状相似、无法区分；"
                  "本组数据请在组检查里按「是否在多条谱上重复出现」进一步甄别。",
    }


# ---------------------------------------------------------------------------
# 多条谱之间：用「重复性」把疑似尖峰分成 真谱线 / 孤立尖刺
# ---------------------------------------------------------------------------
def resolve_spikes_by_recurrence(
    reports: list[dict[str, Any]],
    ratio: float = Q.SPIKE_RECURRENT_RATIO,
    tol_nm: float = Q.SPIKE_CLUSTER_TOL_NM,
) -> dict[str, Any]:
    """用「多条谱之间的重复性」甄别疑似尖峰。

    ★ 这是本模块唯一能真正区分「窄真谱线」与「宇宙射线」的办法。
      单条谱做不到 —— 两者形状一样（都是又高又窄的孤立峰）。
      但一组谱可以：
        真谱线   —— 同一个扫描点上，每条谱都出现在同一波长 → 会重复
        宇宙射线 —— 每次随机打在 CCD 的不同像素上 → 很少重复
      所以判据是：**某波长上的候选，如果在 >= ratio 比例的谱里都出现，
      就判为真谱线**，从尖刺统计里剔除。

    这一步只能在一组谱上做，所以放在 check_many 里调用。
    """
    n_files = len(reports)
    per_file_raw = [int(r["spikes"]["n_suspected"]) for r in reports]

    if n_files < 2:
        return {
            "n_files": n_files,
            "min_files_to_be_recurrent": None,
            "recurrent_peaks_nm": [],
            "n_recurrent": 0,
            "per_file_isolated": per_file_raw,
            "note": "只有一条谱，无法用重复性甄别，尖刺数按单条谱启发式给出。",
        }

    obs: list[tuple[int, float]] = []
    for i, r in enumerate(reports):
        for s in r["spikes"]["spikes"]:
            obs.append((i, float(s["wavelength_nm"])))

    if not obs:
        return {
            "n_files": n_files,
            "min_files_to_be_recurrent": int(max(2, np.ceil(ratio * n_files))),
            "recurrent_peaks_nm": [],
            "n_recurrent": 0,
            "per_file_isolated": [0] * n_files,
            "note": "本组没有检出任何疑似尖峰。",
        }

    # 按波长聚簇（一维，排序后按间距断开）
    obs.sort(key=lambda t: t[1])
    clusters: list[list[tuple[int, float]]] = []
    cur = [obs[0]]
    for item in obs[1:]:
        if item[1] - cur[-1][1] <= tol_nm:
            cur.append(item)
        else:
            clusters.append(cur)
            cur = [item]
    clusters.append(cur)

    min_files = int(max(2, np.ceil(ratio * n_files)))
    recurrent: list[dict[str, Any]] = []
    rec_ranges: list[tuple[float, float]] = []
    for c in clusters:
        files = {i for i, _ in c}
        if len(files) >= min_files:
            wls = [w for _, w in c]
            recurrent.append({
                "wavelength_nm": float(np.median(wls)),
                "n_files": len(files),
                "span_nm": float(max(wls) - min(wls)),
            })
            rec_ranges.append((min(wls) - tol_nm, max(wls) + tol_nm))

    def _is_recurrent(w: float) -> bool:
        return any(lo <= w <= hi for lo, hi in rec_ranges)

    per_file_isolated = []
    for r in reports:
        per_file_isolated.append(
            sum(1 for s in r["spikes"]["spikes"]
                if not _is_recurrent(float(s["wavelength_nm"])))
        )

    recurrent.sort(key=lambda d: -d["n_files"])
    return {
        "n_files": n_files,
        "min_files_to_be_recurrent": min_files,
        "recurrent_peaks_nm": [d["wavelength_nm"] for d in recurrent],
        "recurrent_peaks": recurrent[:20],
        "n_recurrent": len(recurrent),
        "per_file_isolated": per_file_isolated,
        "note": (
            f"在 >= {min_files} 条谱上重复出现的高窄峰判为真谱线"
            f"（共 {len(recurrent)} 处）；其余才算孤立尖刺。"
        ),
    }


_SPIKE_ISSUE_PREFIX = "疑似孤立尖峰"
_SPIKE_ADVICE = "建议在预处理里加中值滤波/尖刺剔除，或对该点多次累加后重测。"


def _recompute_verdict(r: dict[str, Any]) -> None:
    """按当前 issues 重算判级（打饱和永远优先判 FAIL）。"""
    if r["saturation"]["n_saturated"] > 0:
        r["verdict"] = "FAIL"
        r["summary"] = "存在打饱和，该谱的峰高不可直接用于比较或定量。"
    elif r["issues"]:
        r["verdict"] = "WARN"
        r["summary"] = "未饱和，但存在需要留意的问题。"
    else:
        r["verdict"] = "OK"
        r["summary"] = "未发现明显问题。"


def _apply_recurrence(per_file_isolated: list[int], reports: list[dict[str, Any]]) -> None:
    """把「重复性甄别」的结果回写到每条谱的结论上。

    ★ 为什么必须回写：
      单条谱的检查是在**还不知道全局**的情况下做的，它分不开窄真谱线和宇宙射线，
      会把真谱线也算进尖刺。组检查知道了「哪些波长会重复出现」之后，
      如果不回写，报告里就会自相矛盾 ——
      组级结论说「0 条超标」，逐条明细却说「11 处超标」。模型看到会直接懵。
      所以这里统一口径：**之后一律用甄别后的孤立尖刺数**，判级也跟着重算。
    """
    for r, iso_n in zip(reports, per_file_isolated):
        sp = r["spikes"]
        sp["n_suspected_raw"] = int(sp["n_suspected"])
        sp["n_suspected"] = int(iso_n)

        r["issues"] = [t for t in r["issues"] if not str(t).startswith(_SPIKE_ISSUE_PREFIX)]
        r["advice"] = [t for t in r["advice"] if t != _SPIKE_ADVICE]
        if iso_n > Q.MAX_SPIKES_WARN:
            r["issues"].append(
                f"剔除重复出现的真谱线后，仍有 {iso_n} 处孤立尖峰"
                f"（超过提醒线 {Q.MAX_SPIKES_WARN} 处）。"
            )
            r["advice"].append(_SPIKE_ADVICE)
        _recompute_verdict(r)


# ---------------------------------------------------------------------------
# 主入口：单条谱
# ---------------------------------------------------------------------------
def check_quality(spec: Spectrum, ceiling: float = Q.ADC_CEILING) -> dict[str, Any]:
    """对一条光谱做完整质量检查，返回结构化报告 + 一句话结论。"""
    y = np.asarray(spec.intensity, dtype=float)
    wl = np.asarray(spec.wavelength, dtype=float)
    if y.size < 8:
        raise ValueError("光谱点太少，无法做质量检查")

    base = estimate_baseline(y)
    noise = estimate_noise(y, base["curve"])
    sat = detect_saturation(y, wl, ceiling)

    sigma = noise["sigma_used"]
    resid = y - base["curve"]
    peak_idx = int(np.argmax(resid))
    peak_height = float(resid[peak_idx])
    snr = float(peak_height / sigma) if sigma > 0 else float("inf")

    spikes = detect_spikes(y, wl, sigma, peak_height, dominant_wl=float(wl[peak_idx]))

    above = float(np.mean(resid > 3.0 * sigma)) if sigma > 0 else float("nan")

    # ---- 判级 + 人话说明 ----
    issues: list[str] = []
    advice: list[str] = []

    if sat["n_saturated"] > 0:
        rng = sat["saturated_ranges"][0]
        issues.append(
            f"打饱和：{sat['n_saturated']} 个点顶到采集上限 {int(ceiling)}，"
            f"最宽的一段在 {rng['from_nm']:.2f}~{rng['to_nm']:.2f} nm。"
            "峰顶被削平，峰高不再正比于信号强度。"
        )
        advice.append("降低激光能量、缩短积分时间，或加衰减片后重测这一段。")
    elif sat["n_near_saturated"] > 0:
        issues.append(
            f"近饱和：有 {sat['n_near_saturated']} 个点已达上限的 "
            f"{Q.NEAR_SATURATION_RATIO*100:.0f}% 以上，余量很小。"
        )
        advice.append("建议适当降低信号强度，留出余量以免下次直接打饱和。")

    if snr < Q.SNR_FAIR:
        issues.append(f"弱信号：信噪比约 {snr:.1f}，低于 {Q.SNR_FAIR:.0f}。")
        advice.append("提高积分时间或增加累加次数后重测。")
    elif snr < Q.SNR_GOOD:
        issues.append(f"信噪比偏低：约 {snr:.1f}（可接受线为 {Q.SNR_GOOD:.0f}）。")

    if base["level_ratio_of_ceiling"] > Q.BASELINE_HIGH_RATIO:
        issues.append(
            f"背景偏高：基线中位数约 {base['level']:.0f}，占采集上限的 "
            f"{base['level_ratio_of_ceiling']*100:.1f}%（提醒线 "
            f"{Q.BASELINE_HIGH_RATIO*100:.0f}%）。"
        )
        advice.append("检查环境光/杂散光，或适当提高激光能量以拉开信号与背景的差距。")

    if spikes["n_suspected"] > Q.MAX_SPIKES_WARN:
        issues.append(
            f"疑似孤立尖峰 {spikes['n_suspected']} 处（提醒线：超过 "
            f"{Q.MAX_SPIKES_WARN} 处就提示）。单条谱分不开窄真谱线与宇宙射线；"
            "落在主峰上的高窄峰已单独剔除，剩下的需在组检查里按「是否重复出现」甄别。"
        )
        advice.append("建议在预处理里加中值滤波/尖刺剔除，或对该点多次累加后重测。")

    if noise["disagree"]:
        issues.append(
            f"噪声估计两路不一致（相差 {noise['agreement_ratio']:.1f} 倍），"
            "可能谱线很密导致基线估计偏了，信噪比数值请谨慎参考。"
        )

    if sat["n_saturated"] > 0:
        verdict = "FAIL"
        summary = "存在打饱和，该谱的峰高不可直接用于比较或定量。"
    elif issues:
        verdict = "WARN"
        summary = "未饱和，但存在需要留意的问题。"
    else:
        verdict = "OK"
        summary = "未发现明显问题。"

    return {
        "verdict": verdict,
        "summary": summary,
        "issues": issues,
        "advice": advice,
        "name": spec.meta.get("name", "?"),
        "source": spec.meta.get("path", "?"),
        "n_points": int(y.size),
        "wavelength_range_nm": [float(wl.min()), float(wl.max())],
        "saturation": {k: v for k, v in sat.items()},
        "noise": {k: v for k, v in noise.items() if k != "curve"},
        "baseline": {
            "level": base["level"],
            "level_ratio_of_ceiling": base["level_ratio_of_ceiling"],
        },
        "peak": {
            "wavelength_nm": float(wl[peak_idx]),
            "net_height": peak_height,
            "raw_intensity": float(y[peak_idx]),
            "snr": snr,
        },
        "spikes": spikes,
        "fraction_above_3sigma": above,
        "thresholds_used": {
            "adc_ceiling": ceiling,
            "snr_good": Q.SNR_GOOD,
            "snr_fair": Q.SNR_FAIR,
            "baseline_high_ratio": Q.BASELINE_HIGH_RATIO,
            "max_spikes_warn": Q.MAX_SPIKES_WARN,
            "note": "均为工程经验阈值（config/quality.py 可改），非行业标准。",
        },
    }


def check_file(path: str, ceiling: float = Q.ADC_CEILING) -> dict[str, Any]:
    return check_quality(load_spectrum(path), ceiling=ceiling)


# ---------------------------------------------------------------------------
# 多条谱之间：波长轴一致性
# ---------------------------------------------------------------------------
def compare_axes(specs: list[tuple[str, Spectrum]],
                 tol_nm: float = Q.AXIS_TOL_NM) -> dict[str, Any]:
    """比较多条谱的波长轴是否一致。

    ★ 为什么必须做这一步：
      不同来源（不同光谱仪/不同波段）的谱，点数和波长范围都可能不同。
      **直接按数组下标逐点比较是错的** —— 第 500 个点在这个仪器是 220 nm，
      在另一个仪器可能是 235 nm。必须先统一波长轴（插值到共同网格）再比。
    """
    if not specs:
        raise ValueError("没有给任何光谱")

    ref_label, ref = specs[0]
    ref_wl = np.asarray(ref.wavelength, dtype=float)

    rows = []
    all_same = True
    for label, sp in specs:
        wl = np.asarray(sp.wavelength, dtype=float)
        same_n = wl.size == ref_wl.size
        max_diff = None
        first_bad = None
        if same_n:
            diff = np.abs(wl - ref_wl)
            max_diff = float(diff.max())
            if max_diff > tol_nm:
                first_bad = int(np.argmax(diff > tol_nm))
        else:
            all_same = False
        if not same_n or (max_diff is not None and max_diff > tol_nm):
            all_same = False
        rows.append({
            "label": label,
            "n_points": int(wl.size),
            "range_nm": [float(wl.min()), float(wl.max())],
            "same_n_points_as_ref": bool(same_n),
            "max_diff_nm": max_diff,
            "first_mismatch_index": first_bad,
        })

    return {
        "reference": ref_label,
        "tolerance_nm": float(tol_nm),
        "all_consistent": bool(all_same),
        "spectra": rows,
        "conclusion": (
            "波长轴一致，可以直接逐点比较。"
            if all_same else
            "波长轴不完全一致：**不能**直接按下标逐点比较，"
            "需要先插值到统一波长网格（或只比较共同波段）。"
        ),
    }


def check_axis_consistency(paths: list[str], tol_nm: float = Q.AXIS_TOL_NM) -> dict[str, Any]:
    specs = [(p.split("\\")[-1], load_spectrum(p)) for p in paths]
    return compare_axes(specs, tol_nm=tol_nm)


# ---------------------------------------------------------------------------
# 一组数据：批量检查 + 汇总
# ---------------------------------------------------------------------------
def check_many(
    paths: list[str],
    ceiling: float = Q.ADC_CEILING,
    max_files: int = 60,
    check_axis: bool = True,
    keep_per_file: bool = False,
) -> dict[str, Any]:
    """批量检查一组光谱，汇总出「这组数据整体能不能用」。

    为什么要汇总而不是逐条报？
      做二维扫描时一个材料有 25~150 条谱。逐条报没人看得完。
      真正要回答的是：「这组里有多少条打饱和了？多少条信号太弱？波长轴一致吗？」
      —— 这才决定这组数据能不能进入后面的分析。

    keep_per_file
    -------------
    默认 False，返回里没有逐条明细（给模型看的视图越短越好）。
    传 True 会多出一个 compact 的 `per_file` 列表（每条：路径 / 判级 / 信噪比 /
    饱和点数 / 基线占比 / 孤立尖刺数），**Step 8 出报告和挑「哪几条值得做元素匹配」
    就是靠它** —— 顶层汇总只有数量，挑不出具体哪一条。
    注意它比逐条完整报告轻得多：每条的轴信息和尖峰清单都不在里面。
    """
    if not paths:
        raise ValueError("没有给任何光谱路径")
    if len(paths) > max_files:
        paths = paths[:max_files]

    reports: list[dict[str, Any]] = []
    loaded: list[tuple[str, Spectrum]] = []
    failures: list[dict[str, str]] = []

    for p in paths:
        try:
            spec = load_spectrum(p)
            loaded.append((p.split("\\")[-1], spec))
            reports.append(check_quality(spec, ceiling=ceiling))
        except Exception as exc:
            failures.append({"path": p, "error": f"{type(exc).__name__}: {exc}"})

    if not reports:
        raise ValueError(f"这 {len(paths)} 条全都读失败了，第一条的错误：{failures[0]['error']}")

    def _stat(key_path: tuple[str, ...]) -> dict[str, float]:
        vals = []
        for r in reports:
            v: Any = r
            for k in key_path:
                v = v[k]
            if isinstance(v, (int, float)) and np.isfinite(v):
                vals.append(float(v))
        if not vals:
            return {"min": float("nan"), "median": float("nan"), "max": float("nan")}
        return {"min": min(vals), "median": float(np.median(vals)), "max": max(vals)}

    # 尖刺必须用「重复性」甄别后再统计：单条谱分不开窄真线和宇宙射线，
    # 只有「在绝大多数谱里都出现」的才算真谱线，其余才是孤立尖刺。
    # 甄别结果要回写到每条谱上，保证组级结论与逐条明细口径一致。
    spike_res = resolve_spikes_by_recurrence(reports)
    _apply_recurrence(spike_res["per_file_isolated"], reports)

    sat_files = [r["name"] for r in reports if r["saturation"]["n_saturated"] > 0]
    weak_files = [r["name"] for r in reports if r["peak"]["snr"] < Q.SNR_FAIR]
    bg_files = [r["name"] for r in reports
                if r["baseline"]["level_ratio_of_ceiling"] > Q.BASELINE_HIGH_RATIO]
    spike_files = [r["name"] for r in reports
                   if r["spikes"]["n_suspected"] > Q.MAX_SPIKES_WARN]

    verdict_counts: dict[str, int] = {"OK": 0, "WARN": 0, "FAIL": 0}
    for r in reports:
        verdict_counts[r["verdict"]] = verdict_counts.get(r["verdict"], 0) + 1

    # 峰值波长分布（四舍五入到 0.5 nm），看这组数据是不是稳定打在同一条线上
    hist: dict[str, int] = {}
    for r in reports:
        key = f"{round(r['peak']['wavelength_nm'] * 2) / 2:.1f}"
        hist[key] = hist.get(key, 0) + 1
    peak_hist = sorted(hist.items(), key=lambda kv: -kv[1])

    axis = compare_axes(loaded) if (check_axis and len(loaded) >= 2) else None

    n = len(reports)
    concl = []
    if sat_files:
        concl.append(f"{len(sat_files)}/{n} 条打饱和（峰顶被削平，峰高不可比）")
    if weak_files:
        concl.append(f"{len(weak_files)}/{n} 条信噪比低于 {Q.SNR_FAIR:.0f}（信号弱）")
    if bg_files:
        concl.append(f"{len(bg_files)}/{n} 条背景偏高")
    if spike_files:
        concl.append(
            f"{len(spike_files)}/{n} 条孤立尖刺偏多"
            f"（已先剔掉 {spike_res['n_recurrent']} 处会重复出现的真谱线）"
        )
    if axis is not None and not axis["all_consistent"]:
        concl.append("波长轴不一致，必须先对齐再跨文件比较")
    if not concl:
        concl.append("未发现成组性的质量问题")

    # 最该先看的几条：先 FAIL、再按信噪比从低到高
    order = {"FAIL": 0, "WARN": 1, "OK": 2}
    worst = sorted(reports, key=lambda r: (order.get(r["verdict"], 3), r["peak"]["snr"]))[:5]

    # 逐条明细（可选）。★ 必须在 _apply_recurrence 之后生成 ——
    # 尖刺甄别会把结论回写到每一条谱上，早于它就生成的话，
    # 逐条明细里的 verdict 会和顶层汇总对不上（组里说好了、逐条还说超标）。
    per_file = None
    if keep_per_file:
        per_file = [
            {
                "name": r["name"],
                "path": r["source"],
                "verdict": r["verdict"],
                "snr": round(float(r["peak"]["snr"]), 1),
                "peak_wavelength_nm": round(float(r["peak"]["wavelength_nm"]), 3),
                "n_saturated": int(r["saturation"]["n_saturated"]),
                "baseline_ratio_of_ceiling": round(
                    float(r["baseline"]["level_ratio_of_ceiling"]), 4),
                "n_isolated_spikes": int(r["spikes"]["n_suspected"]),
                "issues": list(r["issues"][:3]),
            }
            for r in reports
        ]

    return {
        "n_requested": len(paths),
        "n_checked": n,
        "n_failed_to_read": len(failures),
        "failures": failures[:5],
        "verdict_counts": verdict_counts,
        "saturated": {
            "n_files": len(sat_files),
            "files": sat_files[:12],
            "max_saturated_points_in_one_file": max(
                (r["saturation"]["n_saturated"] for r in reports), default=0
            ),
        },
        "weak_signal": {"n_files": len(weak_files), "files": weak_files[:12]},
        "high_background": {"n_files": len(bg_files), "files": bg_files[:12]},
        "suspected_spikes": {"n_files": len(spike_files), "files": spike_files[:12]},
        "spike_recurrence": spike_res,
        "snr": _stat(("peak", "snr")),
        "baseline_ratio": _stat(("baseline", "level_ratio_of_ceiling")),
        "peak_wavelength_nm_top": peak_hist[:5],
        "axis_consistency": axis,
        "worst_files": [
            {"name": r["name"], "verdict": r["verdict"], "snr": r["peak"]["snr"],
             "issues": r["issues"][:2]}
            for r in worst
        ],
        "conclusion": "；".join(concl) + "。",
        "per_file": per_file,
        "thresholds_used": reports[0]["thresholds_used"],
    }


def slim_group_report(rep: dict[str, Any], max_names: int = 6,
                      max_worst: int = 3) -> dict[str, Any]:
    """把整组报告压成「模型上下文放得下」的体积（结构不变，只砍明细）。

    ★ 为什么必须做这一步：
      一组 25~150 条谱，逐条明细（每条的轴信息、每条 28 个疑似尖峰）能上千行。
      Agent 把这么长的一段塞给模型，既贵又会让模型抓不住重点；
      更糟的是被下游截断后，模型看到的是一段**断掉的 JSON 字符串**。
      所以这里主动换成摘要：**数量和结论留下，逐条明细砍掉**。

      完整版仍然在 check_many 里能拿到（给脚本/出报告用），这条只是给模型的视图。
    """
    out: dict[str, Any] = dict(rep)

    # 逐条明细是「给脚本出报告用的」，正是这里要砍掉的东西（keep_per_file=True 时才有）
    out.pop("per_file", None)

    ax = out.get("axis_consistency")
    if isinstance(ax, dict):
        rows = ax.get("spectra") or []
        out["axis_consistency"] = {
            "reference": ax.get("reference"),
            "tolerance_nm": ax.get("tolerance_nm"),
            "all_consistent": ax.get("all_consistent"),
            "n_spectra": len(rows),
            "distinct_n_points": sorted({r["n_points"] for r in rows}),
            "inconsistent_labels": [
                r["label"] for r in rows if not r.get("same_n_points_as_ref")
            ][:max_names],
            "conclusion": ax.get("conclusion"),
        }

    sr = out.get("spike_recurrence")
    if isinstance(sr, dict):
        iso = list(sr.get("per_file_isolated") or [])
        out["spike_recurrence"] = {
            "min_files_to_be_recurrent": sr.get("min_files_to_be_recurrent"),
            "n_recurrent": sr.get("n_recurrent"),
            "recurrent_peaks_nm": [round(v, 2) for v in (sr.get("recurrent_peaks_nm") or [])][:10],
            "n_files_with_isolated_over_warn": sum(1 for c in iso if c > Q.MAX_SPIKES_WARN),
            "max_isolated_in_one_file": (max(iso) if iso else 0),
            "note": sr.get("note"),
        }

    worst = out.get("worst_files")
    if isinstance(worst, list):
        out["worst_files"] = [
            {"name": w.get("name"), "verdict": w.get("verdict"), "snr": w.get("snr"),
             "issue": (w.get("issues") or [None])[0]}
            for w in worst[:max_worst]
        ]

    for key in ("saturated", "weak_signal", "high_background", "suspected_spikes"):
        blk = out.get(key)
        if isinstance(blk, dict):
            out[key] = {
                k: (v[:max_names] if isinstance(v, list) else v)
                for k, v in blk.items()
            }
    return out
