# -*- coding: utf-8 -*-
r"""
tools.preprocess —— 光谱预处理流水线（纯计算层）。

分层约束（见 tools/__init__.py）：这里只做计算和落盘，
不 print、不调用大模型、不 import agent。

============================ 解决的问题 ============================
原始光谱不能直接拿去比较或定量，原因有三个（都在本机数据上实测确认过）：

  1) **波长网格非均匀**：光栅色散不均匀，本机仪器步长在 0.065~0.142 nm 之间浮动。
     于是「第 500 个点」在两条谱里对应的波长并不是同一个 —— 逐点比较等于比较不同波长。
  2) **有缓慢背景**：连续辐射背景把基线抬起，峰高里混着背景，峰高之比不等于强度之比。
  3) **有随机噪声**：影响弱峰的可信度与峰位判读。

这三件事分别对应 插值 / 基线校正 / 平滑。归一化不是"必须"的，是"可选"的 ——
而且它会改变数据的语义（见下）。

====================== 唯一一个会改变语义的步骤：归一化 ======================
归一化之后，intensity 列**不再是原始计数（counts）**，而是"相对于某个基准的比例"。
后果必须说清楚，否则会被误用：
  · 可以比「形状 / 相对强弱 / 峰高之比」；
  · **不能**再说「这点的绝对强度是多少」——那个信息已经被除掉了；
  · 跨测点做「特征强度图」时，如果每条谱各自归一化，图上反映的是
    「相对自身最亮峰的分布」，而不是「该点的信号有多强」。
所以：**默认不归一化**。要做，必须由调用方显式选，且返回里会带上
intensity_units="relative" 和一条警告，下游报告要照着这个措辞走。

======================== 流水线顺序（固定，不由调用方指定）========================
        插值  →  基线校正  →  平滑  →  归一化
理由写在 config/preprocess.py 的模块注释里（简单说：平滑要求等间隔网格所以插值必须第一；
先平滑再去基线会把峰压矮；归一化的分母必须基于成品而不是半成品）。

============================ 绝 不 外 推 ============================
两组数据波长上限不同（954 nm vs 812 nm）。对齐到统一轴时，
超出一条谱自身覆盖范围的格点**一个都不生成**（既不用 NaN 填，也不重复端点值）。
详细理由见 _uniform_axis 的注释 —— 这是本文件最不能妥协的一条。
"""

from __future__ import annotations

import hashlib
import json
import os
import warnings as _warnings
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Sequence

import numpy as np

from config import preprocess as P
from config.paths import OUTPUT_DIR
from tools.spectrum_loader import Spectrum, load_spectrum

__all__ = [
    "PreprocessError",
    "PreprocessPlan",
    "apply_plan",
    "preprocess_files",
    "list_preprocessed",
    "preprocess_dir",
]


class PreprocessError(Exception):
    """预处理失败时抛出，消息是给人看的中文。"""


# ===========================================================================
# 1) 计划（要做什么 + 参数）
# ===========================================================================
@dataclass
class PreprocessPlan:
    """一条预处理链的全部选择。

    这些都是「方法 + 参数」，**没有顺序** —— 顺序由 PIPELINE_ORDER 固定。
    """

    # --- 插值 ---
    interpolate: str = "uniform"          # none | uniform | common
    grid_step_nm: float | None = None     # 不给就按各自的中位步长（common 模式取最粗的那条）
    grid_range: tuple[float, float] | None = None   # 只在 common 且想手动指定时用

    # --- 基线 ---
    baseline: str = "none"                # none | als | poly
    als_lambda: float = P.ALS_LAMBDA
    als_p: float = P.ALS_P
    als_niter: int = P.ALS_NITER
    poly_order: int = P.POLY_ORDER
    poly_niter: int = P.POLY_NITER
    poly_sigma: float = P.POLY_SIGMA

    # --- 平滑 ---
    # 默认就给 savgol：它是本流水线里唯一"无害"的一步（不改变语义、只压噪声）。
    smooth: str = "savgol"                # none | savgol | moving
    smooth_window_nm: float | None = None # 不给就用 config 里的 0.80 nm
    smooth_polyorder: int = P.SMOOTH_POLYORDER

    # --- 归一化 ---
    normalize: str = "none"               # none | max | area | line
    norm_line_nm: float | None = None     # normalize="line" 时必填
    norm_window_nm: float = P.NORM_LINE_WINDOW_NM

    # ------------------------------------------------------------------
    def validate(self) -> None:
        """参数自检。错就抛中文异常，别让 numpy 抛一句看不懂的英文。"""
        if self.interpolate not in P.INTERP_MODES:
            raise PreprocessError(
                f"interpolate 只能是 {'/'.join(P.INTERP_MODES)}，收到 {self.interpolate!r}")
        if self.baseline not in P.BASELINE_METHODS:
            raise PreprocessError(
                f"baseline 只能是 {'/'.join(P.BASELINE_METHODS)}，收到 {self.baseline!r}")
        if self.smooth not in P.SMOOTH_METHODS:
            raise PreprocessError(
                f"smooth 只能是 {'/'.join(P.SMOOTH_METHODS)}，收到 {self.smooth!r}")
        if self.normalize not in P.NORMALIZE_MODES:
            raise PreprocessError(
                f"normalize 只能是 {'/'.join(P.NORMALIZE_MODES)}，收到 {self.normalize!r}")

        if self.grid_step_nm is not None:
            if not (P.MIN_GRID_STEP_NM <= float(self.grid_step_nm) <= 5.0):
                raise PreprocessError(
                    f"grid_step_nm 要在 {P.MIN_GRID_STEP_NM}~5.0 nm 之间，"
                    f"收到 {self.grid_step_nm}")
        if self.grid_range is not None:
            lo, hi = (float(self.grid_range[0]), float(self.grid_range[1]))
            if not (hi > lo):
                raise PreprocessError(f"grid_range 的上下限反了：{self.grid_range}")

        if self.smooth_window_nm is not None:
            if float(self.smooth_window_nm) < P.MIN_SMOOTH_WINDOW_NM:
                raise PreprocessError(
                    f"smooth_window_nm 不能小于 {P.MIN_SMOOTH_WINDOW_NM} nm"
                    f"（再小就等于没平滑），收到 {self.smooth_window_nm}")
        if int(self.smooth_polyorder) < 1:
            raise PreprocessError("smooth_polyorder 至少要 1")

        if self.normalize == "line":
            if self.norm_line_nm is None:
                raise PreprocessError(
                    "normalize='line' 时必须给 norm_line_nm（拿哪条谱线当分母），"
                    "例如 425.43（Cr I 灵敏线之一）。")
            if float(self.norm_window_nm) <= 0:
                raise PreprocessError("norm_window_nm 必须大于 0")

    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        d = asdict(self)
        if d.get("grid_range") is not None:
            d["grid_range"] = [float(d["grid_range"][0]), float(d["grid_range"][1])]
        return d

    def slug(self) -> str:
        """给输出文件名用的一小段可读摘要 + 6 位参数指纹。

        可读部分让人一眼看出做了什么；指纹保证「同参数必同名、异参数必异名」，
        不会出现两个不同参数的文件互相覆盖。
        """
        d = self.to_dict()
        parts = [
            f"interp-{d['interpolate']}",
            f"base-{d['baseline']}",
            f"smooth-{d['smooth']}",
            f"norm-{d['normalize']}",
        ]
        if d["interpolate"] == "common" and d["grid_step_nm"]:
            parts.append("step%g" % float(d["grid_step_nm"]))
        if d["smooth"] != "none" and d["smooth_window_nm"]:
            parts.append("win%gnm" % float(d["smooth_window_nm"]))
        if d["normalize"] == "line" and d["norm_line_nm"]:
            parts.append("line%gnm" % float(d["norm_line_nm"]))
        fingerprint = hashlib.sha1(
            json.dumps(d, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()[:6]
        return "_".join(parts) + "_" + fingerprint

    def human(self) -> str:
        """一行为人的描述，放进返回值和报告里。"""
        bits = []
        bits.append("插值：不插值" if self.interpolate == "none"
                    else f"插值：{self.interpolate}"
                         + (f"（步长 {self.grid_step_nm:g} nm）" if self.grid_step_nm else ""))
        if self.baseline == "none":
            bits.append("基线：不校正")
        elif self.baseline == "als":
            bits.append(f"基线：ALS（λ={self.als_lambda:g}, p={self.als_p:g}）")
        else:
            bits.append(f"基线：{self.poly_order} 阶多项式")
        if self.smooth == "none":
            bits.append("平滑：不平滑")
        elif self.smooth_window_nm:
            bits.append(f"平滑：{self.smooth}（窗口指定 {self.smooth_window_nm:g} nm）")
        else:
            bits.append(f"平滑：{self.smooth}（窗口自动，最高峰变化上限 "
                        f"±{P.SMOOTH_MAX_PEAK_CHANGE_PCT:g}%）")
        if self.normalize == "none":
            bits.append("归一化：不归一化")
        elif self.normalize == "line":
            bits.append(f"归一化：按 {self.norm_line_nm:g} nm 谱线")
        else:
            bits.append(f"归一化：{self.normalize}")
        return "；".join(bits)


# ===========================================================================
# 2) 基础工具（都是纯函数）
# ===========================================================================
def preprocess_dir() -> str:
    """预处理产物的落盘目录（不存在就建）。"""
    d = os.path.join(OUTPUT_DIR, P.PREPROCESS_DIRNAME)
    os.makedirs(d, exist_ok=True)
    return d


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _median_step(wl: np.ndarray) -> float:
    d = np.diff(wl)
    return float(np.median(d)) if d.size else 0.0


def _is_uniform(wl: np.ndarray, rtol: float = 1e-6) -> bool:
    """步长是否处处相等（相对容差 1e-6）。"""
    d = np.diff(wl)
    if d.size == 0:
        return True
    span = float(d.max() - d.min())
    return bool(span <= rtol * max(1.0, abs(float(d.mean()))))


def _trapezoid_area(x: np.ndarray, y: np.ndarray) -> float:
    """手工梯形积分。

    为什么不直接用 np.trapz？
      NumPy 2.0 把 trapz 改名成了 trapezoid，写哪个都会在某个版本上报警或炸。
      这里就两行，自己算最省心 —— 公式就是 Σ(相邻两点均值 × 间距)。
    """
    if x.size < 2:
        return float(np.sum(y))
    return float(np.sum((y[:-1] + y[1:]) * 0.5 * np.diff(x)))


def _rough_stats(wl: np.ndarray, it: np.ndarray) -> dict:
    """给诊断用的粗统计（和 spectrum_loader.summarize 口径一致，但只取需要的几项）。"""
    base = float(np.percentile(it, 5)) if it.size else 0.0
    idx = int(np.argmax(it)) if it.size else 0
    return {
        "n_points": int(wl.size),
        "wavelength_min_nm": float(wl.min()) if wl.size else None,
        "wavelength_max_nm": float(wl.max()) if wl.size else None,
        "step_median_nm": _median_step(wl),
        "is_uniform": _is_uniform(wl),
        "intensity_max": float(it.max()) if it.size else None,
        "noise_p05": base,
    }


def _uniform_axis(wl: np.ndarray, step_nm: float,
                  grid_range: Sequence[float] | None = None) -> tuple[np.ndarray, dict]:
    """造一条**只落在本谱实际覆盖范围内**的均匀波长轴。

    ★★★ 绝不外推，这是本文件最不能妥协的一条 ★★★

    两组数据的波长上限不同（不锈钢那套到 953.97 nm，快速采集那套只到 812.20 nm）。
    对齐到"统一轴"时，如果为了凑成一样长而把 812 nm 那组补到 954 nm，
    补出来的那些值**纯粹是编的**，而且看起来跟真数据一样平滑、一样像样 ——
    下游的质量检查、元素匹配、出报告全都分辨不出来。这是最危险的一类错误：
    不是报错，而是静默地给出一个假结论。

    所以这里的做法是：先把 [lo, hi] 与 [wl[0], wl[-1]] 求交，
    再在交集上均匀取点。宁可两条谱长度不等，也不外推一个点。
    返回里会带上 requested / actual / n_clipped，让调用方自己知道被裁掉了多少。

    为什么另一个极端（用 NaN 填满再对齐）也不行：
      NaN 会顺着流水线扩散进统计量（max / mean / SNR 全变 NaN），
      下游拿到一串 NaN 只能报错，等于把问题往后推而不是解决。
    """
    lo = float(wl[0])
    hi = float(wl[-1])
    requested = None
    if grid_range is not None:
        rlo, rhi = float(grid_range[0]), float(grid_range[1])
        requested = [rlo, rhi]
        lo = max(lo, rlo)
        hi = min(hi, rhi)

    if not (hi > lo):
        raise PreprocessError(
            f"这条谱的波长覆盖是 {float(wl[0]):.2f}~{float(wl[-1]):.2f} nm，"
            f"与请求的范围 {requested} 没有交集，无法给出任何点（也不会外推）。"
        )

    step_nm = max(float(step_nm), P.MIN_GRID_STEP_NM)
    n = int(np.floor((hi - lo) / step_nm)) + 1
    if n > P.MAX_GRID_POINTS:
        raise PreprocessError(
            f"按 {step_nm:g} nm 步长在 {lo:.2f}~{hi:.2f} nm 上会生成 {n} 个点，"
            f"超过上限 {P.MAX_GRID_POINTS}。请把 grid_step_nm 调大一些。"
        )
    axis = lo + step_nm * np.arange(n, dtype=float)

    # ---- 重采样会不会丢细节？----
    # 本机步长分布（2026-09-21 实测）：
    #   不锈钢数据   p1=0.067  p5=0.069  中位 0.102  p95=0.136  max 0.142 nm
    #   快速采集数据 p1=0.041  p5=0.043  中位 0.079  p95=0.144  max 0.150 nm
    #   （快速采集里那个 0.025 的最小值只有 1 个间隔，是异常值，不能当依据）
    # 步长随波长变化是**波长标定（像素→nm 非线性）**造成的，不是分辨率真的在变。
    # 所以取"中位步长"= 一个新点平均对应一个探测器像素，这是重采样的正解：
    #   用更细的步长 → 只是把曲线插得更密，不会凭空长出信息；
    #   用更粗的步长 → 在原本采样更密的波段确实会丢细节（下面这个比值就是量它）。
    d_src = np.diff(wl)
    if d_src.size:
        densest = float(np.min(d_src))
        merge_ratio = float(step_nm / max(densest, 1e-12))
    else:
        densest, merge_ratio = step_nm, 1.0

    info = {
        "requested_range_nm": requested,
        "actual_range_nm": [float(axis[0]), float(axis[-1])],
        "source_range_nm": [float(wl[0]), float(wl[-1])],
        "clipped": bool(requested is not None
                        and (abs(axis[0] - requested[0]) > 1e-9
                             or abs(axis[-1] - requested[1]) > 1e-9)),
        "step_nm": step_nm,
        "n_points": int(n),
        "extrapolated": False,
        "densest_source_step_nm": round(densest, 5),
        "max_merge_ratio": round(merge_ratio, 4),
    }
    return axis, info


# ---------------------------------------------------------------------------
# 基线：ALS（Eilers & Boelens 非对称最小二乘）
# ---------------------------------------------------------------------------
def _als_baseline(y: np.ndarray, lam: float, p: float, niter: int) -> np.ndarray:
    """非对称最小二乘基线。

    思路：既想贴合数据，又想让基线尽量平滑（惩罚二阶差分），
    同时用一个非对称权重 w —— 明显高于当前基线的点（那就是峰）权重压到 p（很小），
    于是拟合线会从峰的下方"穿过去"，把峰留出来。
    """
    from scipy import sparse
    from scipy.sparse.linalg import spsolve

    n = int(y.size)
    if n < 4:
        return y.astype(float, copy=True)

    # 二阶差分矩阵 D（(n-2) × n），D.T @ D 就是曲率惩罚
    D = sparse.diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(n - 2, n), format="csc")
    curvature = lam * (D.T @ D)

    w = np.ones(n)
    z = y.astype(float, copy=True)
    for _ in range(max(1, int(niter))):
        W = sparse.diags(w, 0, shape=(n, n), format="csc")
        z = spsolve((W + curvature).tocsc(), w * y)
        # 高于基线的点（峰）权重降到 p，低于的留 1-p —— 这就是"非对称"
        w = np.where(y > z, p, 1.0 - p)
    return z


def _poly_baseline(y: np.ndarray, order: int, niter: int, sigma: float) -> np.ndarray:
    """迭代多项式基线（带 σ 剔除）。

    比 ALS 简单，适合背景接近平缓曲线的谱。
    迭代的作用是让拟合逐渐"只认基线点"：每次把明显高于拟合线的点（峰）剔掉再重拟合。
    """
    n = int(y.size)
    if n < 4:
        return y.astype(float, copy=True)

    x = np.arange(n, dtype=float)
    order = int(max(1, min(int(order), n - 2)))
    mask = np.ones(n, dtype=bool)
    coef = np.polyfit(x, y, order)

    with _warnings.catch_warnings():
        # 秩亏时 numpy 会警告"多项式拟合可能不准"，这里用兜底逻辑处理，不必刷屏
        _warnings.simplefilter("ignore")
        for _ in range(max(1, int(niter))):
            if int(mask.sum()) < order + 2:
                break
            coef = np.polyfit(x[mask], y[mask], order)
            fit = np.polyval(coef, x)
            resid = y - fit
            sd = float(np.std(resid[mask]))
            if not np.isfinite(sd) or sd <= 0:
                break
            # 只保留"落在拟合线下方"的点：真正的峰都在基线上方
            new_mask = resid < sigma * sd
            if int(new_mask.sum()) < order + 2 or int(new_mask.sum()) == int(mask.sum()):
                break
            mask = new_mask

    return np.polyval(coef, x)


# ---------------------------------------------------------------------------
# 平滑
# ---------------------------------------------------------------------------
def _window_points(window_nm: float, step_nm: float, polyorder: int, n: int) -> int:
    """把「以 nm 给的窗口宽度」换算成满足 SG 约束的**奇数**点数。

    ★ 为什么用 nm 给、而不是直接给点数：
      这台仪器的像素步长随波长变化（190 nm 处 0.065 nm/像素，780 nm 处 0.133）。
      如果写死"窗口 = 9 点"，那么长波端实际的平滑宽度差不多是短波端的两倍 ——
      同一条谱，不同波长处被磨平的程度不一样，这在物理上说不通。
      给定 nm 由代码换算，才是各处一致的。
      （这条经验直接来自 Step 7：谱线匹配容差也必须跟着像素步长走。）

    SG 的硬约束：window_length 必须是奇数、且大于 polyorder。
    """
    if step_nm <= 0 or n < 3:
        return 0
    w = int(round(float(window_nm) / step_nm))
    if w % 2 == 0:
        w += 1
    w = max(w, int(polyorder) + 2)
    if w % 2 == 0:
        w += 1
    # 不能超过数据长度，且必须奇数
    top = n if n % 2 == 1 else n - 1
    if w > top:
        w = top
    return w if w >= 3 else 0


def _smooth_once(y: np.ndarray, method: str, w: int, polyorder: int) -> np.ndarray:
    if method == "savgol":
        from scipy.signal import savgol_filter
        return np.asarray(
            savgol_filter(y, window_length=int(w), polyorder=int(polyorder), mode="interp"),
            dtype=float)
    if method == "moving":
        from scipy.ndimage import uniform_filter1d
        return np.asarray(uniform_filter1d(y, size=int(w), mode="nearest"), dtype=float)
    raise PreprocessError(f"未知的平滑方法：{method}")


def _peak_change_pct(y: np.ndarray, smoothed: np.ndarray) -> float:
    """平滑前后「全局最高峰」的变化率（%）。

    ★ 符号约定（唯一的定义点，全项目照此理解）：
        正值 = 峰被**压低**（loss）
        负值 = 峰被**抬高**（gain，峰顶小凹口被填平时会这样）
        0    = 峰高不变
    """
    p0 = float(np.max(y))
    if p0 <= 0:
        return 0.0
    return float((p0 - float(np.max(smoothed))) / p0 * 100.0)


def _peak_change_note(change_pct: float) -> str:
    """把变化率翻译成一句人话，避免模型把负号读反。"""
    if abs(change_pct) < 0.05:
        return "最高峰高度基本不变"
    if change_pct > 0:
        return f"最高峰被压低 {change_pct:.2f}%"
    return f"最高峰被抬高 {abs(change_pct):.2f}%"


def _choose_window(y: np.ndarray, method: str, window_nm: float | None,
                   polyorder: int, step_nm: float
                   ) -> tuple[int, dict | None, list[dict]]:
    """决定平滑窗口（换算成点数）。

    调用方给了具体 nm  → 就用它。
    window_nm 是 None（默认）→ **自动选**：候选窗口从小到大挨个试，
      取「全局最高峰变化量 |Δ| ≤ SMOOTH_MAX_PEAK_CHANGE_PCT」的那个**最大**窗口。

    ★ 为什么必须自动，不能给一个固定值：
      本机三条真实谱实测（2026-09-21）—— 同样是 0.8 nm 窗口：
        316 谱（峰 FWHM 5.5~18.5 像素）：峰高损失 −3.3%（几乎无损，好事）
        304 谱（主峰 FWHM 只有 1.6 像素）：峰高损失 −55%（灾难，峰被磨平了）
      峰宽差了 10 倍，固定窗口注定要坑掉其中一边。

    ★ 反直觉但实测成立：**平滑并不总是降低峰值**。316 谱平滑后最高峰反而略升
      （峰顶原本的小凹口被填平）。所以判据是「变化的**绝对值** ≤ 2%」，
      而不是「只许下降」—— 抬高同样是失真，若只卡单边，会出现
      「窗口越大越容易通过」的漏洞，自动选窗就退化成永远返回最大窗口。

    返回 (窗口点数, 选择说明, 候选扫描表)。
    """
    n = int(y.size)
    if n < 5 or step_nm <= 0:
        return 0, None, []

    if window_nm is not None:
        w = _window_points(float(window_nm), step_nm, polyorder, n)
        return (w, {"chosen_by": "调用方指定"}, []) if w > 0 else (0, None, [])

    floor_points = max(P.SMOOTH_MIN_WINDOW_POINTS, int(polyorder) + 2)
    scan: list[dict] = []
    chosen: tuple[int, float, float] | None = None

    for cand in P.SMOOTH_WINDOW_CANDIDATES_NM:
        w = _window_points(float(cand), step_nm, polyorder, n)
        if w < floor_points:
            continue
        change = _peak_change_pct(y, _smooth_once(y, method, w, polyorder))
        ok = bool(abs(change) <= P.SMOOTH_MAX_PEAK_CHANGE_PCT)
        scan.append({"window_nm": float(cand), "window_points": int(w),
                     "peak_change_pct": round(change, 3),
                     "peak_change_note": _peak_change_note(change),
                     "acceptable": ok})
        if ok:
            chosen = (w, float(cand), change)
        # 故意不 break：峰宽差异大时「窗口越大变化越大」并不严格单调，
        # 扫完整张表才能挑到真正最大的那个可行窗口。

    if chosen is None:
        if not scan:                      # 数据太短，连最小窗口都放不下
            return 0, None, scan
        s = scan[0]
        return int(s["window_points"]), {
            "chosen_by": "auto",
            "note": "这条谱的峰太窄，连最小窗口都会明显改峰，已取最小窗口并如实报告代价",
            "peak_change_pct": s["peak_change_pct"],
            "peak_change_note": s["peak_change_note"],
        }, scan

    w, cand, change = chosen
    return int(w), {"chosen_by": "auto", "window_nm": cand,
                    "peak_change_pct": round(change, 3),
                    "peak_change_note": _peak_change_note(change)}, scan


def _smooth(y: np.ndarray, method: str, window_nm: float | None,
            polyorder: int, step_nm: float) -> tuple[np.ndarray, dict]:
    n = int(y.size)
    w, how, scan = _choose_window(y, method, window_nm, polyorder, step_nm)
    if w <= 0:
        return y.astype(float, copy=True), {
            "skipped": True,
            "reason": "数据太短或步长无效，跳过平滑",
        }

    out = _smooth_once(y, method, w, polyorder)
    info = {
        "window_points": int(w),
        "window_nm_requested": (float(window_nm) if window_nm is not None else None),
        "window_nm_effective": float(w * step_nm),
        "polyorder": int(polyorder),
    }
    if how:
        info.update(how)
    if scan:
        # 把候选扫描表一起返回：选了哪个窗口、为什么，是可以被复核的，不是黑箱
        info["window_scan"] = scan
    return out, info


# ---------------------------------------------------------------------------
# 归一化
# ---------------------------------------------------------------------------
def _normalize(y: np.ndarray, x: np.ndarray, mode: str,
               line_nm: float | None, window_nm: float) -> tuple[np.ndarray, dict]:
    if mode == "max":
        divisor = float(np.max(y))
        how = "全谱最大值"
    elif mode == "area":
        divisor = _trapezoid_area(x, y)
        how = "全谱积分面积"
    elif mode == "line":
        assert line_nm is not None
        lo, hi = float(line_nm) - window_nm, float(line_nm) + window_nm
        m = (x >= lo) & (x <= hi)
        n_in = int(m.sum())
        if n_in == 0:
            raise PreprocessError(
                f"归一化指定的谱线 {line_nm:g} nm 不在任何一条谱的覆盖范围内"
                f"（本组谱覆盖 {x[0]:.2f}~{x[-1]:.2f} nm），无法用它当分母。"
            )
        divisor = float(np.max(y[m]))
        how = f"内标线 {line_nm:g} nm 附近峰高"
        extra = {"line_nm": float(line_nm), "window_nm": float(window_nm),
                 "n_points_in_window": n_in,
                 "line_peak_intensity": divisor}
    else:
        raise PreprocessError(f"未知的归一化方式：{mode}")

    if not np.isfinite(divisor) or divisor <= 0:
        raise PreprocessError(
            f"归一化的分母算出来是 {divisor}（用「{how}」），不能除。"
            "通常是这段区间全是 0 或负数，请换一个基准。"
        )

    out = y / divisor
    info = {"divisor": divisor, "based_on": how, "peak_after": float(np.max(out))}
    if mode == "line":
        info.update(extra)
    return out, info


# ===========================================================================
# 3) 主计算：跑完整条链
# ===========================================================================
def apply_plan(spec: Spectrum, plan: PreprocessPlan,
               *, common_step_nm: float | None = None
               ) -> tuple[Spectrum, list[dict], list[str]]:
    """执行「插值 → 基线 → 平滑 → 归一化」。

    返回 (处理后的 Spectrum, 每步诊断, 警告列表)。

    纯计算：不 print、不写盘。
    警告列表是给人和模型看的"必须一起转述的边界"，不是错误 —— 出现警告不代表失败。
    """
    plan.validate()

    wl0 = np.asarray(spec.wavelength, dtype=float)
    it0 = np.asarray(spec.intensity, dtype=float)
    if wl0.size < 8:
        raise PreprocessError(f"光谱只有 {wl0.size} 个点，太少，做不了预处理")

    steps: list[dict] = []
    warns: list[str] = []
    before = _rough_stats(wl0, it0)

    # ---------- 0) 先看原谱状态：饱和是一切后续处理的前提 ----------
    peak0 = float(it0.max())
    if peak0 >= P.ADC_CEILING:
        warns.append(
            f"原谱已打饱和（最强值 {peak0:.0f} ≥ ADC 上限 {P.ADC_CEILING}），峰顶被削平。"
            "后续所有处理都建立在失真的峰上，特别是归一化会直接用一个错误的峰值当分母。"
        )
    elif peak0 >= P.ADC_CEILING * P.NEAR_SATURATION_RATIO:
        warns.append(
            f"原谱接近饱和（最强值 {peak0:.0f}，已达上限的 "
            f"{peak0 / P.ADC_CEILING * 100:.1f}%），峰顶可能已经变钝，慎用它比峰高。"
        )

    cur = Spectrum(wavelength=wl0.copy(), intensity=it0.copy(), meta=dict(spec.meta))

    # ---------- 1) 插值 ----------
    mode = plan.interpolate
    auto_inserted = False
    if mode == "none" and plan.smooth != "none" and not before["is_uniform"]:
        # 平滑（SG 的卷积核）默认假定相邻点间距相等；非均匀网格上算出来的
        # 窗口宽度在两端不一样，等于同一条谱被不同程度地磨平。
        mode = "uniform"
        auto_inserted = True
        warns.append(
            "原谱波长步长非均匀（本机实测 0.065~0.142 nm），而平滑要求等间隔网格，"
            "已自动先做一次均匀插值。"
        )

    if mode != "none":
        if plan.grid_step_nm:
            step = float(plan.grid_step_nm)
        elif mode == "common":
            # 统一轴模式下若没指定步长，取"最粗的那条谱的步长" ——
            # 拿更细的步长去上采样，只会让人误以为分辨率提高了。
            step = float(common_step_nm or _median_step(wl0))
        else:
            step = _median_step(wl0)
        if step <= 0:
            raise PreprocessError("算不出波长步长，无法插值（检查波长列是否全相同）")

        axis, info = _uniform_axis(wl0, step, plan.grid_range)
        new_it = np.interp(axis, wl0, it0)   # 轴已被夹在本谱覆盖范围内 → 不会外推
        cur = Spectrum(wavelength=axis, intensity=new_it, meta=dict(spec.meta))

        after = _rough_stats(axis, new_it)
        info.update({
            "n_points_before": before["n_points"],
            "n_points_after": after["n_points"],
            "step_before_median_nm": before["step_median_nm"],
        })
        steps.append({
            "step": "interpolate",
            "method": mode,
            "params": {"grid_step_nm": step, "grid_range": plan.to_dict()["grid_range"]},
            "effect": info,
            "auto_inserted": auto_inserted,
        })
        if info["clipped"]:
            warns.append(
                f"统一轴被裁剪：请求范围 {info['requested_range_nm']}，"
                f"但这条谱只覆盖 {info['source_range_nm']}，"
                f"实际输出 {info['actual_range_nm']}。"
                "**没有外推**（缺失波段没有生成任何点，也没有用 NaN 填充）。"
            )
        if info["max_merge_ratio"] >= 2.0:
            warns.append(
                f"重采样丢细节：新步长 {info['step_nm']:.4f} nm 是这条谱**最密**采样间隔"
                f"（{info['densest_source_step_nm']:.4f} nm）的 "
                f"{info['max_merge_ratio']:.2f} 倍，在采样最密的波段，一个网格点会合并掉"
                "两个以上的原始点。要保住这些细节，把 grid_step_nm 调小到最密步长附近。"
            )

    # ---------- 2) 基线校正 ----------
    if plan.baseline != "none":
        if plan.baseline == "als":
            base = _als_baseline(cur.intensity, plan.als_lambda, plan.als_p, plan.als_niter)
            params = {"lambda": float(plan.als_lambda), "p": float(plan.als_p),
                      "niter": int(plan.als_niter)}
        else:
            base = _poly_baseline(cur.intensity, plan.poly_order,
                                  plan.poly_niter, plan.poly_sigma)
            params = {"order": int(plan.poly_order), "niter": int(plan.poly_niter),
                      "sigma": float(plan.poly_sigma)}

        corrected = cur.intensity - base
        steps.append({
            "step": "baseline",
            "method": plan.baseline,
            "params": params,
            "effect": {
                "baseline_median": float(np.median(base)),
                "baseline_at_ends": [float(base[0]), float(base[-1])],
                "removed_median": float(np.median(cur.intensity - corrected)),
                "peak_before": float(cur.intensity.max()),
                "peak_after": float(corrected.max()),
                "n_points_going_negative": int((corrected < -1e-9).sum()),
            },
        })
        n_neg = int((corrected < -1e-9).sum())
        if n_neg > 0:
            warns.append(
                f"基线校正后有 {n_neg} 个点的强度变成负数（基线略高于数据）。"
                "这是正常现象（噪声在基线以下），但如果数量很多，说明基线拟合过强了。"
            )
        cur = Spectrum(cur.wavelength, corrected, dict(cur.meta))

    # ---------- 3) 平滑 ----------
    if plan.smooth != "none":
        win = plan.smooth_window_nm      # None = 自动选窗（见 _choose_window）
        step_eff = _median_step(cur.wavelength)
        smoothed, info = _smooth(cur.intensity, plan.smooth, win,
                                 plan.smooth_polyorder, step_eff)
        peak_before = float(cur.intensity.max())
        peak_after = float(smoothed.max())
        change_pct = _peak_change_pct(cur.intensity, smoothed)
        info.update({
            "peak_before": peak_before,
            "peak_after": peak_after,
            "peak_change_pct": round(float(change_pct), 3),
            "peak_change_note": _peak_change_note(change_pct),
        })
        steps.append({
            "step": "smooth",
            "method": plan.smooth,
            "params": {"window_nm": win, "polyorder": int(plan.smooth_polyorder)},
            "effect": info,
        })
        if abs(change_pct) > 5.0:
            warns.append(
                f"平滑后最高峰变化较大：{_peak_change_note(change_pct)}"
                f"（窗口约 {info.get('window_nm_effective', 0):.2f} nm，相对峰宽偏大）。"
                "要拿峰高做比较时请把 smooth_window_nm 调小。"
            )
        cur = Spectrum(cur.wavelength, smoothed, dict(cur.meta))

    # ---------- 4) 归一化 ----------
    if plan.normalize != "none":
        normed, info = _normalize(cur.intensity, cur.wavelength, plan.normalize,
                                  plan.norm_line_nm, plan.norm_window_nm)
        steps.append({
            "step": "normalize",
            "method": plan.normalize,
            "params": {"line_nm": plan.norm_line_nm, "window_nm": plan.norm_window_nm},
            "effect": info,
        })
        warns.append(
            "★ 归一化改变了 intensity 的含义：它已**不是原始计数**，"
            "而是以「" + info["based_on"] + "」为 1 的相对值。"
            "此后只能说「相对强度 / 峰高之比」，"
            "**不能再说某点的绝对强度是多少**，跨测点比也不再反映信号绝对强弱。"
        )
        if plan.normalize == "line" and info.get("line_peak_intensity", 0) >= P.ADC_CEILING:
            warns.append(
                f"用于归一化的内标线（{plan.norm_line_nm:g} nm）本身已打饱和，"
                "拿削平的峰当分母会系统性压缩相对差异，这个归一化结果不可信。"
            )
        cur = Spectrum(cur.wavelength, normed, dict(cur.meta))

    # ---------- 收尾：把全过程写进 meta（溯源用） ----------
    meta = dict(cur.meta)
    meta["preprocess"] = {
        "plan": plan.to_dict(),
        "plan_human": plan.human(),
        "steps": steps,
        "intensity_units": "relative" if plan.normalize != "none" else "counts",
        "n_points_before": int(before["n_points"]),
        "n_points_after": int(cur.wavelength.size),
        "source_path": spec.meta.get("path"),
        "warnings": warns,
    }
    cur.meta = meta
    return cur, steps, warns


# ===========================================================================
# 4) 落盘：CSV（给人看 / 给别的工具读）+ JSON 清单（给机器追溯）
# ===========================================================================
def _write_csv(path: str, wl: np.ndarray, it: np.ndarray) -> None:
    """写两列 CSV，表头固定 wavelength,intensity。

    表头就用这个写法，因为 tools/spectrum_loader.py 认它 ——
    于是预处理产物可以直接再喂回项目里任何一个现有工具。
    """
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("wavelength,intensity\n")
        fh.write("\n".join(f"{x:.5f},{y:.10g}" for x, y in zip(wl, it)))
        fh.write("\n")


def _write_manifest(path: str, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)


# ===========================================================================
# 5) 对外的批量入口（工具层就调这个）
# ===========================================================================
def preprocess_files(paths: Sequence[str], plan: PreprocessPlan,
                     *, save: bool = True) -> dict:
    """对一条或一批光谱跑同一条预处理链，落盘并返回结果摘要。

    ★ 为什么要支持"一批"而不是一条：
      统一轴（interpolate="common"）只有成批才有意义 —— 单条谱的"统一轴"
      就是它自己那条轴。成批时如果没指定 grid_range，会自动取**所有谱覆盖范围的交集**，
      于是各条输出长度一致、可以直接叠成矩阵，而且没有任何一条被外推。
    """
    plan.validate()
    paths = [str(p) for p in paths]
    if not paths:
        raise PreprocessError("paths 不能为空")

    specs: list[Spectrum] = []
    for p in paths:
        specs.append(load_spectrum(p))

    # 统一轴：先算交集，再把同一条步长/范围发给每一条谱
    common_step = None
    common_range = None
    grid_note = None
    if plan.interpolate == "common":
        lo = max(float(s.wavelength[0]) for s in specs)
        hi = min(float(s.wavelength[-1]) for s in specs)
        if not (hi > lo):
            raise PreprocessError(
                "这组谱的波长覆盖范围没有公共交集，无法统一到同一条轴："
                + "；".join(f"{os.path.basename(s.meta.get('path', '?'))} "
                            f"{s.wavelength[0]:.2f}~{s.wavelength[-1]:.2f} nm"
                            for s in specs)
                + "。请改用 interpolate='uniform'（各自成均匀网格，不做跨谱对齐）。"
            )
        common_range = (lo, hi)
        common_step = max(_median_step(s.wavelength) for s in specs)
        grid_note = (
            f"统一轴 = 这 {len(specs)} 条谱覆盖范围的交集 "
            f"{lo:.2f}~{hi:.2f} nm，步长 {common_step:g} nm"
            "（取各谱中位步长里最粗的那个，避免用更细的步长假装提高分辨率）。"
        )
        if plan.grid_range is None:
            plan = PreprocessPlan(**{**plan.to_dict(), "grid_range": [lo, hi]})
    elif plan.interpolate == "uniform":
        grid_note = (
            "每条谱各自重采样成均匀网格（不做跨谱对齐），步长默认取该谱自己的中位步长"
            "（≈一个探测器像素的色散）—— 各条谱的点数因此可能不同。"
            "想跨谱逐点比较（比如叠成矩阵做二维强度图），请改用 interpolate='common'。"
            "若担心在最密的波段丢细节，把 grid_step_nm 设到最密步长附近"
            "（本机不锈钢数据约 0.065 nm、快速采集数据约 0.043 nm）。"
        )

    outdir = preprocess_dir()
    outputs: list[dict] = []
    all_warns: list[str] = []
    units = "counts"

    for spec in specs:
        src_path = str(spec.meta.get("path") or "?")
        proc, steps, warns = apply_plan(spec, plan, common_step_nm=common_step)
        units = proc.meta["preprocess"]["intensity_units"]

        stem = os.path.splitext(os.path.basename(src_path))[0]
        fname = f"{stem[:60]}__{plan.slug()}.csv"
        out_path = os.path.join(outdir, fname)
        man_path = os.path.join(outdir, fname[:-4] + ".json")

        if save:
            _write_csv(out_path, proc.wavelength, proc.intensity)

        manifest = {
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "original_path": src_path,
            "original_sha256": _sha256_file(src_path),
            "original_points": int(spec.wavelength.size),
            "original_range_nm": [float(spec.wavelength[0]), float(spec.wavelength[-1])],
            "plan": plan.to_dict(),
            "plan_human": plan.human(),
            "steps": steps,
            "output_path": out_path,
            "output_points": int(proc.wavelength.size),
            "output_range_nm": [float(proc.wavelength[0]), float(proc.wavelength[-1])],
            "intensity_units": units,
            "warnings": warns,
            "generator": "tools/preprocess.py (libs_agent Step 12)",
        }
        if save:
            _write_manifest(man_path, manifest)

        all_warns.extend(w for w in warns if w not in all_warns)
        outputs.append({
            "original_path": src_path,
            "original_sha256": manifest["original_sha256"],
            "output_path": out_path if save else None,
            "manifest_path": man_path if save else None,
            "n_points_before": manifest["original_points"],
            "n_points_after": manifest["output_points"],
            "range_nm": manifest["output_range_nm"],
            "steps": steps,
            "diagnostics": {
                "peak_before": _rough_stats(spec.wavelength, spec.intensity)["intensity_max"],
                "peak_after": float(proc.intensity.max()),
                "noise_p05_before": _rough_stats(spec.wavelength, spec.intensity)["noise_p05"],
                "noise_p05_after": _rough_stats(proc.wavelength, proc.intensity)["noise_p05"],
            },
        })

    return {
        "ok": True,
        "plan": plan.to_dict(),
        "plan_human": plan.human(),
        "plan_slug": plan.slug(),
        "intensity_units": units,
        "n_spectra": len(outputs),
        "grid_note": grid_note,
        "outputs": outputs,
        "warnings": all_warns,
        "output_dir": outdir,
    }


def list_preprocessed(limit: int = 30) -> dict:
    """列出已经生成过的预处理产物（读 JSON 清单，不读光谱本身）。"""
    d = preprocess_dir()
    items = []
    for name in sorted(os.listdir(d)):
        if not name.endswith(".json"):
            continue
        p = os.path.join(d, name)
        try:
            with open(p, encoding="utf-8") as fh:
                m = json.load(fh)
        except Exception:
            continue
        items.append({
            "manifest_path": p,
            "output_path": m.get("output_path"),
            "original_path": m.get("original_path"),
            "plan_human": m.get("plan_human"),
            "intensity_units": m.get("intensity_units"),
            "n_points": m.get("output_points"),
            "range_nm": m.get("output_range_nm"),
            "created_at": m.get("created_at"),
            "n_warnings": len(m.get("warnings") or []),
        })
    items.sort(key=lambda x: x.get("created_at") or "", reverse=True)
    return {"ok": True, "dir": d, "n_total": len(items), "items": items[:max(1, limit)]}
