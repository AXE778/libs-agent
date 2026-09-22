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

============================== 可用的算法 ==============================
  基线  ：none / als（Eilers 非对称最小二乘）/ poly（迭代多项式 + σ 剔除）/
          airpls（自适应迭代重加权，移植自 <上位机软件> 那套工业上位机 ↔ Zhang 2010）
  平滑  ：none / savgol（Savitzky-Golay）/ moving（滑动平均）/
          whittaker（Eilers 惩罚最小二乘，同样移植自 <上位机软件>）
  归一化：none / max / area / line

★ 几种基线其实共用同一个核（_whittaker_solve，解 (W+λDᵀD)z = W y），
  唯一的差别是权重 w 怎么给：ALS 给固定两档 p / 1-p，airPLS 给逐轮指数权重，
  多项式版换成带 σ 剔除的迭代拟合。别被几个名字唬住。
★ 来源对照、以及「为什么不照抄 <上位机软件> 的执行顺序」，写在 config/preprocess.py 里。

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
    baseline: str = "none"                # none | als | airpls | poly
    als_lambda: float = P.ALS_LAMBDA
    als_p: float = P.ALS_P
    als_niter: int = P.ALS_NITER
    als_order: int = P.ALS_ORDER          # 差分阶数 d（<上位机软件> 的 als(data, lam, d, …) 也公开它）
    airpls_lambda: float = P.AIRPLS_LAMBDA
    airpls_order: int = P.AIRPLS_ORDER
    airpls_niter: int = P.AIRPLS_NITER
    airpls_tol: float = P.AIRPLS_TOL
    poly_order: int = P.POLY_ORDER
    poly_niter: int = P.POLY_NITER
    poly_sigma: float = P.POLY_SIGMA

    # --- 平滑 ---
    # 默认就给 savgol：它是本流水线里唯一"无害"的一步（不改变语义、只压噪声）。
    smooth: str = "savgol"                # none | savgol | moving | whittaker
    smooth_window_nm: float | None = None # 不给（None）= **自动选强度**，见 _choose_smoother()
    smooth_polyorder: int = P.SMOOTH_POLYORDER
    whittaker_lambda: float | None = None # 只在 smooth="whittaker" 时用；None（默认）= 自动选 λ
    whittaker_order: int = P.WHITTAKER_ORDER   # Whittaker 惩罚的差分阶数 d

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

        # 三个"差分阶数"共用一条约束（1 阶=斜率，2 阶=曲率，再高基线会软到跟着背景跑）
        for _name, _val in (("als_order", self.als_order),
                            ("airpls_order", self.airpls_order),
                            ("whittaker_order", self.whittaker_order)):
            if not (1 <= int(_val) <= 4):
                raise PreprocessError(
                    f"{_name} 要在 1~4 之间（惩罚几阶差分），收到 {_val}")
        if self.whittaker_lambda is not None and float(self.whittaker_lambda) <= 0:
            raise PreprocessError(
                f"whittaker_lambda 必须大于 0（它是平滑强度），收到 {self.whittaker_lambda}")
        if int(self.airpls_niter) < 1:
            raise PreprocessError("airpls_niter 至少要 1")
        if float(self.airpls_tol) <= 0:
            raise PreprocessError("airpls_tol 必须大于 0")

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
            bits.append(f"基线：ALS（λ={self.als_lambda:g}, p={self.als_p:g}, "
                        f"{self.als_order} 阶差分）")
        elif self.baseline == "airpls":
            bits.append(f"基线：airPLS（λ={self.airpls_lambda:g}, "
                        f"{self.airpls_order} 阶差分）")
        else:
            bits.append(f"基线：{self.poly_order} 阶多项式")
        if self.smooth == "none":
            bits.append("平滑：不平滑")
        elif self.smooth == "whittaker":
            if self.whittaker_lambda:
                bits.append(f"平滑：Whittaker（λ={self.whittaker_lambda:g}）")
            else:
                bits.append("平滑：Whittaker（λ 自动，最高峰变化上限 "
                            f"±{P.SMOOTH_MAX_PEAK_CHANGE_PCT:g}%）")
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
# 基线的共同内核：Whittaker 惩罚最小二乘
# ---------------------------------------------------------------------------
def _diff_matrix(n: int, order: int):
    """d 阶差分矩阵 D（形状 (n-d) × n），稀疏。

    构造方式与 <上位机软件> 的 __WhittakerSmooth / __speyediff 一致：
    从稀疏单位阵出发，反复做 D = D[1:] - D[:-1]。

    ★ 为什么不用 np.diff(np.eye(n), d)：
      <上位机软件> 的 als.py 就是那么写的，但 np.eye(7745) 是 7745² 个 float64
      ≈ 480 MB —— 单次调用就吃掉半个 G。稀疏逐次差分得到的矩阵完全一样，
      内存却是 O(n·d)。
    """
    from scipy import sparse

    n = int(n)
    order = int(order)
    if order <= 0 or n <= order:
        return None
    D = sparse.eye(n, format="csc")
    for _ in range(order):
        D = D[1:] - D[:-1]
    return D


def _whittaker_solve(w: np.ndarray, y: np.ndarray, lam: float, order: int) -> np.ndarray:
    """解 (W + λ·DᵀD) z = W y —— ALS / airPLS / Whittaker 平滑共用的一个内核。

    三个方法只差在 w 怎么给：
      · ALS / airPLS：w 是「峰 / 非峰」的权重，于是解偏向从峰的下方穿过；
      · Whittaker 平滑：w 全 1，于是就是一个纯平滑器。
    """
    from scipy import sparse
    from scipy.sparse.linalg import spsolve

    y = np.asarray(y, dtype=float)
    n = int(y.size)
    D = _diff_matrix(n, order)
    if D is None:
        return y.copy()
    w = np.asarray(w, dtype=float)
    W = sparse.diags(w, 0, shape=(n, n), format="csc")
    A = (W + float(lam) * (D.T @ D)).tocsc()
    return np.asarray(spsolve(A, w * y))


# ---------------------------------------------------------------------------
# 基线 1：ALS（Eilers & Boelens 非对称最小二乘）
# ---------------------------------------------------------------------------
def _als_baseline(y: np.ndarray, lam: float, p: float, niter: int,
                  order: int = P.ALS_ORDER) -> np.ndarray:
    """非对称最小二乘基线。

    思路：既想贴合数据，又想让基线尽量平滑（惩罚 d 阶差分），
    同时用一个非对称权重 w —— 明显高于当前基线的点（那就是峰）权重压到 p（很小），
    于是拟合线会从峰的下方「穿过去」，把峰留出来。

    d（差分阶数）从 <上位机软件> 的 als(data, lam, d, p, niter) 对齐过来：
    d=1 惩罚斜率、d=2 惩罚曲率（默认）。阶数越高基线越"软"、越容易跟着缓变背景起伏。
    """
    y = np.asarray(y, dtype=float)
    if y.size < 4:
        return y.copy()

    w = np.ones(int(y.size))
    z = y.copy()
    for _ in range(max(1, int(niter))):
        z = _whittaker_solve(w, y, lam, order)
        # 高于基线的点（峰）权重降到 p，低于的留 1-p —— 这就是「非对称」
        w = np.where(y > z, p, 1.0 - p)
    return z


# ---------------------------------------------------------------------------
# 基线 2：airPLS（自适应迭代重加权，移植自 <上位机软件> ↔ Zhang et al. 2010）
# ---------------------------------------------------------------------------
def _airpls_baseline(y: np.ndarray, lam: float, order: int, niter: int,
                     tol: float) -> np.ndarray:
    """airPLS 基线（返回**基线本身**，不是扣完的谱）。

    与 ALS 用同一个惩罚最小二乘内核，差别只在**权重更新规则**：
      · 高于基线的点（峰）权重直接置 0 —— 比 ALS 的 p=0.01 更彻底地「看不见」峰；
      · 低于基线的点权重按 exp(i·|d⁻| / Σ|d⁻|) 逐轮**指数**加重（i 是轮次）；
      · 两端点单独钉住（见下）。
    停机判据：负残差总量已经很小（|Σd⁻| < tol·Σ|y|），也就是基线贴着数据下沿了。
    收敛比 ALS 快：本机真实谱实测 4~5 轮就触发停机。

    ★ 端点那一行是本算法最容易抄错的地方。
      <上位机软件> 写的是 w[0] = exp(i · d⁻.max() / Σ|d⁻|)，其中 d⁻ 是**带符号的**负残差，
      所以 d⁻.max() 取的是「最接近 0 的那个负值」（仍是负数）→ 指数为负 → 权重 < 1
      → 两端被拉向基线。
      若写成 np.abs(d⁻).max()（幅值最大的那个负残差），指数会翻成正的、权重 > 1，
      端点反而被允许往上抬。实测两种写法最终权重差 0.062、整条基线差 ~1e-5 相对量 ——
      画图看不出来，但那是抄错，不是等价变形。改回带符号写法之后，
      本实现与 <上位机软件> 原版**逐位一致**（复现脚本：scripts/verify_baseline_port.py）。
    """
    y = np.asarray(y, dtype=float)
    n = int(y.size)
    if n < 4:
        return y.copy()

    w = np.ones(n)
    z = y.copy()
    total = float(np.abs(y).sum())
    for i in range(1, int(niter) + 1):
        z = _whittaker_solve(w, y, lam, order)
        d = y - z
        neg = d < 0
        dssn = float(np.abs(d[neg]).sum())
        # dssn == 0：没有任何点低于基线，权重失去依据；再往下就是除零。
        # （<上位机软件> 只判 dssn < tol·total，而 0 < 0 为假 → 极端输入下会除零。
        #   这里补上 dssn <= 0 这一条，其余判据与它逐条一致。）
        if dssn <= 0.0 or dssn < float(tol) * total or i == int(niter):
            break
        w = np.zeros(n)
        w[neg] = np.exp(i * np.abs(d[neg]) / dssn)
        w[0] = w[-1] = float(np.exp(i * float(d[neg].max()) / dssn))
    return z


# ---------------------------------------------------------------------------
# 平滑 0：Whittaker 平滑（移植自 <上位机软件> ↔ Eilers 2003 "A perfect smoother"）
# ---------------------------------------------------------------------------
def _whittaker_smooth(y: np.ndarray, lam: float, order: int) -> np.ndarray:
    """Whittaker 平滑：w 全 1 的惩罚最小二乘解，即 min Σ(y-z)² + λ‖D_d z‖²。

    没有「窗口」概念，λ 是唯一的强度旋钮。与 SG 的差别、以及
    「它同样救不了窄峰」的实测数据，见 config/preprocess.py 里 WHITTAKER_LAMBDA 那段。
    """
    y = np.asarray(y, dtype=float)
    if y.size < 4:
        return y.copy()
    return _whittaker_solve(np.ones(int(y.size)), y, lam, order)


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


def _peak_fwhm(y: np.ndarray, step_nm: float) -> tuple[int, float]:
    """全局最高峰的半高全宽，用「半高以上的连续点数」度量（返回 (点数, nm)）。

    ★ 为什么值得专门算出来返回：
      实测（2026-09-22）模型在被问「峰有没有被改」时，会**自己编一个峰宽** ——
      一次回答说「这条谱的峰非常窄（约 1.3 个像素宽）」，而真实值是 2 个点。
      与其让它猜，不如算给它。这正是本项目「绝不让模型做算术」那条纪律的延伸。

    用点数而不是去拟合一个 FWHM：本机像素步长 0.1 nm 量级，而我们要区分的峰宽
    只差个位数像素，拟合出来的小数位是假精度。
    """
    y = np.asarray(y, dtype=float)
    if y.size == 0:
        return 0, 0.0
    k = int(np.argmax(y))
    half = float(y[k]) / 2.0
    if half <= 0:
        return 0, 0.0
    lo = k
    while lo - 1 >= 0 and y[lo - 1] >= half:
        lo -= 1
    hi = k
    while hi + 1 < y.size and y[hi + 1] >= half:
        hi += 1
    n = int(hi - lo + 1)
    return n, float(n * step_nm)


def _smooth_candidate(y: np.ndarray, method: str, cand: float,
                      polyorder: int, step_nm: float
                      ) -> tuple[np.ndarray | None, dict | None]:
    """按一个候选强度做一次平滑。

    cand 的含义随方法而变：
      savgol / moving → 窗口宽度（nm）
      whittaker       → λ
    候选不合法（窗口放不下 / λ 非正）时返回 (None, None)。
    """
    if method == "whittaker":
        lam = float(cand)
        if lam <= 0:
            return None, None
        return _whittaker_smooth(y, lam, polyorder), {
            "lambda": lam, "order": int(polyorder)}

    w = _window_points(float(cand), step_nm, polyorder, int(y.size))
    floor_points = max(P.SMOOTH_MIN_WINDOW_POINTS, int(polyorder) + 2)
    if w < floor_points:
        return None, None
    return _smooth_once(y, method, w, polyorder), {
        "window_nm": float(cand), "window_points": int(w)}


def _choose_smoother(y: np.ndarray, method: str, window_nm: float | None,
                     wt_lambda: float | None, polyorder: int, step_nm: float
                     ) -> tuple[np.ndarray | None, dict | None, list[dict]]:
    """挑平滑强度，顺便把平滑做掉。返回 (平滑后的谱, 选择说明, 候选扫描表)。

    两种方法族的「强度」量纲不同（SG 是窗口 nm，Whittaker 是 λ），
    但**选择判据是同一条**：候选强度从小到大挨个试，取
      「全局最高峰变化量 |Δ| ≤ SMOOTH_MAX_PEAK_CHANGE_PCT」的那个**最大**强度。

    ★ 为什么必须自动选，不能写死一个强度：
      本机真实谱实测（2026-09-22）—— 同样是 SG 最小窗（5 点）：
        316 谱（主峰半高以上 9 个点）：峰高变化 −1.0%（几乎无损，好事）
        700V 谱（半高以上 8 个点）：   −1.9%（可接受）
        304 谱（652.38 nm 主峰半高以上只有 1 个点）：+28.0%（灾难，峰被抹平）
      峰宽差了近一个数量级，写死强度注定要坑掉其中一边。

      ⚠ 2026-09-22 更正两处早年记错的数：
      （a）之前把 304 那个峰记成「FWHM≈1.6 像素的窄峰」，还怀疑过它可能是宇宙射线
          （单像素尖刺形状和窄谱线一模一样，单条谱分不开）。这次拿 data-304-1
          整组 25 条谱查了重复性 —— 25/25 条在该波长都有峰、峰顶波长标准差仅 0.030 nm、
          峰/肩比值中位 4.13（且多数已打饱和）→ **它是一条真实的、正好卡在仪器
          分辨率极限上的分析线**，不是坏点。也就是说这个谱的平滑损失是真损失，
          不能用「那是假峰」把它打发掉。
      （b）峰宽**随网格而变**：同一条线在原始网格上半高以上是 2 个点，
          插值到 0.102 nm 均匀网格后只剩 1 个点 —— 重采样本身就把峰顶削平了一点。
          而最小 SG 窗是 5 个点，所以这个损失在流水线里是躲不掉的。
          下面报的 peak_fwhm_* 量的都是「平滑前的那一刻、即插值后的网格」，口径统一。

    ★ 反直觉但实测成立：**平滑并不总是降低峰值**。304 谱平滑后最高峰反而明显抬高
      （峰顶原本的小凹口被填平）。所以判据是「变化的**绝对值** ≤ 2%」，
      而不是「只许下降」—— 抬高同样是失真，若只卡单边，会出现
      「强度越大越容易通过」的漏洞，自动选强就退化成永远返回最大强度。
    """
    n = int(y.size)
    if n < 5 or step_nm <= 0:
        return None, None, []

    explicit = wt_lambda if method == "whittaker" else window_nm
    if explicit is not None:
        out, info = _smooth_candidate(y, method, explicit, polyorder, step_nm)
        if out is None:
            return None, None, []
        return out, {"chosen_by": "调用方指定", **info}, []

    cands = (P.WHITTAKER_LAMBDA_CANDIDATES if method == "whittaker"
             else P.SMOOTH_WINDOW_CANDIDATES_NM)

    scan: list[dict] = []
    valid: list[tuple[float, np.ndarray, dict, float]] = []
    for cand in cands:
        out, info = _smooth_candidate(y, method, cand, polyorder, step_nm)
        if out is None:
            continue
        change = _peak_change_pct(y, out)
        ok = bool(abs(change) <= P.SMOOTH_MAX_PEAK_CHANGE_PCT)
        scan.append({**info, "peak_change_pct": round(change, 3),
                     "peak_change_note": _peak_change_note(change),
                     "acceptable": ok})
        valid.append((float(cand), out, info, change))
        # 故意不 break：峰宽差异大时「强度越大变化越大」并不严格单调，
        # 扫完整张表才能挑到真正最大的那个可行强度。

    if not valid:                      # 数据太短，连最小强度都放不下
        return None, None, scan

    passed = [v for v in valid if abs(v[3]) <= P.SMOOTH_MAX_PEAK_CHANGE_PCT]
    if not passed:
        # 这条谱的峰太窄：连最小强度都会明显改峰。
        # 现在仍按最小强度平滑，但把代价**如实写进返回**（不静默、不假装无损）。
        # 「要不要干脆跳过平滑」是一条策略选择，见 config/preprocess.py 的说明。
        cand, out, info, change = valid[0]
        return out, {
            "chosen_by": "auto",
            "note": "这条谱的峰太窄，连最小强度都会明显改峰，已取最小强度并如实报告代价",
            **info,
            "peak_change_pct": round(change, 3),
            "peak_change_note": _peak_change_note(change),
        }, scan

    cand, out, info, change = passed[-1]
    return out, {"chosen_by": "auto", **info,
                 "peak_change_pct": round(change, 3),
                 "peak_change_note": _peak_change_note(change)}, scan


def _smooth(y: np.ndarray, method: str, window_nm: float | None,
            wt_lambda: float | None, polyorder: int, step_nm: float
            ) -> tuple[np.ndarray, dict]:
    smoothed, how, scan = _choose_smoother(y, method, window_nm, wt_lambda,
                                           polyorder, step_nm)
    if smoothed is None:
        return y.astype(float, copy=True), {
            "skipped": True,
            "reason": "数据太短、步长无效，或该方法的候选强度全都放不下，跳过平滑",
        }

    info: dict = {**how}
    if method != "whittaker":
        # Whittaker 没有窗口概念，这两个字段对它无意义，不写（避免下游读到一个假的 0 nm）
        info["window_nm_requested"] = (float(window_nm) if window_nm is not None else None)
        w = info.get("window_points")
        info["window_nm_effective"] = float(w * step_nm) if w else 0.0
        # 注意：对 savgol/moving，polyorder 是 SG 的多项式阶数；
        #       对 whittaker，同一个入参是惩罚差分的阶数，见 _smooth_candidate 里的 order 字段。
        info["polyorder"] = int(polyorder)
    if scan:
        # 把候选扫描表一起返回：选了哪个强度、为什么，是可以被复核的，不是黑箱
        info["window_scan"] = scan
    return smoothed, info


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
            base = _als_baseline(cur.intensity, plan.als_lambda, plan.als_p,
                                 plan.als_niter, plan.als_order)
            params = {"lambda": float(plan.als_lambda), "p": float(plan.als_p),
                      "niter": int(plan.als_niter), "order": int(plan.als_order)}
        elif plan.baseline == "airpls":
            base = _airpls_baseline(cur.intensity, plan.airpls_lambda,
                                    plan.airpls_order, plan.airpls_niter,
                                    plan.airpls_tol)
            params = {"lambda": float(plan.airpls_lambda),
                      "order": int(plan.airpls_order),
                      "niter": int(plan.airpls_niter),
                      "tol": float(plan.airpls_tol)}
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
        win = plan.smooth_window_nm      # None = 自动选强度（见 _choose_smoother）
        # SG 的「多项式阶数」与 Whittaker 的「惩罚差分阶数」是两个概念、各有默认值，
        # 在这里解析成最终要传下去的那一个。
        eff_order = (plan.whittaker_order if plan.smooth == "whittaker"
                     else plan.smooth_polyorder)
        step_eff = _median_step(cur.wavelength)
        smoothed, info = _smooth(cur.intensity, plan.smooth, win,
                                 plan.whittaker_lambda, eff_order, step_eff)
        peak_before = float(cur.intensity.max())
        peak_after = float(smoothed.max())
        change_pct = _peak_change_pct(cur.intensity, smoothed)
        fwhm_pts, fwhm_nm = _peak_fwhm(cur.intensity, step_eff)
        info.update({
            "peak_before": peak_before,
            "peak_after": peak_after,
            "peak_change_pct": round(float(change_pct), 3),
            "peak_change_note": _peak_change_note(change_pct),
            # 峰宽一起给出来：它是「为什么这个窗口会把峰削掉」的直接依据，
            # 也堵住模型自己编一个峰宽的漏洞（见 _peak_fwhm 的注释）。
            "peak_fwhm_points": fwhm_pts,
            "peak_fwhm_nm": round(fwhm_nm, 4),
        })
        steps.append({
            "step": "smooth",
            "method": plan.smooth,
            "params": ({"lambda": plan.whittaker_lambda, "order": int(eff_order)}
                       if plan.smooth == "whittaker"
                       else {"window_nm": win, "polyorder": int(eff_order)}),
            "effect": info,
        })
        if abs(change_pct) > 5.0:
            knob = ("whittaker_lambda" if plan.smooth == "whittaker"
                    else "smooth_window_nm")
            scale = (f"λ={info['lambda']:g}" if plan.smooth == "whittaker"
                     else f"窗口约 {info.get('window_nm_effective', 0):.2f} nm")
            warns.append(
                f"平滑后最高峰变化较大：{_peak_change_note(change_pct)}"
                f"（{scale}，相对峰宽偏大）。"
                f"要拿峰高做比较时请把 {knob} 调小。"
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
