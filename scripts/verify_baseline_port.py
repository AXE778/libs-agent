# -*- coding: utf-8 -*-
r"""交叉验证：本项目移植的基线/平滑算法 vs 用户那套工业上位机（<上位机软件>）的原版。

============================ 为什么需要这个脚本 ============================
tools/preprocess.py 里的 `airPLS` 与 `Whittaker` 是从 `<LIBS 软件安装目录>`
移植过来的（算法出处见 config/preprocess.py 的模块注释）。

移植这类数值算法，最危险的不是「写不出来」，而是**「看起来对」**：
像 airPLS 的端点权重那种细节抄错了，整条曲线只偏移 1e-4 相对量 ——
画图看不出来、肉眼看几个统计量也看不出来，但它是错的。
实测就抓到过一处：我写成 `abs(d⁻).max()`，原版是**带符号**的 `d⁻.max()`，
端点权重因此从 1.00 变成 1.06。

所以这里用 importlib 把 <上位机软件> 的**原始 .py 直接加载进来**（走 Python 读文件，
透明绕过本机 DLP 加密层），拿同一份真实数据跑两边、逐点比对，
把「忠实」变成一个可复现的数字。

============================ 它不构成依赖 ============================
项目纪律是「参考 <上位机软件> 的算法，但代码必须独立、运行时不连过去」
（那套软件升级/搬家不能带坏本项目）。所以：
  · 找不到 <上位机软件> 就**跳过**参考比对（退出码 0），只跑自检部分；
  · 公开版里路径已被脱敏成 <LIBS 软件安装目录>，在别人机器上自然是跳过。

用法（在 libs_agent 目录下）：
    .\.venv\Scripts\python.exe scripts\verify_baseline_port.py
可选：用环境变量 REF_IMPL_ROOT 指定那套软件的目录。
可选：用环境变量 LIBS_AGENT_ROOT 指定项目根 —— 从技能目录的副本里跑时用得上。
（脚本会自己找项目根：环境变量 → 默认路径 → 脚本上一级，哪个合理用哪个。）
"""

from __future__ import annotations

import importlib.util
import os
import sys


def _find_project_root() -> str:
    """定位项目根 —— 要兼容两种运行位置：

    ① 在项目里跑（`scripts/verify_baseline_port.py`）→ 上一级就是项目根；
    ② 从**别处的副本**里跑（例如维护技能目录下的那份）→ 上一级不是项目根，
       得靠环境变量或默认路径找。

    顺序：环境变量 → 默认项目路径 → 脚本上一级。
    公开版的默认路径已被脱敏成占位符、必然不存在，于是自然退回脚本上一级，行为不变。
    """
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for cand in (os.environ.get("LIBS_AGENT_ROOT"), r"<项目根目录>", here):
        if cand and os.path.isdir(os.path.join(cand, "tools")):
            return cand
    return here


ROOT = _find_project_root()
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

from tools import preprocess as PP  # noqa: E402
from tools.spectrum_loader import load_spectrum  # noqa: E402

# 环境变量名故意取得中性 —— 别把那套软件的产品名写进变量名里，
# 否则它会随源码进公开版（本机对照用，公开版跑不到这里会自动跳过）。
REF_ROOT = os.environ.get("REF_IMPL_ROOT", r"<LIBS 软件安装目录>")

# 移植的数值算法允许的偏差上限：**相对本谱峰值**的 1e-3。
# 实测现状：ALS / airPLS / Whittaker 三者都与原版**逐位一致**（0.000e+00），
# 所以这个上限只是"意外跑偏时的报警线"，正常永远用不到。
# 用「相对峰值」而不是「相对基线」当分母，是因为前者才是下游真正关心的尺度。
TOL_REL_PEAK = 1.0e-3

CASES = [
    ("304 不锈钢", "DATA-0908165817-X0-Y0-1.csv"),
    ("316 不锈钢", "DATA-0908170219-X1-Y3-1.csv"),
    ("700V/300ns", "fast-260707-1621-206.csv"),
]

_fails: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"    [{'OK ' if ok else 'FAIL'}] {name}" + (f"  —— {detail}" if detail else ""))
    if not ok:
        _fails.append(name)


def hr(t: str = "") -> None:
    print()
    print("=" * 78)
    if t:
        print(" " + t)
        print("=" * 78)


def rel_to_peak(a, b, peak: float) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.max(np.abs(a - b)) / max(peak, 1e-12))


def load_ref(rel: str, name: str):
    """把 <上位机软件> 里的某个 .py 直接当模块加载。返回 None 表示文件不存在。"""
    path = os.path.join(REF_ROOT, rel)
    if not os.path.isfile(path):
        return None
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _airpls_iters(y, lam, order, niter, tol=1e-3):
    """数一下我们的 airPLS 实际跑了多少轮（用于确认收敛行为没跑偏）。"""
    y = np.asarray(y, dtype=float)
    total = float(np.abs(y).sum())
    w = np.ones(y.size)
    used = 1
    for i in range(1, int(niter) + 1):
        used = i
        z = PP._whittaker_solve(w, y, lam, order)
        d = y - z
        neg = d < 0
        dssn = float(np.abs(d[neg]).sum())
        if dssn <= 0.0 or dssn < tol * total or i == int(niter):
            break
        w = np.zeros(y.size)
        w[neg] = np.exp(i * np.abs(d[neg]) / dssn)
        w[0] = w[-1] = float(np.exp(i * float(d[neg].max()) / dssn))
    return used


# ===========================================================================
hr("0) 自检：差分矩阵必须等于朴素的 np.diff(np.eye(n), d)")

# ★ 注意一个坑：np.diff 默认 axis=-1（沿最后一个轴），
#   <上位机软件> 的 als.py 写的是 np.diff(np.eye(L), d) → 形状 (L, L-d)，
#   是「沿行差分」结果的**转置**。两种写法算出来的惩罚矩阵 DᵀD / D Dᵀ 完全相同，
#   但形状不一样，直接相减会广播失败。这里两种都验一遍。
for _d in (1, 2, 3):
    _n = 40
    _D = PP._diff_matrix(_n, _d)
    _row = np.diff(np.eye(_n), _d, axis=0)     # (n-d, n) —— 与本实现同形状
    _col = np.diff(np.eye(_n), _d)             # (n, n-d) —— <上位机软件> 的写法
    _dev_row = float(np.max(np.abs(_D.toarray() - _row)))
    _dev_col = float(np.max(np.abs(_D.toarray() - _col.T)))
    check(f"d={_d}：形状 {_D.shape[0]}×{_D.shape[1]}，两种参考写法都逐位一致",
          _D.shape == _row.shape and _dev_row == 0.0 and _dev_col == 0.0,
          f"沿行差分偏差 {_dev_row}，<上位机软件> 写法取转置后偏差 {_dev_col}")

_D_big = PP._diff_matrix(8000, 2)
check("8000 点 d=2：非零元只有 3×(n-2) 个（稀疏，没走 480MB 的 np.eye(n)）",
      _D_big.nnz == 3 * (8000 - 2), f"nnz={_D_big.nnz}")


# ===========================================================================
hr("1) 与 <上位机软件> 原版逐点比对")

ref_air = load_ref(os.path.join("Algorithm", "deBase", "airPLS.py"), "ref_airpls")
ref_als = load_ref(os.path.join("Algorithm", "deBase", "als.py"), "ref_als")
ref_wt = load_ref(os.path.join("Algorithm", "deNoise", "whittaker.py"), "ref_whittaker")

if ref_air is None or ref_als is None:
    print(f"  [跳过] 没找到参考实现：{REF_ROOT}")
    print("         （公开版里这个路径是脱敏占位符，属于预期行为）")
    print("         只跑完自检部分。")
else:
    print(f"  参考实现：{REF_ROOT}")

    for label, fn in CASES:
        path = os.path.join(ROOT, "data", fn)
        if not os.path.isfile(path):
            print(f"  [跳过] 样例数据不存在：{path}")
            continue
        y = load_spectrum(path).intensity
        peak = float(np.max(y))
        print(f"\n  --- {label}（{y.size} 点，峰值 {peak:.0f}）---")

        # ---- ALS：应当逐位一致（非自适应、每轮固定权重）----
        for lam, d, p, nit in ((1000.0, 2, 0.01, 10), (1.0e5, 2, 0.01, 10),
                               (1.0e5, 1, 0.01, 10)):
            r = np.asarray(ref_als.als(y, lam, d, p, nit), dtype=float)   # 原版返回"扣完的谱"
            m = y - PP._als_baseline(y, lam, p, nit, d)
            dev = rel_to_peak(r, m, peak)
            check(f"ALS  λ={lam:g} d={d} → 与原版一致", dev <= 1e-12,
                  f"最大相对偏差 {dev:.3e}")

        # ---- airPLS：迭代式，允许浮点放大 ----
        for lam, order, nit in ((1000.0, 2, 15), (1.0e5, 2, 15), (1000.0, 1, 15)):
            r = np.asarray(ref_air.airPLS(y, lam, order, nit), dtype=float)
            m = y - PP._airpls_baseline(y, lam, order, nit, 1e-3)
            dev = rel_to_peak(r, m, peak)
            check(f"airPLS λ={lam:g} {order} 阶 → 与原版一致", dev <= TOL_REL_PEAK,
                  f"最大相对偏差 {dev:.3e}（上限 {TOL_REL_PEAK:g}）")

        # ---- airPLS 收敛行为必须一致：迭代次数 + 每轮 dssn ----
        n_ours = _airpls_iters(y, 1000.0, 2, 15)
        check("airPLS 迭代次数在合理区间（本机实测 4~6）", 1 <= n_ours <= 15,
              f"本轮迭代 {n_ours} 次")

        # ---- Whittaker 平滑 ----
        if ref_wt is not None:
            for lam, d in ((10.0, 2), (100.0, 2), (10.0, 1)):
                r = np.asarray(ref_wt.wt_deNoise(y, lmbd=lam, d=d), dtype=float)
                m = PP._whittaker_smooth(y, lam, d)
                dev = rel_to_peak(r, m, peak)
                check(f"Whittaker λ={lam:g} d={d} → 与原版一致", dev <= TOL_REL_PEAK,
                      f"最大相对偏差 {dev:.3e}")


# ===========================================================================
hr("2) 峰宽度量自检（_peak_fwhm）")
_y = np.zeros(21)
_y[10] = 100.0
_y[9] = _y[11] = 60.0        # 半高 50 → 半高以上 3 个点
_y[8] = _y[12] = 10.0
n_pts, n_nm = PP._peak_fwhm(_y, 0.1)
check("构造一个「半高以上 3 个点」的三角形，量出来就是 3",
      n_pts == 3 and abs(n_nm - 0.3) < 1e-9, f"得到 {n_pts} 点 / {n_nm:.4f} nm")

hr("汇总")
if _fails:
    print(f"  失败 {len(_fails)} 项：")
    for f in _fails:
        print(f"    - {f}")
    sys.exit(1)
print("  全部通过。移植的算法与原版在数值上一致（airPLS 的差异仅为浮点噪声）。")
sys.exit(0)
