r"""
config.preprocess —— 预处理流水线的全部默认参数与可选方法（集中一处）。

为什么单独一个文件？
  和 config/quality.py 同一个理由：参数会被反复调，代码不该跟着动。
  想换平滑窗口、换 ALS 强度、换默认归一化方式，只改这里，tools/ 一行都不用碰。

=================== 流水线的固定顺序（不接受 AI 自定义顺序）===================
        插值  →  基线校正  →  平滑  →  归一化

这个顺序是唯一说得通的一条路，理由：
  1) **插值必须排第一**：SG 平滑和多项式基线都**要求等间隔网格**。
     本机仪器步长是 0.065~0.142 nm 的非均匀网格（因为光栅色散不均匀），
     在非均匀网格上直接跑 savgol_filter / polyfit 等于算错 —— 不是误差大小的问题，
     是模型假设被违反。所以先把它变均匀。
  2) **基线要在平滑之前去**：平滑是低通滤波，会把缓慢起伏的背景一起"抹匀"，
     先平滑再去基线，峰会被背景压矮、峰位也可能被拖偏。
  3) **归一化必须排最后**：它的分母（最大值 / 面积 / 内标线峰高）
     必须基于"已经处理好的谱"来算。先归一化再平滑，等于拿半成品的最大值当基准。

把顺序写死，是为了让 AI 只挑「用哪种方法 + 参数多少」，不挑「顺序」。
顺序错了结果既不可复现、也解释不通，而模型没有任何依据去选顺序 ——
给它这个自由度只有坏处。

======================= 一条红线：绝不外推 =======================
两组数据的波长上限不同（不锈钢那套到 954 nm，快速采集那套只到 812 nm）。
统一到同一个轴上时，超出一条谱自身覆盖范围的格点**一个都不生成**，
既不用 NaN、也不重复端点值。理由见 tools/preprocess.py 里 _uniform_axis 的注释。
"""

from __future__ import annotations

# 饱和判据复用 config/quality.py —— 阈值只能有一个出处。
# 自己再写一份 65535，以后换算位数（如 18 位 ADC）必然漏改一处。
from config.quality import ADC_CEILING, NEAR_SATURATION_RATIO

__all__ = [
    "INTERP_MODES", "BASELINE_METHODS", "SMOOTH_METHODS", "NORMALIZE_MODES",
    "DEFAULT_GRID_STEP_NM", "MIN_GRID_STEP_NM", "MAX_GRID_POINTS",
    "SMOOTH_WINDOW_NM", "SMOOTH_WINDOW_CANDIDATES_NM", "SMOOTH_MAX_PEAK_CHANGE_PCT",
    "SMOOTH_MIN_WINDOW_POINTS", "MIN_SMOOTH_WINDOW_NM", "SMOOTH_POLYORDER",
    "ALS_LAMBDA", "ALS_P", "ALS_NITER",
    "POLY_ORDER", "POLY_NITER", "POLY_SIGMA",
    "NORM_LINE_WINDOW_NM", "PREPROCESS_DIRNAME",
    "PIPELINE_ORDER", "ADC_CEILING", "NEAR_SATURATION_RATIO",
]

# ---------------------------------------------------------------- 方法可选值
INTERP_MODES = ("none", "uniform", "common")
BASELINE_METHODS = ("none", "als", "poly")
SMOOTH_METHODS = ("none", "savgol", "moving")
NORMALIZE_MODES = ("none", "max", "area", "line")

# 方法的固定执行顺序（代码里按这个顺序跑，不由调用方指定）
PIPELINE_ORDER = ("interpolate", "baseline", "smooth", "normalize")

# ------------------------------------------------------------------ 插值/网格
# 统一轴的默认步长。0.05 nm 比仪器本来的中位步长（0.0999 nm）更细，
# 属于"上采样"：只让曲线更平滑，不凭空造出新信息 —— 但不能据此声称分辨率提高了。
DEFAULT_GRID_STEP_NM = 0.05
MIN_GRID_STEP_NM = 0.005      # 比这更细没有意义，只会白烧内存
MAX_GRID_POINTS = 120_000     # 单条谱插值后的点数上限（防护，正常用不到）

# -------------------------------------------------------------------- 平滑
# SG 窗口用 **nm** 给，而不是用点数 —— 这台仪器不同波长处的像素步长不一样
# （190 nm 处 0.065 nm/像素，780 nm 处 0.133 nm/像素），写死"窗口 = 9 点"
# 会让长波端实际平滑宽度是短波端的两倍。给定 nm、由代码换算成奇数点数，才物理一致。
# （同一条经验来自 Step 7：谱线匹配容差也必须跟着像素步长走。）

# ★ 为什么默认是"自动选窗口"而不是一个固定 nm 值 ★
# 2026-09-21 在本机三条真实谱上实测了峰宽，结果差得非常远：
#
#   谱                    最强峰 FWHM          0.8 nm 窗口造成的峰高损失
#   DATA-...X0-Y0-1(304)  0.163 nm ≈ 1.6 像素    -55%   ← 灾难级
#   DATA-...X1-Y3-1(316)  0.556~1.883 nm          -3.3% ← 几乎无损
#   fast-260707-1621-206  0.174~0.632 nm          -0.3% ← 无损
#
# 同一个 0.8 nm 窗口，对 316 是好事、对 304 是毁数据。
# 所以**任何固定 nm 值都不可能同时合适** —— 必须按每条谱自己的峰宽来定。
# 做法：把候选窗口从小到大试一遍，取"最高峰变化不超过 SMOOTH_MAX_PEAK_CHANGE_PCT"
# 的那个最大窗口（窗口越大噪声压得越狠，但要保住峰高）。
#
# ⚠ 判据用「变化的绝对值」，不是「只许下降」：
#   实测平滑后最高峰**也会往上走**（304 谱 X1-Y3 那条，窗口越大反而抬高 4.7%），
#   因为峰顶原本的小凹口被填平了。既然抬高同样是失真，就必须一起卡住 ——
#   否则会出现「窗口越大越容易通过」的漏洞，自动选窗永远返回最大窗口。
SMOOTH_WINDOW_NM = None       # None = 自动（推荐）；给具体数字（nm）则强制用它
SMOOTH_WINDOW_CANDIDATES_NM = (0.20, 0.30, 0.40, 0.50, 0.70, 1.00, 1.50)
SMOOTH_MAX_PEAK_CHANGE_PCT = 2.0  # 自动选窗的硬约束：最高峰被压低或抬高都不得超过 2%
SMOOTH_MIN_WINDOW_POINTS = 5     # SG(2 阶) 的最小合法窗口（奇数且 > polyorder）
MIN_SMOOTH_WINDOW_NM = 0.15      # 手动指定时的下限；再小就等于没平滑
SMOOTH_POLYORDER = 2             # SG 多项式阶数；2 阶保峰形，3 阶更激进

# -------------------------------------------------------------- ALS 基线
# Eilers & Boelens 的非对称最小二乘。λ 越大基线越"硬"（越贴直线），
# p 越小越只认"线下方的点"（越不容易把峰当成基线）。
ALS_LAMBDA = 1.0e5
ALS_P = 0.01
ALS_NITER = 10

# ------------------------------------------------------------ 多项式基线
POLY_ORDER = 5                # 阶数太高会追着峰跑，把峰当基线扣掉
POLY_NITER = 8                # 迭代次数（含 σ 剔除）
POLY_SIGMA = 2.0

# -------------------------------------------------------------------- 归一化
# 用内标线归一化时，在目标波长 ±该窗口内找峰顶作为分母。
NORM_LINE_WINDOW_NM = 0.30

# 预处理产物的落盘目录（相对于 config.paths.OUTPUT_DIR）
PREPROCESS_DIRNAME = "preprocessed"
