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

============ 与 <上位机软件>（用户那套工业上位机）的关系（2026-09-22 对照）============
对照了 <LIBS 软件安装目录> 的预处理源码。结论一句话：
**算法全部对齐，顺序故意不同。**

取它有的（因为那套软件就是现场在跑的东西，结果要和它对得上）：
  · 基线 **airPLS**（Zhang et al. 2010）—— 它的「基线校正」下拉框就两项 airPLS/als，
    而且默认项就是 airPLS（λ=1000、阶数=2）。见 tools/preprocess.py 的 _airpls_baseline；
  · 基线 ALS 的**差分阶数**参数 —— 它的 als(data, lam, d, p, niter) 是公开 d 的；
  · 平滑 **Whittaker**（Eilers 2003 "A perfect smoother"）—— 它的 wt_deNoise。
    （它的平滑是 SG / WT / EMD，SG 我们本来就有；EMD 要额外依赖，没取。）

不取它的（都是有意为之，理由在下面）：
  · **执行顺序**。<上位机软件> 是「基线 → 插值 → 平滑」。我们是「插值 → 基线 → 平滑」，
    理由见上面第 1 条：ALS/airPLS 的 λ 惩罚的是**相邻采样点的差分**，
    所以同一个 λ 在 0.065 nm 网格和 0.142 nm 网格上的物理平滑宽度完全不同。
    本机步长随波长从 0.065 变到 0.142 nm，若先基线后插值，λ 对每条谱的含义都不同、
    跨谱不可比 —— 这与本文件「平滑窗口用 nm 给、而不是给点数」是同一条道理。
    （代价：λ 的绝对值与 <上位机软件> 面板上填的那个数不能直接互抄，需按本网格重标定。）
  · 它的 polynomial.py：全局 polyfit，峰会把拟合线整个拽上去。
    我们保留自己的迭代 σ 剔除版（_poly_baseline）。
  · 它的 MSC / SNV / max_min：MSC 需要"一组谱的平均谱"当参考，
    是批处理概念而不是单谱变换；SNV 与 standardization 两份实现**逐字节相同**
    （同一个公式抄了两遍），我们没跟着抄这份重复。
    （归一化我们给的是 max / area / line，语义见下。）"""

from __future__ import annotations

# 饱和判据复用 config/quality.py —— 阈值只能有一个出处。
# 自己再写一份 65535，以后换算位数（如 18 位 ADC）必然漏改一处。
from config.quality import ADC_CEILING, NEAR_SATURATION_RATIO

__all__ = [
    "INTERP_MODES", "BASELINE_METHODS", "SMOOTH_METHODS", "NORMALIZE_MODES",
    "MIN_GRID_STEP_NM", "MAX_GRID_POINTS",
    "SMOOTH_WINDOW_NM", "SMOOTH_WINDOW_CANDIDATES_NM", "SMOOTH_MAX_PEAK_CHANGE_PCT",
    "SMOOTH_MIN_WINDOW_POINTS", "MIN_SMOOTH_WINDOW_NM", "SMOOTH_POLYORDER",
    "ALS_LAMBDA", "ALS_P", "ALS_NITER", "ALS_ORDER",
    "AIRPLS_LAMBDA", "AIRPLS_ORDER", "AIRPLS_NITER", "AIRPLS_TOL",
    "POLY_ORDER", "POLY_NITER", "POLY_SIGMA",
    "WHITTAKER_LAMBDA", "WHITTAKER_ORDER", "WHITTAKER_LAMBDA_CANDIDATES",
    "NORM_LINE_WINDOW_NM", "PREPROCESS_DIRNAME",
    "PIPELINE_ORDER", "ADC_CEILING", "NEAR_SATURATION_RATIO",
]

# ---------------------------------------------------------------- 方法可选值
INTERP_MODES = ("none", "uniform", "common")
BASELINE_METHODS = ("none", "als", "airpls", "poly")
SMOOTH_METHODS = ("none", "savgol", "moving", "whittaker")
NORMALIZE_MODES = ("none", "max", "area", "line")

# 方法的固定执行顺序（代码里按这个顺序跑，不由调用方指定）
PIPELINE_ORDER = ("interpolate", "baseline", "smooth", "normalize")

# ------------------------------------------------------------------ 插值/网格
# ★ 这里**故意没有**「默认步长」常量 —— 默认步长是**每条谱自己的中位步长**
#   （≈ 一个探测器像素的色散；本机不锈钢数据约 0.102 nm），由
#   tools/preprocess.py 的 _median_step() 现算；interpolate="common" 时
#   取所有谱里**最粗**的那条的中位步长。这样「重采样」只是换个采样点，
#   不凭空造出新信息。
#   （历史上这里曾有 DEFAULT_GRID_STEP_NM = 0.05，但代码从来没引用过它，
#     留着会让人误以为默认步长是 0.05 nm —— 2026-09-22 删除。）
#   想指定步长就传 grid_step_nm（下限 MIN_GRID_STEP_NM）；担心最密波段丢细节时
#   设到最密步长附近（不锈钢约 0.065 nm、快速采集约 0.043 nm）。
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
ALS_ORDER = 2                 # 惩罚的差分阶数 d（1=斜率, 2=曲率）。<上位机软件> 的 als(..., d) 也公开它
# ⚠ λ 的绝对值绑定网格步长：它惩罚的是相邻采样点的差分，所以 1e5 只有在
#   「步长 ≈ 0.1 nm 的均匀网格」上才是现在这个效果。改 grid_step_nm 要重标定。

# ------------------------------------------------------------ airPLS 基线
# ★ 来源：<LIBS 软件安装目录>\Algorithm\deBase\airPLS.py
#   用户那套工业上位机的「基线校正」下拉框只有 airPLS / als 两项，
#   且 init_status() 里默认值就是 airPLS + λ=1000 + 阶数=2 —— 所以这是现场默认算法。
#   算法出处：Zhang, Chen & Liang, "Baseline correction using adaptive iteratively
#   reweighted penalized least squares", Analyst 135 (2010) 1138.
#
# 与 ALS 的区别只有**权重更新规则**（其余：同一个 Whittaker 惩罚最小二乘内核）：
#     ALS    ：w = p 或 1-p —— 固定两档，峰点权重 p=0.01
#     airPLS ：w = exp(i·|d⁻|/Σ|d⁻|) —— 逐轮指数加重；峰点(d≥0)权重直接置 0
#   后果：airPLS 更"敢"把峰完全忽略，收敛也更快（本机真实谱实测 4~5 轮就触发停机）。
AIRPLS_LAMBDA = 1.0e3         # 与 <上位机软件> 上位机默认一致
AIRPLS_ORDER = 2              # 惩罚二阶差分
AIRPLS_NITER = 15             # 与 <上位机软件> 的 itermax 默认一致
AIRPLS_TOL = 1.0e-3           # 停机阈值：|Σd⁻| < TOL·Σ|y| 就停（<上位机软件> 里是写死的 0.001）

# ------------------------------------------------------------ 多项式基线
POLY_ORDER = 5                # 阶数太高会追着峰跑，把峰当基线扣掉
POLY_NITER = 8                # 迭代次数（含 σ 剔除）
POLY_SIGMA = 2.0

# --------------------------------------------------------- Whittaker 平滑
# 来源：<上位机软件> 的 Algorithm/deNoise/whittaker.py（wt_deNoise，默认 λ=10、d=2）。
# 它是**全局惩罚最小二乘** min Σ(y-z)² + λ‖D_d z‖² —— 没有"窗口"这个概念，
# 平滑强度全由 λ 决定。和 SG 的本质差别：SG 是局部卷积，窄峰会被卷积核定宽抹平；
# Whittaker 是全局解，看起来"更讲道理"。
#
# ★ 但实测（2026-09-22，本机真实谱）它**救不了窄峰**，别指望：
#   304 那条 FWHM≈2 像素的 652.38 nm 真谱线（同组 25/25 条都有，是真线）：
#       SG 最小窗(5 点) → 峰高变化 +25.8%
#       Whittaker λ=1   → +41.1%
#       Whittaker λ=10  → +62.7%
#   原因：要压住全谱噪声，λ 必须大到让**全局**曲率受限，而窄峰正是全谱曲率最大处，
#   必然被优先削掉。所以窄峰问题不是"换个平滑器"能解的，见 tools/preprocess.py。
WHITTAKER_LAMBDA = None       # None = 自动选 λ（按同一条"最高峰变化 ≤ 2%"判据扫候选）
WHITTAKER_ORDER = 2
WHITTAKER_LAMBDA_CANDIDATES = (1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0)

# -------------------------------------------------------------------- 归一化
# 用内标线归一化时，在目标波长 ±该窗口内找峰顶作为分母。
NORM_LINE_WINDOW_NM = 0.30

# 预处理产物的落盘目录（相对于 config.paths.OUTPUT_DIR）
PREPROCESS_DIRNAME = "preprocessed"
