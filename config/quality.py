r"""
config.quality —— 光谱质量判据的阈值（集中一处，方便调）。

============================ 先明确一件事 ============================

下面这些数字是**工程经验阈值**，不是行业标准，也不是论文里的硬指标。
它们的作用是「把明显有问题的谱挑出来」，不是「给光谱下合格/不合格的判决书」。

所以：
  - 任何结论都要在报告里带上「依据是哪条阈值、当时设的是多少」；
  - 觉得太松/太严，直接改这个文件，代码不用动。

（如果是做重建/成像质量的评价，那要另用 RMSE / SSIM / NRMSE 那一套，
 跟这里的「单条光谱自身采集质量」不是一回事，别混用。）
"""

from __future__ import annotations

__all__ = [
    "ADC_CEILING",
    "NEAR_SATURATION_RATIO",
    "SNR_GOOD",
    "SNR_FAIR",
    "BASELINE_HIGH_RATIO",
    "MAX_SPIKES_WARN",
    "SPIKE_SIGMA_K",
    "SPIKE_WINDOW",
    "SPIKE_MAX_WIDTH",
    "SPIKE_MIN_HEIGHT_RATIO",
    "SPIKE_DOMINANT_TOL_NM",
    "SPIKE_RECURRENT_RATIO",
    "SPIKE_CLUSTER_TOL_NM",
    "BASELINE_WINDOW",
    "BASELINE_Q",
    "AXIS_TOL_NM",
]

# ---- 饱和判定 ----
# 本机仪器是 16 位采集（<LIBS 软件安装目录> 的接口清单里写明多数型号 16 位，
# 如海 DQPROPLUS 支持 18 位）。16 位的天花板就是 65535。
ADC_CEILING = 65535
NEAR_SATURATION_RATIO = 0.99  # 达到天花板的 99% 记为「近饱和」，也要提醒

# ---- 信噪比判定（峰值净高度 / 噪声标准差）----
SNR_GOOD = 50.0   # >= 50 视为可接受
SNR_FAIR = 20.0   # 20~50 偏低；< 20 视为弱信号，建议重测

# ---- 背景（基线）判定 ----
# 基线中位数占采集上限的比例。太高说明连续辐射/杂散光强，弱峰会淹掉。
BASELINE_HIGH_RATIO = 0.10

# ---- 尖刺（宇宙射线）判定 ----
# 宇宙射线打在 CCD 上会形成 1~2 个像素的孤立尖峰。它不算谱线，会污染寻峰和定量。
#
# ★ 必须知道的局限：**只看单条谱，窄的真谱线和宇宙射线在形状上无法区分。**
#   两者都是「又高又窄的孤立峰」。所以在单条谱里只能报「疑似孤立尖峰」，
#   并把最像真谱线的那一类（落在该谱主峰上的）单独归到 dominant_overlaps，
#   不参与「尖刺过多」的判定。
#   真正能把两者分开的是**重复性**：真谱线在同一个扫描点上每一条谱都出现在
#   同一波长；宇宙射线是随机打上来的，换一条谱就换位置。
#   这件事在组检查（check_many）里做，见下面 SPIKE_RECURRENT_RATIO。
SPIKE_SIGMA_K = 8.0    # 偏离局部中位数超过 k 倍噪声标准差，算疑似尖刺
SPIKE_WINDOW = 7       # 局部中位数的窗口（点）
SPIKE_MAX_WIDTH = 2    # 超过这个宽度就不算尖刺（那是真实谱线）
SPIKE_MIN_HEIGHT_RATIO = 0.08  # 尖刺高度还要不小于峰值净高度的这个比例
MAX_SPIKES_WARN = 5    # 一条谱超过这个数就提醒

# 单条谱：落在主峰 ±这个波长范围内的高窄峰，判为「就是主峰本身」，不算尖刺。
# 依据：全组 25 条谱的主峰都稳定落在同一波长（真实谱线是重复的），
#       而宇宙射线不会每次都落在主峰上。
SPIKE_DOMINANT_TOL_NM = 0.5

# 组检查：同一波长上的疑似尖峰如果在 >= 这个比例的谱里都出现，
# 说明它不是随机打上来的 → 判为真谱线，从尖刺里剔除。
SPIKE_RECURRENT_RATIO = 0.30
# 判定「同一波长」的聚类容差（nm）。
SPIKE_CLUSTER_TOL_NM = 0.5

# ---- 基线估计参数 ----
BASELINE_WINDOW = 51   # 滚动窗口（点）
BASELINE_Q = 5.0       # 窗口内取第 5 百分位作为局部基线

# ---- 波长轴一致性判定 ----
AXIS_TOL_NM = 1e-3     # 波长逐点比较的容差（nm）
