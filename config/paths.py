r"""
config.paths —— 项目的路径约定（集中一处，别在代码里散落硬编码）。

为什么单独一个文件？
  数据在哪、输出写到哪，这类信息会变（换电脑、换项目自带数据、换仪器）。
  集中在这里，换的时候只改一个地方，Agent 和工具都不动。

DATA_ROOT 的取值顺序（V3）
-------------------------
  1. 环境变量 `LIBS_DATA_ROOT`（推荐方式 —— 改环境变量不会在 git 里留下本地差异）
  2. 若本机存在 <你的数据目录>（原始仪器导出目录），就用它
  3. 否则退回项目内的 data/sample（合成示例数据）

为什么要第 3 步：代码要在两种机器上都跑得起来 ——
  你自己的机器上（有真实数据）走第 2 步；
  别人 clone 下来（没有真实数据）自动走第 3 步，一条命令就能看 demo。
  这样**同一份代码不需要为「公开版」做任何改动**。
"""

from __future__ import annotations

import os

__all__ = [
    "PROJECT_ROOT", "DATA_ROOT", "DATA_ROOT_SOURCE",
    "OUTPUT_DIR", "DATA_DIR", "SAMPLE_DIR", "CONFIG_DIR",
]

# 项目根目录（本文件在 <项目>/config/ 里，所以往上退一层）
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
SAMPLE_DIR = os.path.join(DATA_DIR, "sample")

# 原始仪器数据的默认位置（2026-09-21 用户指定：不锈钢 + 铝合金的二维扫描数据）
_LOCAL_DEFAULT = r"<你的数据目录>"


def _resolve_data_root() -> tuple[str, str]:
    """返回 (数据根目录, 来源说明)。来源说明用来让 check_env.py 讲清楚「现在读的是哪份数据」。"""
    env = os.getenv("LIBS_DATA_ROOT")
    if env:
        return os.path.abspath(env), "环境变量 LIBS_DATA_ROOT"
    if os.path.isdir(_LOCAL_DEFAULT):
        return _LOCAL_DEFAULT, "本机默认目录（真实仪器数据）"
    return SAMPLE_DIR, "项目内合成示例数据 data/sample"


DATA_ROOT, DATA_ROOT_SOURCE = _resolve_data_root()

# 出图 / 报告往这里写
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")

CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))
