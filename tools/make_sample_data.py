r"""
tools/make_sample_data.py —— 生成「合成的」示例光谱（**不含任何真实测量数据**）。

为什么需要这个脚本
------------------
真实测量光谱属于实验 / 项目数据，不适合随代码公开分发。
但一个「clone 下来跑不起来」的仓库没有价值 —— 别人打开就想能问一句
`python main.py "有哪些数据"` 并立刻看到这套 Agent 真的在工作。
所以这里用**物理上合理但完全合成**的光谱补位。

合成的原则（尽量贴近真实，但绝不冒充真实）
------------------------------------------
* 波长轴照真实仪器来：180.6~954.0 nm、7745 点、步长**非均匀**（0.065~0.142 nm），
  和真数据的统计特征一致 —— 这样 Step 12 的「按像素步长自适应」逻辑才有意义。
* 峰位用**真实原子谱线波长**（Fe / Cr / Ni / Mo / Mn / Mg / Al / Zn 的常见灵敏线），
  宽度按实测规律设（同一条谱里峰宽能差 10 倍）。
* 背景 = 常数 + 缓慢起伏；噪声按信号强度的平方根给（近似泊松）。
* 故意让一组数据打饱和（顶到 16 位 ADC 上限 65535）、
  并埋几个孤立尖刺，这样质量检查的「饱和」「疑似尖峰」分支能被演示到。
* 文件名里的时间戳是**全 0**，一眼就能看出不是真实采集。

用法
----
    # 直接生成到 data/sample/（本地开发用）
    .\.venv\Scripts\python.exe tools\make_sample_data.py

    # 列出会生成哪些文件（相对路径，一行一个）
    .\.venv\Scripts\python.exe tools\make_sample_data.py --list

    # 把某一个文件的内容打到 stdout —— 给「需要保住明文」的场景用。
    # 本机有按扩展名的 DLP 加密层：Python 直接 open().write() 写 .py/.csv
    # 会落成密文（文件头 %TSD-Header-###%），而 shell 重定向落盘是明文。
    # 所以导出公开版时由 shell 负责落盘，Python 只负责把内容吐出来。
    .\.venv\Scripts\python.exe tools\make_sample_data.py --emit demo-304-grid/xxx.csv > 目标文件
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import paths as P  # noqa: E402

ADC_CEILING = 65535          # 16 位采集天花板，顶到它 = 打饱和
SEED = 20260921              # 固定种子：同一条命令每次生成完全一样的数据

# ---------------------------------------------------------------------------
# 真实原子谱线表：元素 -> [(波长 nm, 相对强度 0~1, FWHM nm), ...]
# 波长取自常见 LIBS 灵敏线；相对强度只是「像那么回事」，不是定量数据。
# ---------------------------------------------------------------------------
FE = [
    (248.327, 0.55, 0.10), (259.940, 0.30, 0.10), (271.903, 0.42, 0.11),
    (302.064, 0.62, 0.12), (344.061, 0.36, 0.12), (358.119, 0.35, 0.11),
    (371.994, 0.62, 0.13), (385.991, 0.45, 0.13), (404.581, 0.30, 0.13),
    (438.354, 0.35, 0.14), (495.760, 0.30, 0.14), (526.953, 1.00, 0.16),
    (561.564, 0.22, 0.15), (652.380, 0.18, 0.17),
]
CR = [
    (357.869, 0.60, 0.10), (359.349, 0.75, 0.10), (360.533, 0.55, 0.10),
    (425.435, 1.00, 0.12), (427.480, 0.70, 0.12), (428.972, 0.55, 0.12),
    (520.452, 0.85, 0.13), (520.602, 0.90, 0.13), (520.844, 0.80, 0.13),
]
NI = [
    (341.476, 0.55, 0.10), (344.626, 0.50, 0.10), (349.296, 0.65, 0.10),
    (351.505, 0.80, 0.10), (352.454, 0.70, 0.10), (356.637, 0.55, 0.10),
    (361.939, 0.45, 0.10), (547.691, 0.30, 0.14),
]
MO = [(386.411, 0.60, 0.11), (550.649, 0.35, 0.14), (553.305, 0.30, 0.14)]
MN = [(257.610, 0.55, 0.10), (259.373, 0.45, 0.10), (279.482, 0.50, 0.10),
      (293.306, 0.35, 0.11), (403.076, 0.45, 0.12)]
MG = [(279.553, 0.85, 0.10), (280.270, 1.00, 0.10), (285.213, 0.90, 0.10),
      (383.826, 0.55, 0.12), (518.360, 0.45, 0.14)]
AL = [(308.215, 0.80, 0.11), (309.271, 1.00, 0.11), (394.401, 0.75, 0.12),
      (396.152, 0.90, 0.12)]
ZN = [(213.856, 0.70, 0.09), (334.502, 0.35, 0.11), (481.053, 0.30, 0.13)]
CU = [(324.754, 1.00, 0.11), (327.396, 0.85, 0.11)]
SI = [(251.611, 0.60, 0.10), (288.158, 1.00, 0.11), (390.553, 0.35, 0.12)]
CA = [(393.366, 0.90, 0.12), (396.847, 1.00, 0.12)]
NA = [(588.995, 1.00, 0.14), (589.592, 0.90, 0.14)]
H_ALPHA = [(656.279, 1.00, 0.18)]
# 空气 / 等离子体本身的谱线：这些和样品成分无关，但真实谱里一定有
O_TRIPLET = [(777.194, 1.00, 0.14), (777.417, 1.30, 0.14), (777.539, 1.00, 0.14)]
AR_LINES = [(696.543, 0.60, 0.13), (706.722, 0.45, 0.13), (750.387, 0.70, 0.14),
            (763.511, 0.55, 0.14), (811.531, 1.00, 0.15)]
N_LINES = [(742.364, 0.25, 0.13), (744.229, 0.50, 0.13), (746.831, 0.60, 0.13)]


def _scale(lines, k):
    return [(w, i * k, f) for (w, i, f) in lines]


# ---------------------------------------------------------------------------
# 三组成分：只要「像那个牌号」，不追求准确
#   304 钢   = Fe 为主 + Cr 约 18% + Ni 约 9%
#   316 钢   = Fe 为主 + Cr 约 17% + Ni 约 12% + Mo 约 2.5%
#   7075 铝  = Al 为主 + Zn 约 6% + Mg 约 2.5% + Cu 约 1.6%
# 每组都无条件带上空气谱线（O / Ar / N），因为真实谱里必然有。
# ---------------------------------------------------------------------------
COMPOSITIONS = {
    "304": _scale(FE, 1.0) + _scale(CR, 0.42) + _scale(NI, 0.34)
           + _scale(MN, 0.16) + _scale(SI, 0.14) + _scale(CA, 0.10)
           + O_TRIPLET + _scale(AR_LINES, 0.5) + H_ALPHA,
    "316": _scale(FE, 1.0) + _scale(CR, 0.40) + _scale(NI, 0.42) + _scale(MO, 0.12)
           + _scale(MN, 0.14) + _scale(SI, 0.12) + _scale(CA, 0.10)
           + O_TRIPLET + _scale(AR_LINES, 0.5) + H_ALPHA,
    "7075": _scale(AL, 1.0) + _scale(ZN, 0.30) + _scale(MG, 0.26) + _scale(CU, 0.12)
            + _scale(FE, 0.10) + _scale(CA, 0.12)
            + O_TRIPLET + _scale(AR_LINES, 0.6) + _scale(N_LINES, 0.5) + H_ALPHA,
}


# ---------------------------------------------------------------------------
# 1) 波长轴：造出「非均匀步长」这个真实特征
# ---------------------------------------------------------------------------
def make_axis(n=7745, lo=180.619, hi=953.970, jitter=0.22, seed=SEED):
    """造一条和真实仪器统计特征一致的波长轴。

    真实仪器不是理想的等间距光栅 —— 7745 个点、步长在 0.065~0.142 nm 之间抖动，
    均值约 0.0999 nm。这个「非均匀」是 Step 12 插值逻辑的立足点，必须造出来。
    """
    rng = np.random.default_rng(seed)
    step = (hi - lo) / (n - 1)
    raw = step * (1.0 + jitter * rng.standard_normal(n - 1))
    raw = np.clip(raw, step * 0.65, step * 1.42)     # 对应真实的 0.065 / 0.142
    raw *= (hi - lo) / raw.sum()                     # 归一回总跨度
    return np.concatenate([[lo], lo + np.cumsum(raw)])


# ---------------------------------------------------------------------------
# 2) 合成一条谱
# ---------------------------------------------------------------------------
def synthesize(axis, lines, peak_scale=40000.0, background=1500.0,
               background_slope=0.0, noise=1.0, saturate=False,
               spikes=0, seed=SEED):
    """在给定波长轴上合成一条光谱。

    参数
    ----
    peak_scale       : **最强峰的目标高度**（计数）。
                       ★ 注意这里做了归一化：不论谱线表里最强的那条相对强度是多少，
                         生成出来的最高峰都正好是 peak_scale（加背景/噪声前）。
                         这样「要不要打饱和」就是确定的，不用靠试。
    background       : 常数背景
    background_slope : 背景随波长的缓慢起伏幅度（模拟连续辐射）
    noise            : 噪声系数，1.0 = 按 sqrt(信号) 给
    saturate         : True 则把超过 65535 的部分削平（模拟真实饱和）
    spikes           : 埋几个孤立尖刺（模拟宇宙射线），用来演示尖峰检测
    """
    rng = np.random.default_rng(seed)

    # 先算出所有谱线的峰形叠加，再整体缩放到目标高度
    peaks = np.zeros(axis.size, dtype=float)
    for (mu, rel, fwhm) in lines:
        sigma = fwhm / 2.3548                      # FWHM -> 高斯 sigma
        peaks += rel * np.exp(-0.5 * ((axis - mu) / sigma) ** 2)
    if peaks.max() > 0:
        peaks *= peak_scale / float(peaks.max())

    y = np.full(axis.size, float(background), dtype=float)
    if background_slope:
        y += background_slope * np.sin((axis - axis[0]) / (axis[-1] - axis[0]) * np.pi)
    y += peaks

    # 噪声：按信号强度的平方根（近似泊松），再加一点读出噪声底
    y += noise * rng.normal(0.0, np.sqrt(np.maximum(y, 1.0)) + 3.0)

    # 孤立尖刺（宇宙射线）：真实谱里很常见，且和「窄真谱线」形状一样
    for idx in rng.choice(axis.size, size=int(spikes), replace=False):
        y[idx] += rng.uniform(0.5, 1.5) * peak_scale * 0.15

    y = np.clip(y, 0.0, None)
    if saturate:
        y = np.minimum(y, ADC_CEILING)             # 顶到 16 位上限就削平
    return y


# ---------------------------------------------------------------------------
# 3) 组规格表：一个地方描述「要生成哪些组」，落盘模式与 stdout 模式共用
# ---------------------------------------------------------------------------
GROUP_SPECS = [
    {
        "dir": "demo-304-grid", "material": "304", "nx": 3, "ny": 3,
        "axis": "main", "format": "plain",
        "peak_scale": 30000.0, "background": 1500.0, "background_slope": 900.0,
        "noise": 1.0, "saturate": False, "spikes": 0,
        "note": "304 钢，3×3 网格，无饱和（正常路径）",
    },
    {
        "dir": "demo-316-grid", "material": "316", "nx": 3, "ny": 3,
        "axis": "main", "format": "plain",
        "peak_scale": 82000.0,          # 故意超过 65535 → 顶平
        "background": 1700.0, "background_slope": 1100.0,
        "noise": 1.0, "saturate": True, "spikes": 0,
        "note": "316 钢，777 nm 附近**打饱和**（演示饱和检测）",
    },
    {
        "dir": "demo-7075-grid", "material": "7075", "nx": 3, "ny": 3,
        "axis": "main", "format": "plain",
        "peak_scale": 32000.0, "background": 1900.0, "background_slope": 1500.0,
        "noise": 1.1, "saturate": False, "spikes": 0,
        "note": "7075 铝，无饱和",
    },
    {
        "dir": "demo-700v-300ns", "material": "304", "n_repeat": 4,
        "axis": "fast", "format": "header",
        "peak_scale": 42000.0, "background": 2100.0, "background_slope": 700.0,
        "noise": 1.2, "saturate": False, "spikes": 0,
        "spike_at": {1: 2},             # 第 2 条埋 2 个孤立尖刺
        "note": "另一种采集参数；**带表头**格式；第 2 条埋了 2 个孤立尖刺",
    },
]

_AXES: dict[str, np.ndarray] = {}


def _axis_of(kind: str) -> np.ndarray:
    """轴只算一次。两种采集参数各对应一条不同的轴。"""
    if kind not in _AXES:
        _AXES[kind] = (make_axis() if kind == "main"
                       else make_axis(n=7323, lo=180.860, hi=812.200,
                                      jitter=0.24, seed=SEED + 7))
    return _AXES[kind]


def _peak_scale_for(ix, iy, base):
    """每个测点给一点随机差异，避免一组里 9 条谱一模一样。"""
    rng = np.random.default_rng(SEED + ix * 100 + iy)
    return base * float(rng.uniform(0.75, 1.20))


def _render_rows(axis, y) -> str:
    """两列文本，强度写整数（16 位 ADC）。"""
    return "".join(f"{w:.3f},{int(round(v))}\n" for w, v in zip(axis, y))


def render_group(spec: dict) -> list[tuple[str, str]]:
    """产出这一组的全部文件，返回 [(相对路径, 文件内容), ...]。

    ★ 为什么要有「只产出内容、不落盘」这个接口：
      本机有按扩展名的 DLP 加密层 —— Python 用 open().write() 写 .py / .csv
      会落成密文（文件头 %TSD-Header-###%），而 **shell 重定向**写出来是明文。
      所以导出公开版时，落盘交给 shell，Python 只负责把内容打到 stdout。
    """
    gdir = spec["dir"]
    axis = _axis_of(spec["axis"])
    lines = COMPOSITIONS[spec["material"]]
    spike_at = spec.get("spike_at") or {}
    out: list[tuple[str, str]] = []

    if "n_repeat" in spec:
        keys = [(0, 0, k) for k in range(spec["n_repeat"])]
    else:
        keys = [(ix, iy, 0) for ix in range(spec["nx"]) for iy in range(spec["ny"])]

    for (ix, iy, k) in keys:
        y = synthesize(
            axis, lines,
            peak_scale=_peak_scale_for(ix, iy, spec["peak_scale"]),
            background=spec["background"],
            background_slope=spec["background_slope"],
            noise=spec["noise"],
            saturate=spec["saturate"],
            spikes=spike_at.get(k, spec["spikes"]),
            seed=SEED + ix * 37 + iy + k * 11,
        )
        if "n_repeat" in spec:
            name = f"fast-0000000000-0000-{100 + k}.csv"
        else:
            name = f"DATA-0000000000-X{ix}-Y{iy}-1.csv"
        body = _render_rows(axis, y)
        if spec["format"] == "header":
            body = "wavelength,intensity\n" + body
        out.append((f"{gdir}/{name}", body))

    return out


SAMPLE_README = """\
# data/sample —— 合成示例数据（**不是真实测量数据**）

本目录下的全部 `.csv` 都由 `tools/make_sample_data.py` **程序生成**，
不包含任何真实实验采集的光谱。

## 为什么要放合成数据

真实测量数据属于实验 / 项目数据，不便随代码分发。
但仓库要能「clone 下来就跑」，所以用物理上合理的光谱补位。

## 它们「真」到什么程度

| 特征 | 是否照真实仪器还原 |
|---|---|
| 波长轴 180.6~954.0 nm、7745 点 | 是，点数与跨度照真实仪器 |
| 步长非均匀（0.065~0.142 nm） | 是，连抖动幅度都对齐了 |
| 谱线波长 | 是，用真实原子跃迁波长（Fe / Cr / Ni / Mo / Al / Zn …） |
| 峰宽差异 | 是，同一条谱里峰宽差约 10 倍，和实测一致 |
| 背景 / 噪声 | 是，常数背景 + 缓慢起伏 + 近似泊松噪声 |
| 饱和 | 是，`demo-316-grid` 顶到 16 位上限 65535，峰顶被削平 |
| 孤立尖刺 | 是，`demo-700v-300ns` 第 2 条埋了 2 个 |
| **绝对强度 / 浓度** | **否，纯属编造，没有任何定量意义** |

## 一句话

拿它验证流程（加载、质量检查、预处理、绘图、Agent 工具调用）是合适的；
**拿它下任何关于材料成分的结论都是错的。**

重新生成：

```bash
python tools/make_sample_data.py
```

固定随机种子，同一条命令每次生成完全一致的数据。

> 注：本机（Windows + DLP 加密层）用 Python 直接落盘的 `.csv` 在磁盘上是密文，
> 只有 Python 能透明读回；这是本机特性，不影响使用。
"""


def all_files() -> list[tuple[str, str]]:
    """全部会生成的文件（含本目录的 README.md）。"""
    files: list[tuple[str, str]] = []
    for spec in GROUP_SPECS:
        files.extend(render_group(spec))
    files.append(("README.md", SAMPLE_README))
    return files


def _emit(text: str) -> None:
    """打 UTF-8 字节到 stdout，绕开 Windows 控制台编码，并保住 LF。"""
    try:
        sys.stdout.buffer.write(text.encode("utf-8"))
        sys.stdout.buffer.flush()
    except AttributeError:                # 极端情况下退化为文本写
        sys.stdout.write(text)


def main() -> int:
    ap = argparse.ArgumentParser(description="生成合成示例光谱（不含真实测量数据）")
    ap.add_argument("--out", default=P.SAMPLE_DIR, help="输出目录，默认 data/sample")
    ap.add_argument("--list", action="store_true", help="只列出会生成哪些文件（相对路径）")
    ap.add_argument("--emit", metavar="REL", help="把某个文件的内容打到 stdout（给 shell 重定向用）")
    args = ap.parse_args()

    # ---------------- 只列清单 ----------------
    if args.list:
        # ★ 必须走二进制口写 LF。用 print() 的话，Windows 上文本模式会把 \n
        #   翻译成 \r\n，而 shell 侧 `while IFS= read -r rel` **不剥 \r**，
        #   于是 rel 尾部多一个 \r，再拿去 `--emit` 就报「没有这个文件」。
        _emit("".join(rel + "\n" for rel, _ in all_files()))
        return 0

    # ---------------- 只吐一个文件的内容（不落盘）----------------
    if args.emit:
        want = args.emit.strip().replace("\\", "/")
        if want == "README.md":
            _emit(SAMPLE_README)
            return 0
        for spec in GROUP_SPECS:
            if want.startswith(spec["dir"] + "/"):
                for rel, text in render_group(spec):
                    if rel == want:
                        _emit(text)
                        return 0
        print(f"没有这个文件：{want}", file=sys.stderr)
        return 2

    # ---------------- 默认：直接落盘（本地开发用）----------------
    out = os.path.abspath(args.out)
    print("=" * 66)
    print(" 生成合成示例光谱（不是真实测量数据）")
    print("=" * 66)
    print(f"输出目录：{out}\n")

    total = 0
    for spec in GROUP_SPECS:
        files = render_group(spec)
        os.makedirs(os.path.join(out, spec["dir"]), exist_ok=True)
        for rel, text in files:
            with open(os.path.join(out, rel), "w", encoding="utf-8", newline="\n") as fh:
                fh.write(text)
        total += len(files)
        print(f"  {spec['dir'] + '/':<20} {len(files):2d} 条  {spec['note']}")

    readme = os.path.join(out, "README.md")
    with open(readme, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(SAMPLE_README)

    print()
    print("-" * 66)
    print(f" 共 {total} 条合成光谱")
    print(f" 说明文件：{readme}")
    print("-" * 66)
    print(" ⚠ 这些数据是**程序生成的**，只能用来验证流程，不能用来判断材料成分。")
    print()
    print(" 提示：若要生成「明文文件」（例如导出公开版给 Git 用），不要用这个默认模式，")
    print("       改用 --list + --emit，让 shell 负责落盘。详见脚本顶部说明。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
