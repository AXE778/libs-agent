r"""
agent.tool_registry —— 把 tools/ 里的函数「登记」成大模型能调用的工具。

============================ 这一层为什么必要 ============================

Python 函数和「模型能调用的工具」是两回事。模型看不见你的代码，
它只能看见一份**说明书**：工具叫什么、干什么、要哪些参数。

那份说明书就是 `TOOLS` 里的 JSON Schema。模型读完说明书后，
会输出一段「我要调用 load_spectrum_summary，参数 path=...」，
真正执行它的是 `execute()`。

所以这个文件干两件事：
    1. TOOLS      —— 给模型看的说明书（描述写得好不好，直接决定模型会不会用对）
    2. execute()  —— 把模型的请求翻译成真正的 Python 调用

分层规矩：tools/ 不许 import agent/；agent/ 可以 import tools/（单向依赖）。
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Callable

# 让本文件能被 `python -m agent.xxx` 和 `python main.py` 两种方式导入
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from config.paths import DATA_ROOT, OUTPUT_DIR  # noqa: E402
from tools import (  # noqa: E402
    element_analysis,
    preprocess,
    report,
    spectrum_catalog,
    spectrum_loader,
    spectrum_quality,
)

__all__ = ["TOOLS", "TOOL_NAMES", "execute", "execute_to_json"]

# 单次工具结果最大字符数（防止把 2000 多个文件路径一次性灌进模型上下文）
# 2026-09-21 从 8000 提到 12000：元素匹配的结果天生更大（元素表 + 判定理由 +
# 峰列表 + 注意事项），8000 会触发通用瘦身把元素表砍到只剩 2 条 ——
# 那样模型就看不到第 3 个「有支持」的元素了。宁可多给点，也别砍结论。
_MAX_CHARS = 12000


# ---------------------------------------------------------------------------
# 1) 给模型看的说明书
# ---------------------------------------------------------------------------
TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "list_available_spectra",
            "description": (
                "查看光谱数据有哪些。只读文件名和目录，不读光谱内容，很快。"
                "两种用法：① 不传 query → 返回数据总览（有几组数据、每组多大、每组是什么网格、多少测点）；"
                "② 传 query → 返回匹配的文件路径，**并附上命中组摘要 matched_groups**，"
                "供 load_spectrum_summary / plot_spectra 使用。"
                "★ 问「有哪些数据 / 测了哪些点 / 有哪些材料」时**不要传 query**，"
                "取一次总览就够（总览里已经有网格与测点数），别先筛一遍再回头补总览。"
                f"数据根目录固定在 {DATA_ROOT}。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "筛选关键字，匹配子目录名。例如 304 / 316 / 6061 / 7075 / 新 / data-304-1。"
                            "留空则只返回总览，不列文件。"
                        ),
                    },
                    "limit": {
                        "type": "integer",
                        "description": "传了 query 时最多返回多少个文件路径，默认 25，建议不超过 100。",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "describe_material_grid",
            "description": (
                "把某一组测量的测点还原成二维网格，告诉你网格多大、每个点打了几次、有没有缺测点。"
                "当用户问「这个数据的扫描网格是什么样/覆盖多少点」时用它。"
                "这也正是「二维元素成像」的第一步：先知道网格，才谈得上重建。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "material": {
                        "type": "string",
                        "description": "子目录名或关键字，如 data-304-1 / 304 / 6061。建议先跑 list_available_spectra 拿到准确名字。",
                    }
                },
                "required": ["material"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "load_spectrum_summary",
            "description": (
                "读取一条光谱文件，返回它的摘要：点数、波长范围、波长步长、强度范围、"
                "最强峰在哪、粗估信噪比，以及数据来源信息（编码/分隔符/有没有表头）。"
                "这是「看一条谱」的标准动作。参数必须是真实的文件路径。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "光谱文件完整路径，例如 <你的数据目录>\\data-304-1\\DATA-0908165817-X0-Y0-1.csv",
                    }
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_spectrum_quality",
            "description": (
                "对一条光谱做采集质量检查，返回判级（OK/WARN/FAIL）和具体问题清单："
                "是否打饱和（含饱和波段范围）、信噪比、背景是否偏高、疑似宇宙射线尖刺。"
                "当用户问「这条谱质量怎么样 / 能不能用 / 有没有饱和 / 信噪比多少」时用它 —— "
                "它比 load_spectrum_summary 更专业、给出的判据更明确，**优先用它**。"
                "参数必须是真实文件路径。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "光谱文件完整路径，例如 <你的数据目录>\\data-304-1\\DATA-0908165817-X0-Y0-1.csv",
                    }
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_group_quality",
            "description": (
                "批量检查一整组光谱（例如一个材料的所有测点），汇总回答「这组数据整体能不能用」："
                "多少条打饱和、多少条信号太弱、背景是否普遍偏高、波长轴是否一致、"
                "以及最该先看的几条是哪些。当用户问「这批数据行不行 / 哪些点有问题 / 数据能不能用」时用它。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "material": {
                        "type": "string",
                        "description": "子目录名或关键字，如 data-304-1 / 304 / 6061。建议先跑 list_available_spectra 确认名字。",
                    },
                    "max_files": {
                        "type": "integer",
                        "description": "最多检查多少条，默认 60。一组 25 条时建议全查；150 条时建议先查 60 条。",
                    },
                },
                "required": ["material"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "plot_spectra",
            "description": (
                "把若干条光谱画成一张网页图并保存，返回文件路径供用户打开查看。"
                "当用户说「画出来/画个图/对比一下这几条」时用它。"
                "需要先有确切路径（先跑 list_available_spectra）。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "paths": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "要画的文件路径列表，2~6 条比较合适。",
                    },
                    "out_name": {
                        "type": "string",
                        "description": "输出文件名（可省），例如 compare_304.html。会写到 outputs/ 下。",
                    },
                },
                "required": ["paths"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_element_lines",
            "description": (
                "对一条光谱做**元素谱线匹配**：找出谱里的峰，把它们和谱线库里的谱线对照，"
                "回答「哪些元素的证据值得人工复核」。"
                "当用户问「这条谱里有什么元素 / 是不是 304 / 有没有 Cr、Ni / 帮我认一下这些峰」时用它。"
                "\n\n★ 必须先向用户说明这个工具的边界，否则会误导人："
                "\n① 它**不判「这个峰属于哪个元素」**。库里每 nm 有 20 多条线，"
                "单个峰对上一两条线是常事、几乎不构成证据。"
                "\n② 输出里只有 verdict=『有支持』的才值得看；"
                "『不可采信』= 证据不比瞎猜强；『证据不足』= 峰太少。"
                "\n③ 即使判为『有支持』，也只是**候选**，不等于检出，必须人工核对谱图。"
                "\n④ 空气元素（N / O / Ar）本身就在空气中，它们『有支持』不代表样品含这些元素；"
                "输出里带 in_air=true 的就是这类，报的时候要单独说清楚。"
                "\n⑤ Fe 在库里线最多，本工具基本**无法**对 Fe 下结论，不要硬报。"
                "\n⑥ 数据没有浓度标定，绝不能说含量/浓度。"
                "\n⑦ 如果输出里有 near_miss_peaks，那说明有强峰「差一点点」对上强线，"
                "通常意味着这段波长标定有偏差，要提醒用户去核对，而不是当成「没有该元素」。"
                "\n参数必须是真实文件路径（先跑 list_available_spectra）。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "光谱文件完整路径，例如 <你的数据目录>\\data-304-1\\DATA-0908165817-X0-Y0-1.csv",
                    },
                    "tol_nm": {
                        "type": "number",
                        "description": (
                            "波长匹配容差（nm）。**一般不要传**：默认会按峰所在处的像素步长自动算"
                            "（这台仪器 0.065~0.142 nm/像素），这是实测最稳的做法。"
                            "只有在想手动放宽/收紧时才传，范围 0~0.20。"
                        ),
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_report",
            "description": (
                "把一次检测汇总成一份**可交付的 Markdown 报告文件**（写到 outputs/），"
                "内容 = 数据概况 + 采集质量 + 元素谱线匹配 + 局限 + 附录（文件清单、判据阈值）。"
                "当用户说「出一份报告 / 给我一份文档 / 汇总一下这批数据 / 把结果整理成报告」时用它。"
                "它同时也能当「这批数据到底怎么样」的**一站式回答**，"
                "因为一次调用就把质量检查和元素匹配都跑了。"
                "\n\n★ 用法与纪律："
                "\n① 参数二选一：`material`（一组数据的关键字，如 316 / data-316-1）"
                "或 `paths`（明确的文件路径列表，建议不超过 5 条；给 1 条则出单条谱报告）。"
                "\n② 一次调用可能要跑十几秒（元素匹配是重活），这是正常的，别重复调用。"
                "\n③ 返回里**没有报告正文** —— 正文在磁盘上（report_path）。"
                "你要做的是：把 report_path 给用户，再把 headline 里的要点讲一遍，"
                "并至少带上 warnings 里与你回答相关的边界。"
                "\n④ **不要自己重新组织或推算 headline 里的数字**，照实转述；"
                "也**绝对不要说含量 / 浓度**（这批数据没有浓度标定，报告里也没有）。"
                "\n⑤ 报告里判为「有支持」的元素仍然只是**候选**，不是检出 —— 转述时必须保留这个措辞。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "material": {
                        "type": "string",
                        "description": (
                            "一组数据的关键字。**优先用完整子目录名**（如 data-7075-1、data-316-1），"
                            "这样报告只针对一组；只给牌号（如 7075）会同时命中 data-7075-1 / "
                            "data-7075-2 / data-7075-新 并跨目录汇总 —— 不同批次混在一起统计是有风险的，"
                            "报告里会就此给出警告。建议先跑 list_available_spectra 确认准确名字。"
                        ),
                    },
                    "paths": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "明确要写进报告的光谱文件路径。给了就优先用它；只给 1 条时输出单条谱报告。",
                    },
                    "max_files": {
                        "type": "integer",
                        "description": "按 material 出报告时最多检查多少条，默认 60。一组 25 条建议全查。",
                    },
                    "element_files": {
                        "type": "integer",
                        "description": (
                            "对几条谱做元素匹配，默认 3，传 0 等于不做（快很多）。"
                            "为什么不是全组都做：元素匹配要对全体峰跑 200 次置换检验，"
                            "而同一组的谱线分布高度相似，全做等于把同一个结论重复 N 次。"
                        ),
                    },
                    "out_name": {
                        "type": "string",
                        "description": "输出文件名（可省）。默认带时间戳，避免覆盖上一次的报告。",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "preprocess_spectrum",
            "description": (
                "对一条或一组光谱做**预处理**（插值 / 基线校正 / 平滑 / 归一化），"
                "把处理后的谱写成一类新文件（outputs/preprocessed/），返回新文件路径，"
                "供后续工具（analyze_element_lines / plot_spectra 等）继续使用。"
                "当用户说「先平滑一下 / 降噪 / 做个基线校正 / 归一化 / 把几条谱对到同一个波长轴上」时用它。"
                "\n\n执行顺序**固定**：插值 → 基线校正 → 平滑 → 归一化。你只选方法和参数，不能改顺序"
                "（顺序写死是为了结果可复现、可解释）。"
                "\n\n★★ 三条边界见系统提示「第六纪律」，必须执行、也必须转述给用户 ★★"
                "\n① **归一化改变强度含义** —— 除非用户明确要求，normalize 一律保持 none；"
                "\n② **预处理后的谱不得再送去质量检查**（check_spectrum_quality / check_group_quality）"
                "—— 会得出假的合格结论，质检一律用原始谱；"
                "\n③ **绝不外推** —— 统一波长轴时自动取交集，超出覆盖范围的波段一个点都不生成"
                "（不锈钢那套到 953.97 nm、快速采集那套只到 812.20 nm），用户要更长就得明说数据不存在。"
                "\n\n★ 参数：**不传任何参数 = 最保守的默认链**（插值成均匀网格 + 自适应平滑，不基线、不归一化），"
                "适合大多数「帮我处理一下这条谱」。要让多条谱**能逐点比较**（如叠成矩阵做二维强度图）"
                "才用 interpolate=\"common\"；背景明显偏高（质检报 baseline 偏高）再上 baseline=\"als\"。"
                "**别手填 smooth_window_nm / grid_step_nm / grid_range** —— 平滑窗口默认自动挑"
                "（跟随谱线宽度，本机实测同一仪器上峰宽相差 10 倍），手填容易把峰削掉或把点数抬爆。"
                "要归一化就先问用户基准：max / area / line（line 必须同时给 norm_line_nm）。"
                "\n\n★ 返回里出现 warnings、或 peak_change_pct 明显不为 0，**必须一并转述**，"
                "不要只说「处理好了」。⚠ peak_change_pct：**正数=最高峰被压低、负数=被抬高，都算失真**，"
                "优先直接念 peak_change_note。峰有多宽要念返回里的 peak_fwhm_points / peak_fwhm_nm，"
                "**不要自己估**（那是代码算好给你的，就是为了避免估错）。并把 output_path 给用户，后续分析用那个路径。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "paths": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "要处理的光谱文件路径（一条或一批）。"
                            "必须是 list_available_spectra 返回过的真实路径。"
                            "一批建议不超过 25 条。"
                        ),
                    },
                    "interpolate": {
                        "type": "string",
                        "enum": ["none", "uniform", "common"],
                        "description": (
                            "插值方式。uniform=各自重采样成均匀网格（默认，解决步长非均匀）；"
                            "common=统一到同一条波长轴（多条谱要逐点比较时用，自动取覆盖范围的交集）；"
                            "none=不插值（仅当原谱已经是均匀网格时）。"
                        ),
                    },
                    "grid_step_nm": {
                        "type": "number",
                        "description": (
                            "重采样步长（nm）。**一般不要传**：默认取该谱的中位步长，"
                            "即「一个新点 ≈ 一个探测器像素」。"
                            "本机最密步长约 0.065（不锈钢）/ 0.043（快速采集）；"
                            "担心丢细节就传这个量级的值，但点数会明显变多。"
                        ),
                    },
                    "grid_range": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": (
                            "统一轴的目标范围 [下限, 上限]（nm），只在 interpolate=\"common\" 时有意义。"
                            "**不要传**：默认自动取所有谱覆盖范围的交集，这样每条输出长度一致且都不外推。"
                            "只有用户明确要求某个波段时才传。"
                        ),
                    },
                    "baseline": {
                        "type": "string",
                        "enum": ["none", "als", "airpls", "poly"],
                        "description": (
                            "基线校正方法。none=不校正（默认）；"
                            "als=非对称最小二乘（能贴合缓慢起伏的背景）；"
                            "airpls=自适应迭代重加权（同一套算法族，对峰的抑制更狠、"
                            "收敛更快；用户那套工业上位机的默认项就是它，要与现场结果对齐时用它）；"
                            "poly=多项式拟合（更简单，背景接近平缓曲线时够用）。"
                            "★ 三者都是把「缓慢背景」扣掉；如果谱里没有明显抬高的背景，"
                            "不要为了保险起见硬加基线校正 —— 扣多了会把弱峰一起削掉。"
                        ),
                    },
                    "smooth": {
                        "type": "string",
                        "enum": ["none", "savgol", "moving", "whittaker"],
                        "description": (
                            "平滑方法。savgol=Savitzky-Golay（默认，保峰形最好）；"
                            "moving=滑动平均（更简单，但更容易削峰）；"
                            "whittaker=全局惩罚最小二乘（没有窗口概念，强度由 λ 自动选；"
                            "注意实测它在窄峰上比 savgol 更差，不要以为它更高级就用它）；"
                            "none=不平滑。"
                        ),
                    },
                    "smooth_window_nm": {
                        "type": "number",
                        "description": (
                            "平滑窗口宽度（nm）。**强烈建议不要传** —— 默认自动选，"
                            "自动会挑「峰高损失不超过 2%」的最大窗口。"
                            "手动指定只在用户明确要求「更强的平滑」时才用，"
                            "并且要如实告诉他这会削掉多少峰高（返回里会给出）。"
                        ),
                    },
                    "normalize": {
                        "type": "string",
                        "enum": ["none", "max", "area", "line"],
                        "description": (
                            "归一化方式。none=不归一化（默认，保持强度是原始计数）；"
                            "max=按全谱最大值；area=按全谱积分面积；"
                            "line=按指定内标线附近峰高（必须同时给 norm_line_nm）。"
                            "★ 选任何非 none 的值，都必须向用户说明「强度已不再是原始计数」。"
                        ),
                    },
                    "norm_line_nm": {
                        "type": "number",
                        "description": (
                            "normalize=\"line\" 时的内标线波长（nm），例如 425.43。"
                            "要选一条在**所有**待处理谱都覆盖到、且**没有打饱和**的谱线。"
                        ),
                    },
                },
                "required": ["paths"],
            },
        },
    },
]

TOOL_NAMES = [t["function"]["name"] for t in TOOLS]


# ---------------------------------------------------------------------------
# 2) 真正的执行函数（每个都返回「可 JSON 化的 dict」）
# ---------------------------------------------------------------------------
def _t_list_available_spectra(query: str | None = None, limit: int = 25) -> dict:
    """不传 query → 数据总览；传 query → 匹配的文件路径。

    为什么做成两种形态？
      「不筛选就列出全部文件」会把上下文撑爆，而模型真正需要的第一步信息是
      「有哪些组数据、每组多大」。所以不传 query 时只给总览，
      等模型知道要找什么了，再带 query 精确捞文件。
    """
    if not query or not str(query).strip():
        ov = spectrum_catalog.overview(DATA_ROOT)
        ov["hint"] = (
            "这是数据总览（故意没列具体文件）。要拿具体文件路径，"
            "把材料牌号或子目录名作为 query 再调一次，例如 query=\"304\" 或 query=\"data-6061-1\"。"
        )
        return ov

    limit = max(1, min(int(limit or 25), 200))
    res = spectrum_catalog.find_spectra(DATA_ROOT, str(query), limit=limit)

    # ★ 传了 query 时，顺带附一份「命中组摘要」。
    #   为什么这么做？实测（2026-09-22）模型问「304 测了哪些点」时，本能地会带 query 调一次，
    #   拿到的却是一长串被截断的文件路径 → 于是**不得不再调一次不带 query 的总览**，
    #   白烧一整轮模型往返（约 2～3 秒，占总耗时 20%）。把摘要一起给它，第一轮就够用。
    q = str(query).strip().lower()
    ov = spectrum_catalog.overview(DATA_ROOT)
    hit_dirs = {g["dir"] for g in ov["groups"] if q in str(g["dir"]).lower()}
    if hit_dirs:
        res["matched_groups"] = [g for g in ov["groups"] if g["dir"] in hit_dirs]
        res["matched_by_material"] = [
            b for b in ov["by_material"] if hit_dirs.intersection(b["dirs"])
        ]

    res["hint"] = (
        f"只显示了 {res['n_returned']}/{res['n_matched']} 个文件路径（已截断）。"
        "「有哪些数据 / 测了哪些点」看 matched_groups 就够了，**不必再调一次不带 query 的总览**。"
        "想读某一条，把它的 path 传给 load_spectrum_summary；想画图传给 plot_spectra。"
        if res["truncated"] else
        f"共 {res['n_matched']} 个文件，全部已返回；"
        "「有哪些数据 / 测了哪些点」看 matched_groups 即可。"
        "想读某一条，把它的 path 传给 load_spectrum_summary；想画图传给 plot_spectra。"
    )
    return res


def _t_describe_material_grid(material: str) -> dict:
    return spectrum_catalog.scan_grid(DATA_ROOT, material)


def _t_load_spectrum_summary(path: str) -> dict:
    spec = spectrum_loader.load_spectrum(path)
    s = spectrum_loader.summarize(spec)
    # 饱和判断（16 位 ADC 上限）——很重要的质量信号
    if s["intensity_max"] >= 65535:
        s["saturation_warning"] = "最强值达到 65535（16 位 ADC 上限），这条谱打饱和了，峰顶被削平。"
    else:
        s["saturation_warning"] = None
    return s


def _t_check_spectrum_quality(path: str) -> dict:
    """单条谱的质量检查。"""
    return spectrum_quality.check_file(path)


def _t_check_group_quality(material: str, max_files: int = 60) -> dict:
    """整组批量检查。

    先靠 spectrum_catalog 把「材料关键字」变成真实路径列表，再交给纯计算的 check_many。
    注意：这层只做「找文件 → 调计算 → 补说明」，不掺任何算法。
    """
    max_files = max(1, min(int(max_files or 60), 200))
    found = spectrum_catalog.find_spectra(DATA_ROOT, str(material), limit=max_files)
    paths = [f["path"] for f in found["files"]]
    if not paths:
        raise ValueError(f"没有找到匹配「{material}」的光谱文件")

    rep = spectrum_quality.check_many(paths, max_files=max_files)
    # check_many 给的是完整版（逐条明细都在），直接塞给模型会撑爆上下文，
    # 所以这里换成摘要视图；数量和结论全保留，只砍逐条明细。
    rep = spectrum_quality.slim_group_report(rep)
    rep["group"] = found["files"][0]["dir"]
    rep["n_matched_in_group"] = found["n_matched"]
    if found["n_matched"] > len(paths):
        rep["partial"] = (
            f"这组共有 {found['n_matched']} 条，本次只抽查了前 {len(paths)} 条"
            f"（受 max_files={max_files} 限制）。"
        )
    return rep


def _t_plot_spectra(paths: list[str], out_name: str | None = None) -> dict:
    from plot_spectra_html import build  # 根目录脚本，只在需要出图时才导入

    if not paths:
        raise ValueError("paths 不能为空")
    name = out_name or "spectra.html"
    if not name.lower().endswith(".html"):
        name += ".html"
    # 只取文件名，禁止路径穿越
    name = os.path.basename(name)
    out = os.path.join(OUTPUT_DIR, name)
    build([str(p) for p in paths], out)
    return {
        "html_path": out,
        "n_spectra": len(paths),
        "note": "已经存成网页，让用户用浏览器打开这个路径查看。",
    }


def _t_analyze_element_lines(path: str, tol_nm: float | None = None) -> dict:
    """元素谱线匹配。

    注意这一层**不做任何化学判断**，只把纯计算模块的结果整理成合适的大小
    交给模型。凡是要说「有/没有某元素」的话，都必须基于这里返回的 verdict 字段。
    """
    rep = element_analysis.analyze_file(path, tol_nm=tol_nm)
    return _slim_element_report(rep)


def _slim_element_report(rep: dict[str, Any], max_peaks_shown: int = 12) -> dict:
    """把元素匹配的完整结果压成「够模型下结论、又不会撑爆上下文」的样子。

    砍的原则：**只砍重复的明细，绝不砍结论**。
      · 元素表（含判定和理由）原样保留 —— 这是模型唯一能引用结论的地方；
      · caveats 原样保留 —— 这是防止模型说错话的护栏；
      · 峰列表只留最强的若干个（模型不需要看 40 个峰的全部候选线）；
      · 谱线库只留几个关键数字，不把整份统计塞进去。
    """
    rows = rep.get("elements", [])
    peaks = []
    for p in rep.get("peaks", [])[:max_peaks_shown]:
        cand = p.get("candidates", [])
        peaks.append({
            "wavelength_nm": p["wavelength_nm"],
            "snr": round(p["snr"], 1),
            "n_candidates": p["n_candidates"],
            "n_base_elements": p["n_base_elements"],
            "top_candidates": [
                f"{c['state']}@{c['line_wavelength_nm']:.3f}(D{c['delta_nm']:.3f})"
                for c in cand[:3]
            ],
            "near_miss": p.get("near_miss"),
        })

    lib = rep.get("library", {}) or {}
    pf = rep.get("peak_finding", {}) or {}
    th = rep.get("thresholds_used", {}) or {}
    return {
        "ok": rep.get("ok"),
        "name": rep.get("name"),
        "source": rep.get("source"),
        "verdict": rep.get("verdict"),
        "n_peaks_analyzed": rep.get("n_peaks"),
        "n_peaks_listed": rep.get("n_peaks_listed"),
        "n_peaks_ambiguous": rep.get("n_peaks_ambiguous"),
        "n_peaks_near_miss": rep.get("n_peaks_near_miss"),
        "peak_selection": pf.get("selection"),
        "n_peaks_found": pf.get("n_peaks_found"),
        "n_excluded_as_spike": pf.get("n_excluded_as_spike"),
        "elements": rows,
        "n_elements_reported": rep.get("n_elements_reported"),
        "n_elements_judged_worth_review": sum(
            1 for r in rows if isinstance(r, dict) and r.get("verdict") == "有支持"),
        "near_miss_peaks": (rep.get("near_miss_peaks") or [])[:5],
        "peaks_shown": len(peaks),
        "peaks": peaks,
        "library": {
            "n_lines_kept": lib.get("n_lines_kept"),
            "density_lines_per_nm": lib.get("density_lines_per_nm"),
            "n_elements": lib.get("n_elements"),
            "sensitive_lines_kept": lib.get("sensitive_lines_kept"),
            "filter": lib.get("filter"),
        },
        "thresholds_used": th,
        "caveats": rep.get("caveats", []),
    }


def _t_generate_report(
    material: str | None = None,
    paths: list[str] | None = None,
    max_files: int = 60,
    element_files: int = 3,
    out_name: str | None = None,
) -> dict:
    """汇总一份检测报告（Step 8）。

    这一层只做参数兜底和裁剪，**不碰任何结论** —— 报告怎么写、写什么，
    全在 tools/report.py 里（那才是能被复用、能被单独测试的纯计算层）。
    """
    if not material and not paths:
        raise ValueError(
            "要么给 material（一组数据的关键字，如 316），要么给 paths（文件路径列表）。"
            "不知道有哪些数据就先调 list_available_spectra。"
        )
    max_files = max(1, min(int(max_files or 60), 200))
    element_files = max(0, min(int(element_files if element_files is not None else 3), 5))

    return report.generate_report(
        material=material,
        paths=paths,
        max_files=max_files,
        element_files=element_files,
        out_name=out_name,
    )


def _fmt_smooth_scan(x: dict[str, Any]) -> str:
    """把平滑候选扫描表的一行压成可读短串。

    强度的量纲随方法变：savgol/moving 是「窗口 nm / 点数」，whittaker 是 λ。
    """
    if x.get("lambda") is not None:
        label = "λ=%g" % x["lambda"]
    else:
        label = "%snm/%s点" % (x.get("window_nm"), x.get("window_points"))
    return "%s:Δ%+.2f%%" % (label, x.get("peak_change_pct") or 0)


def _slim_preprocess_report(res: dict[str, Any], max_outputs: int = 8) -> dict:
    """把预处理结果压成「模型够用、又不撑上下文」的样子。

    原则和元素匹配那次一样：**砍重复明细，绝不砍结论和护栏**。
      · warnings 全保留 —— 这是模型必须转述的边界；
      · output_path 全保留 —— 这是模型下一步要用的东西；
      · 每步只留关键几个数，平滑的候选扫描表压成一行一个的可读字符串。
    """
    outs: list[dict[str, Any]] = []
    for o in (res.get("outputs") or [])[:max_outputs]:
        steps: list[dict[str, Any]] = []
        for s in o.get("steps") or []:
            e = s.get("effect") or {}
            item: dict[str, Any] = {"step": s.get("step"), "method": s.get("method")}
            if s.get("step") == "interpolate":
                item.update({
                    "points": [e.get("n_points_before"), e.get("n_points_after")],
                    "step_nm": round(e.get("step_nm") or 0, 4),
                    "range_nm": [round(v, 3) for v in (e.get("actual_range_nm") or [])],
                    "clipped": e.get("clipped"),
                    "max_merge_ratio": e.get("max_merge_ratio"),
                    "extrapolated": e.get("extrapolated"),
                })
            elif s.get("step") == "baseline":
                item.update({
                    "peak": [round(e.get("peak_before") or 0), round(e.get("peak_after") or 0)],
                    "removed_median": round(e.get("removed_median") or 0, 1),
                    "n_points_negative": e.get("n_points_going_negative"),
                })
            elif s.get("step") == "smooth":
                # ★ 平滑强度的量纲随方法变（SG/滑动平均是窗口 nm 与点数；whittaker 是 λ），
                #   所以不能写死字段名，按实际出现的那一个填 —— 否则 whittaker 会读出一堆 None。
                item.update({
                    "chosen_by": e.get("chosen_by"),
                    "window_note": e.get("note"),
                    "peak_change_pct": e.get("peak_change_pct"),
                    "peak_change_note": e.get("peak_change_note"),
                    "peak_fwhm_points": e.get("peak_fwhm_points"),
                    "peak_fwhm_nm": e.get("peak_fwhm_nm"),
                })
                if e.get("skipped"):
                    item["skipped"] = True
                if e.get("lambda") is not None:
                    item["lambda"] = e.get("lambda")
                    item["order"] = e.get("order")
                if e.get("window_points") is not None:
                    item["window_points"] = e.get("window_points")
                    if e.get("window_nm_effective") is not None:
                        item["window_nm_effective"] = round(e["window_nm_effective"], 3)
                if e.get("window_scan"):
                    item["scan_strength_change"] = [
                        _fmt_smooth_scan(x) for x in e["window_scan"]
                    ]
            elif s.get("step") == "normalize":
                item.update({"based_on": e.get("based_on"),
                             "divisor": e.get("divisor")})
            steps.append(item)

        outs.append({
            "original_path": o.get("original_path"),
            "output_path": o.get("output_path"),
            "manifest_path": o.get("manifest_path"),
            "n_points": [o.get("n_points_before"), o.get("n_points_after")],
            "range_nm": [round(v, 3) for v in (o.get("range_nm") or [])],
            "steps": steps,
        })

    return {
        "ok": res.get("ok"),
        "plan_human": res.get("plan_human"),
        "intensity_units": res.get("intensity_units"),
        "n_spectra": res.get("n_spectra"),
        "output_dir": res.get("output_dir"),
        "grid_note": res.get("grid_note"),
        "outputs": outs,
        "n_outputs_shown": len(outs),
        "warnings": res.get("warnings") or [],
        "hint": (
            "后续分析请用 output_path 那个文件（它已经是标准两列格式，"
            "analyze_element_lines / plot_spectra 能直接读）。"
            "★ 但质量检查（check_spectrum_quality / check_group_quality）必须用**原始谱**，"
            "不要用预处理后的谱。"
            "★ warnings 里每一条都要如实转述给用户。"
        ),
    }


def _t_preprocess_spectrum(
    paths: list[str],
    interpolate: str = "uniform",
    grid_step_nm: float | None = None,
    grid_range: list[float] | None = None,
    baseline: str = "none",
    smooth: str = "savgol",
    smooth_window_nm: float | None = None,
    normalize: str = "none",
    norm_line_nm: float | None = None,
) -> dict:
    """预处理入口（Step 12）。

    这一层只做参数兜底和裁剪，**不做任何算法** ——
    算法与落盘都在 tools/preprocess.py（那才是能被单独测试的纯计算层）。
    """
    if isinstance(paths, str):
        paths = [paths]
    paths = [str(p) for p in (paths or []) if str(p).strip()]
    if not paths:
        raise ValueError("paths 不能为空：先跑 list_available_spectra 拿到真实文件路径。")
    if len(paths) > 25:
        raise ValueError(
            f"一次最多处理 25 条，收到 {len(paths)} 条。请分批处理。")
    if grid_range is not None and len(grid_range) != 2:
        raise ValueError("grid_range 要写成 [下限, 上限] 两个数，例如 [180, 812]。")

    plan = preprocess.PreprocessPlan(
        interpolate=interpolate,
        grid_step_nm=grid_step_nm,
        grid_range=tuple(grid_range) if grid_range else None,
        baseline=baseline,
        smooth=smooth,
        smooth_window_nm=smooth_window_nm,
        normalize=normalize,
        norm_line_nm=norm_line_nm,
    )
    return _slim_preprocess_report(preprocess.preprocess_files(paths, plan))


_REGISTRY: dict[str, Callable[..., dict]] = {
    "list_available_spectra": _t_list_available_spectra,
    "describe_material_grid": _t_describe_material_grid,
    "load_spectrum_summary": _t_load_spectrum_summary,
    "check_spectrum_quality": _t_check_spectrum_quality,
    "check_group_quality": _t_check_group_quality,
    "plot_spectra": _t_plot_spectra,
    "analyze_element_lines": _t_analyze_element_lines,
    "generate_report": _t_generate_report,
    "preprocess_spectrum": _t_preprocess_spectrum,
}


# ---------------------------------------------------------------------------
# 3) 执行入口
# ---------------------------------------------------------------------------
def execute(name: str, arguments: dict[str, Any] | str | None) -> dict:
    """跑一个工具，永远返回 dict（出错也返回 dict，不抛异常）。

    为什么要吃掉异常？
      工具报错不应该让整个 Agent 崩掉 —— 应该把错误当成一条「观察结果」
      交回给模型，让它自己决定换个参数、换个文件，或者如实告诉用户做不到。
      这才是「Agent 能自己纠错」的关键。
    """
    if name not in _REGISTRY:
        return {"ok": False, "error": f"没有这个工具：{name}", "available": TOOL_NAMES}

    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments) if arguments.strip() else {}
        except json.JSONDecodeError as exc:
            return {"ok": False, "error": f"参数不是合法 JSON：{exc}", "raw": arguments[:200]}
    arguments = arguments or {}

    try:
        result = _REGISTRY[name](**arguments)
        return {"ok": True, "tool": name, "result": result}
    except TypeError as exc:  # 参数名/个数不对
        return {"ok": False, "tool": name, "error": f"参数不对：{exc}"}
    except Exception as exc:
        return {"ok": False, "tool": name, "error": f"{type(exc).__name__}: {exc}"}


def _slim(payload: Any, max_items: int, max_str: int) -> Any:
    """递归瘦身：长列表只留前几项，长字符串截断并标注。

    注意这跟「把 JSON 字符串切一半」完全不同 —— 这里动的是**结构**，
    切完仍然是一个合法的、字段齐全的 JSON（只是明细少了）。
    """
    if isinstance(payload, dict):
        return {k: _slim(v, max_items, max_str) for k, v in payload.items()}
    if isinstance(payload, list):
        head = [_slim(v, max_items, max_str) for v in payload[:max_items]]
        if len(payload) > max_items:
            head.append(f"...（另有 {len(payload) - max_items} 项已省略）")
        return head
    if isinstance(payload, str) and len(payload) > max_str:
        return payload[:max_str] + f"...（省略 {len(payload) - max_str} 字）"
    return payload


def execute_to_json(name: str, arguments: dict[str, Any] | str | None) -> str:
    """给模型看的字符串形态：**永远是合法 JSON**；太长就按结构瘦身，不切字符串。

    为什么不「字符串切一半再拼个括号」？
      那样返回的是断掉的 JSON，模型解析不动、只能瞎猜，比不给还糟。
      所以这里逐级降级：
        1) 原样返回；
        2) 列表只留前 8 项、字符串留前 300 字；
        3) 更狠：列表留 2 项、字符串留前 120 字；
        4) 实在还超，才退化成「只报信封 + 前 400 字预览」并**显式标注已截断**。
    """
    payload = execute(name, arguments)
    text = json.dumps(payload, ensure_ascii=False, default=str)
    if len(text) <= _MAX_CHARS:
        return text

    for max_items, max_str in ((8, 300), (2, 120)):
        slimmed = _slim(payload, max_items, max_str)
        text = json.dumps(slimmed, ensure_ascii=False, default=str)
        if len(text) <= _MAX_CHARS:
            return text

    return json.dumps(
        {
            "ok": payload.get("ok"),
            "tool": name,
            "truncated": True,
            "result_keys": sorted(payload.get("result", {}).keys())
            if isinstance(payload.get("result"), dict) else None,
            "preview": json.dumps(payload, ensure_ascii=False, default=str)[:400],
            "note": "结果太长已截断（结构已保留）。请减小 limit / max_files，或用更精确的 query 重新调用。",
        },
        ensure_ascii=False,
        default=str,
    )
