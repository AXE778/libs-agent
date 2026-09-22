r"""scripts/latency_probe.py —— 回答速度探针：拆开一次提问的时间都花在哪。

为什么需要它？
    「Agent 回复慢」这件事，凭直觉猜一定会猜错。实测（2026-09-22）结果显示：
    **96% 的时间花在模型往返上，工具执行只占 0.2～2 秒**。
    所以优化方向必须是「减少模型往返次数」，而不是「把工具写快一点」。
    这个脚本就是把这件事量化出来，改完代码可以立刻复测。

跑法（在 <项目根目录> 下）：
    .\.venv\Scripts\python.exe scripts\latency_probe.py "304 测了哪些点？"
    .\.venv\Scripts\python.exe scripts\latency_probe.py "316 那组数据质量怎么样？"

它会打三样东西：
    1. 每次**模型调用**的耗时，以及这一轮发出去多少字符（上下文在滚雪球，要盯住）
    2. 每次**工具调用**的名字、参数、耗时、返回大小
    3. 汇总：几次模型调用 / 几次工具调用 / 总耗时

★ 记两个「好」的基线（清点类问题，如「304 测了哪些点」）：
      模型调用 ≤ 2，工具调用 ≤ 1，总耗时 ≈ 8～15 秒。
  如果某个问题忽然变成 3～4 次模型调用，先看它是不是多调了一个能省掉的工具。
  （2026-09-22 就从 4 次降到 2 次：见 agent/prompts.py 里 list_available_spectra 那条★）
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, ROOT)

from agent import agent as agent_mod  # noqa: E402
from agent import llm_client  # noqa: E402
from agent.prompts import LIBS_AGENT_SYSTEM_PROMPT  # noqa: E402
from agent.tool_registry import TOOLS  # noqa: E402

_t0 = time.perf_counter()
_stats = {"model": [], "tool": []}


def _fmt_args(raw) -> str:
    if isinstance(raw, str):
        return raw
    try:
        return json.dumps(raw, ensure_ascii=False)
    except Exception:
        return str(raw)


def _patch() -> None:
    """包住模型调用与工具执行，只记录耗时，不改变行为。"""
    orig_chat = llm_client.chat

    def timed_chat(messages, tools=None, **kw):
        payload_chars = sum(len(json.dumps(m, ensure_ascii=False)) for m in messages)
        if tools:
            payload_chars += len(json.dumps(tools, ensure_ascii=False))
        t = time.perf_counter()
        out = orig_chat(messages, tools=tools, **kw)
        dt = time.perf_counter() - t
        _stats["model"].append(dt)
        print(f"    [模型] {dt:6.2f}s   发出 {len(messages)} 条消息、{payload_chars} 字符",
              flush=True)
        return out

    llm_client.chat = timed_chat
    agent_mod.llm_client.chat = timed_chat

    orig_exec = agent_mod.execute_to_json

    def timed_exec(name, raw_args):
        t = time.perf_counter()
        out = orig_exec(name, raw_args)
        dt = time.perf_counter() - t
        _stats["tool"].append((name, dt))
        print(f"    [工具] {name}({_fmt_args(raw_args)}) {dt:6.2f}s  → {len(out)} 字符",
              flush=True)
        return out

    agent_mod.execute_to_json = timed_exec


def probe(question: str) -> None:
    _stats["model"].clear()
    _stats["tool"].clear()
    print("=" * 70)
    print(f"问题：{question}")
    print("-" * 70)
    t = time.perf_counter()
    try:
        result = agent_mod.run_agent(question, on_event=lambda ev: None, max_steps=8)
        ok, err = True, ""
    except Exception as exc:
        result, ok, err = None, False, f"{type(exc).__name__}: {exc}"
    total = time.perf_counter() - t

    print("-" * 70)
    if ok:
        print(f"stop={result['stop']}  工具调用 {len(_stats['tool'])} 次  "
              f"模型调用 {len(_stats['model'])} 次  答案 {len(result['answer'])} 字符")
    else:
        print("失败：", err)
    tool_s = sum(d for _, d in _stats["tool"])
    model_s = sum(_stats["model"])
    print(f"★ 总耗时 {total:6.2f}s   其中模型 {model_s:.2f}s"
          f"（{model_s / total * 100:.0f}%） 工具 {tool_s:.2f}s")


def main() -> int:
    questions = sys.argv[1:] or ["304 测了哪些点？"]
    print(f"固定开销：系统提示 {len(LIBS_AGENT_SYSTEM_PROMPT)} 字符"
          f" + 工具说明书 {len(json.dumps(TOOLS, ensure_ascii=False))} 字符"
          f"（每轮都要重发，见下面每行的「发出 N 字符」）")
    _patch()
    for q in questions:
        probe(q)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
