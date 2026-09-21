r"""
main.py —— LIBS AI Agent 主程序（命令行入口）。

运行方式（在 <项目根目录> 下）：
    .\.venv\Scripts\Activate.ps1

    # 问一句就退出
    python main.py "304 测了哪些点？"

    # 连续对话（带工具）
    python main.py

    # 只想聊天、不让它碰磁盘
    python main.py --no-tools "你好"

    # 看看现在挂了哪些工具
    python main.py --list-tools

    # 把整段对话记到文件里（排查问题时很有用）
    python main.py "307 有没有饱和" --log run.log

和 agent/chat.py 的关系：
    chat.py 是「纯聊天终端」；main.py 是「正式入口」，多了单次提问、--log、
    --list-tools 这些方便脚本化和排查的开关。两者都走同一个 Agent 循环。
"""

from __future__ import annotations

import argparse
import io
import sys

from agent.agent import run_agent
from agent.llm_client import chat, get_settings, to_message_dict
from agent.prompts import LIBS_AGENT_SYSTEM_PROMPT
from agent.tool_registry import TOOLS


class Tee:
    """同时往终端和日志文件写。"""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)

    def flush(self):
        for st in self.streams:
            st.flush()


def enable_lenient_output() -> None:
    for stream in (sys.__stdout__, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except Exception:
            pass


def print_tools() -> None:
    print(f"当前挂了 {len(TOOLS)} 个工具：\n")
    for t in TOOLS:
        fn = t["function"]
        params = fn["parameters"].get("properties", {})
        req = fn["parameters"].get("required", [])
        print(f"  ● {fn['name']}")
        print(f"      {fn['description'].splitlines()[0]}")
        for pname, pmeta in params.items():
            flag = "必填" if pname in req else "可选"
            print(f"        - {pname} ({pmeta.get('type','any')}, {flag})")
        print()


def show_event(verbose: bool):
    def on_event(ev: dict) -> None:
        if not verbose:
            return
        if ev["type"] == "tool_call":
            print(f"  [工具] 调用 {ev['tool']}")
            print(f"         参数 {ev['args']}")
        elif ev["type"] == "tool_result":
            print(f"  [工具] {ev['tool']} -> {'成功' if ev['ok'] else '失败'}")
    return on_event


def ask_once(question: str, use_tools: bool, verbose: bool, short: str) -> str:
    if not use_tools:
        reply = chat([{"role": "user", "content": question}])
        return reply.content or ""
    result = run_agent(question, on_event=show_event(verbose))
    return result["answer"]


def main() -> int:
    ap = argparse.ArgumentParser(
        description="LIBS AI Agent —— 能读你真实光谱的大模型智能体",
    )
    ap.add_argument("question", nargs="*", help="要问的问题；留空则进入连续对话")
    ap.add_argument("--no-tools", action="store_true", help="关闭工具调用，退回纯聊天")
    ap.add_argument("--quiet", action="store_true", help="不显示工具调用细节")
    ap.add_argument("--max-steps", type=int, default=8, help="单次提问最多几轮工具调用")
    ap.add_argument("--list-tools", action="store_true", help="列出所有工具后退出")
    ap.add_argument("--log", default=None, help="把输出同时写入这个文件")
    args = ap.parse_args()

    logfh = None
    if args.log:
        logfh = io.open(args.log, "w", encoding="utf-8", errors="replace")
        sys.stdout = Tee(sys.__stdout__, logfh)

    try:
        if args.list_tools:
            print_tools()
            return 0

        cfg = get_settings()
        short = cfg["short"]

        # ---------- 单次提问 ----------
        if args.question:
            q = " ".join(args.question)
            print(f"你 > {q}")
            if args.no_tools:
                print("（已关闭工具，纯聊天模式）")
            answer = ask_once(q, not args.no_tools, not args.quiet, short)
            print(f"{short} > {answer}")
            return 0

        # ---------- 连续对话 ----------
        # 直接交给 chat.py 的 REPL，它本身就带工具；这里不再重复实现一遍。
        import agent.chat as chat_mod

        return chat_mod.main()
    finally:
        if logfh:
            sys.stdout = sys.__stdout__
            logfh.close()


if __name__ == "__main__":
    enable_lenient_output()
    raise SystemExit(main())
