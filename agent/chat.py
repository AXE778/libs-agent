r"""多轮对话终端程序 —— 现在它是「带工具的 Agent」了（Step 5 + Step 6）。

运行方式：
    cd <项目根目录>
    .\.venv\Scripts\Activate.ps1
    python -m agent.chat

跟上一版的区别（很重要）：
    上一版：模型只能「说」。问它数据，它看不见你的磁盘，只能编。
    这一版：模型能「看」。它会先列文件、再读光谱，每个数字都来自真实文件。
           而且你会在屏幕上**看见它调用了什么工具**（带 [工具] 前缀的行）。

可用指令：
    /exit  或 /quit    退出
    /reset              清空对话历史，重新开始
    /tools              开关「工具调用」（默认开）
    /system             开关「LIBS 智能体人设」（默认开）

为什么人设现在默认开了？
    agent/prompts.py 是工具导向的（要求「遇到数据类问题必须先调工具」）。
    Step 5 之前没有工具，带着它模型只会一直说「我需要调用工具」，
    所以那时默认关。现在工具接上了，它才真正派上用场。
"""

from __future__ import annotations

import sys

from agent.agent import run_agent
from agent.llm_client import chat, get_settings, to_message_dict
from agent.prompts import LIBS_AGENT_SYSTEM_PROMPT
from agent.tool_registry import TOOL_NAMES

EXIT_COMMANDS = {"/exit", "/quit", "/q"}
WIDTH = 64


def enable_lenient_output() -> None:
    """让终端输出「绝不因为编码问题崩溃」。

    Windows 控制台默认编码是 GBK。万一模型或文件名里有 GBK 编不出的字符
    （生僻字、特殊符号），print 会抛 UnicodeEncodeError 把程序打断。
    把错误处理降级成「打一个替代字符」，对话就不会莫名挂掉。
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except Exception:
            pass


def build_messages(use_persona: bool) -> list[dict]:
    if use_persona:
        return [{"role": "system", "content": LIBS_AGENT_SYSTEM_PROMPT}]
    return []


def make_printer():
    """把 Agent 的每一步显示出来 —— 不显示的话，你会以为它卡住了。"""

    def on_event(ev: dict) -> None:
        if ev["type"] == "tool_call":
            print(f"    [工具] 调用 {ev['tool']}  参数 {ev['args']}")
        elif ev["type"] == "tool_result":
            flag = "成功" if ev["ok"] else "失败"
            print(f"    [工具] {ev['tool']} 返回：{flag}")

    return on_event


def main() -> int:
    enable_lenient_output()
    cfg = get_settings()  # Key 没填的话，这里会给人话提示并退出
    short = cfg["short"]

    use_tools = True
    use_persona = True
    messages = build_messages(use_persona)

    print("=" * WIDTH)
    print(" LIBS AI Agent —— 带工具的对话")
    print("=" * WIDTH)
    print(f" 模型   {cfg['model']}")
    print(f" 服务   {cfg['label']}")
    print(f" 工具   {'开启' if use_tools else '关闭'}（{len(TOOL_NAMES)} 个：{', '.join(TOOL_NAMES)}）")
    print(" 指令   /exit 退出   /reset 清空   /tools 开关工具   /system 开关人设")
    print("=" * WIDTH)
    print(" 试试问： 304 测了哪些点？        /        看看 data-304-1 的 X0-Y0 那条谱")
    print("=" * WIDTH)
    print()

    while True:
        try:
            line = input("你 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not line:
            continue

        if line in EXIT_COMMANDS:
            break

        if line == "/reset":
            messages = build_messages(use_persona)
            print("（对话历史已清空，从头开始）\n")
            continue

        if line == "/tools":
            use_tools = not use_tools
            messages = build_messages(use_persona)
            print(f"（工具调用{'已开启' if use_tools else '已关闭'}，历史已清空）\n")
            continue

        if line == "/system":
            use_persona = not use_persona
            messages = build_messages(use_persona)
            print(f"（LIBS 智能体人设{'已开启' if use_persona else '已关闭'}，历史已清空）\n")
            continue

        print(f"[{short} 处理中…]")

        # ---------- 带工具的路线：交给 Agent 循环 ----------
        if use_tools:
            try:
                result = run_agent(line, messages, on_event=make_printer())
            except SystemExit:
                # chat() 遇到鉴权 / 模型 / 网络错误时会打印人话提示并 sys.exit。
                # 在对话程序里直接退出太粗暴，这里撤回刚加的那句，继续下一轮。
                if messages and messages[-1].get("role") == "user":
                    messages.pop()
                print("（本轮失败，已撤回这句话。可以 /reset 重来，或直接再问一次）\n")
                continue
            messages = result["messages"]
            print(f"{short} > {result['answer']}\n")
            continue

        # ---------- 不带工具的路线：退回 Step 2 的纯聊天 ----------
        messages.append({"role": "user", "content": line})
        try:
            reply = chat(messages)
        except SystemExit:
            messages.pop()
            print("（本轮失败，已撤回这句话。可以 /reset 重来，或直接再问一次）\n")
            continue
        # 必须用 to_message_dict()：Kimi 思考模型要求 reasoning_content 原样回传
        messages.append(to_message_dict(reply))
        print(f"{short} > {reply.content}\n")

    print("（对话结束）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
