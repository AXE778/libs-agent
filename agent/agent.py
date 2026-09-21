r"""
agent.agent —— Agent 主循环（Step 6）。

============================ 这就是「Agent」的全部 ============================

一句话：**让模型自己决定「要不要调工具、调哪个、拿结果再想一遍」，循环到它不再调为止。**

伪代码只有 5 行：

    messages = [系统提示, 用户问题]
    while True:
        msg = 调模型(messages, 工具说明书)
        messages.append(msg)
        if msg 没有 tool_calls:  break        # 它决定直接回答了
        for 每个 tool_call:
            messages.append(工具执行结果)

看起来简单，但有两个坑必须处理对：

坑 1：**回传 assistant message 必须用 llm_client.to_message_dict()**
      （Kimi 思考模型要求把 reasoning_content 原样带上，手写 dict 会丢字段）

坑 2：**一次 assistant 消息里的所有 tool_calls，必须一次性全部回完**
      不能只回一个就再次请求模型（接口会报「tool_call_id 不匹配」）。

跟 Step 2 的聊天有什么区别？
    Step 2：模型只能「说」。它看不见你的磁盘，问它数据它只会编。
    Step 6：模型能「看」。它先列文件、再读光谱，每个数字都来自真实文件。
"""

from __future__ import annotations

import json
from typing import Any, Callable

from . import llm_client
from .prompts import LIBS_AGENT_SYSTEM_PROMPT
from .tool_registry import TOOLS, execute_to_json

__all__ = ["new_messages", "run_agent", "DEFAULT_MAX_STEPS"]

DEFAULT_MAX_STEPS = 8  # 防止模型来回绕圈；正常问一句用不了 3 步


def new_messages(system_prompt: str | None = LIBS_AGENT_SYSTEM_PROMPT) -> list[dict]:
    """开一段新对话。system_prompt 传 None 就不带人设。"""
    return [{"role": "system", "content": system_prompt}] if system_prompt else []


def run_agent(
    user_input: str,
    messages: list[dict] | None = None,
    *,
    max_steps: int = DEFAULT_MAX_STEPS,
    on_event: Callable[[dict], None] | None = None,
) -> dict[str, Any]:
    """跑一轮「用户提问 → 工具 → 回答」。

    Args:
        user_input: 用户这句话
        messages:   已有的对话历史（会被原地继续追加，便于多轮）
        max_steps:  最多允许模型「思考 + 调工具」几轮
        on_event:   回调，用来实时显示「正在调用 xxx」。
                    事件形如 {"type": "tool_call", "tool": ..., "args": ...}
                    或 {"type": "tool_result", "tool": ..., "ok": True}

    Returns:
        {"answer": str, "messages": list, "steps": list, "stop": "final"|"max_steps"}
    """
    msgs = messages if messages is not None else new_messages()
    msgs.append({"role": "user", "content": user_input})
    steps: list[dict[str, Any]] = []

    def emit(ev: dict) -> None:
        if on_event:
            on_event(ev)

    for round_no in range(1, max_steps + 1):
        message = llm_client.chat(msgs, tools=TOOLS)

        # 关键：全量转 dict 再放回历史（别手写 {"role":"assistant","content":...}）
        msgs.append(llm_client.to_message_dict(message))

        tool_calls = getattr(message, "tool_calls", None)

        # 模型没要求调工具 → 这就是最终回答，收工
        if not tool_calls:
            return {
                "answer": message.content or "",
                "messages": msgs,
                "steps": steps,
                "stop": "final",
            }

        # 有工具要调：一次全调完，结果按 tool_call_id 回传
        for call in tool_calls:
            name = call.function.name
            raw_args = call.function.arguments
            emit({"type": "tool_call", "tool": name, "args": raw_args})

            result_text = execute_to_json(name, raw_args)
            msgs.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": result_text,
            })

            try:
                parsed = json.loads(result_text)
            except json.JSONDecodeError:
                parsed = {"ok": None}
            steps.append({
                "round": round_no,
                "tool": name,
                "args": raw_args,
                "ok": parsed.get("ok"),
            })
            emit({"type": "tool_result", "tool": name, "ok": parsed.get("ok"),
                  "preview": result_text[:300]})

    # 用完了步数还没收敛：如实告诉用户，不要假装答完
    return {
        "answer": (
            f"（已达到最大工具调用轮数 {max_steps}，对话被中断。）\n"
            "建议把问题拆小一点，例如先问「304 有哪些测点」，再指定具体文件让它读。"
        ),
        "messages": msgs,
        "steps": steps,
        "stop": "max_steps",
    }
