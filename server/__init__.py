"""server 包：把 Agent 包成 HTTP 服务（Step 10）。

设计约束（和 agent/、tools/ 都不同）：
  这一层**只做协议转换** —— 把 HTTP 请求翻译成 `run_agent()` 的一次调用，
  再把结果翻译回 JSON。
  它不碰算法、不碰 prompt、不碰工具表。将来换成别的传输方式（WebSocket、
  gRPC、消息队列），换的只是这一层。

为什么单独一个包而不是塞进 agent/：
  agent/ 的定位是「决策」，它对「有没有 HTTP」一无所知，
  所以命令行（main.py）、终端聊天（agent/chat.py）、HTTP 服务（server/）
  三者可以共用同一套 Agent 逻辑，互不影响。
"""
