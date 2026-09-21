r"""
server.app —— Step 10：把 Agent 包成一个 HTTP 服务（FastAPI）。

================================ 为什么要有这一步 ================================

到这里为止，「Agent」只能活在两个地方：命令行（main.py）和终端聊天（agent/chat.py）。
要说它是「一个软件」，还差一件事：**能被别的程序调用。**

Step 11 的 Qt 上位机不可能去 import 你的 Python 模块再自己拼 messages ——
它是另一个进程、另一种语言（C++）。两个进程之间最省事的通信方式就是 HTTP：
    Qt 发一个 JSON 请求 → 这里跑一遍 Agent → 返回一个 JSON。

所以这一层的职责就一句话：**做协议转换，不做别的。**

端点
----
    GET    /                    调试页面（浏览器直接打开就能聊，见下方说明）
    GET    /health              健康检查：服务商 / 模型 / 工具数（不调用大模型，很快）
    GET    /tools               列出当前挂载的工具（给客户端做「工具面板」用）
    POST   /chat                问一句，等它答完（一次请求返回完整答案）
    GET    /sessions            看现在有哪些会话（调试用）
    DELETE /sessions/{id}       清空某个会话

-------------------------------- 六条设计决定 --------------------------------

① **会话状态放在服务端内存里，客户端只传一个 session_id。**

   为什么不让客户端自己维护对话历史？因为 Kimi 的思考模型要求把整条
   assistant message（含 `reasoning_content`）原样回传，手写历史必然丢字段。
   让 Qt 端去管这件事，等于把 Step 6 踩过的坑在另一种语言里再踩一遍。
   服务端持有 `messages` 列表，客户端只有一个 id —— 简单、且不会错。

   代价：服务重启会话就没了。这是刻意的，实验室单机服务不需要「跨重启记住聊天」。

② **第一版不做流式（SSE）。**

   这个账号每分钟只允许 3 次请求，一次提问要 3~5 个请求、**必然撞限流**，
   单次请求可能等 20 秒以上。流式只会让客户端更难做超时和断线重连。
   先做「一次请求 → 完整答案」，把过程信息（调了哪些工具、各花多久）
   放进响应体的 `steps` 字段。客户端拿到后一次性渲染即可。

③ **端点写成 `def` 而不是 `async def`。**

   `run_agent()` 是同步阻塞的，里面还有 `time.sleep()` 等着重试限流。
   写成 `async def` 会把整个事件循环卡住（期间连 /health 都不响应）；
   写成 `def`，FastAPI 会自动把它丢进线程池。

④ **同一个会话同一时刻只允许一个请求在跑。**

   `messages` 是一个共享的可变列表。两个请求并发进来会交叉追加，
   导致 `tool_call_id` 对不上、对话彻底乱掉 —— 这种 bug 极难查。
   所以每个会话一把锁，撞上了就返回 409，并明确告诉客户端「等上一句答完再发」。

⑤ **失败的那一轮必须回滚。**

   如果模型调用中途挂了（Key 过期、断网、限流重试耗尽），
   `messages` 里可能已经塞进半条 user / assistant / tool 消息。
   留着它会污染下一轮，所以出错时把会话恢复成本轮开始前的样子。

⑥ **默认只监听 127.0.0.1。**

   这是实验室里的服务，默认不该暴露到局域网。要给别人访问就显式传
   `LIBS_API_HOST=0.0.0.0`，但那意味着**任何人都能调你付费的大模型 Key**，
   最好先确认网络环境。

运行
----
    cd <项目根目录>
    .\.venv\Scripts\Activate.ps1
    python -m server.app
    # 或者：python -m uvicorn server.app:app --port 8000

然后浏览器打开 http://127.0.0.1:8000 —— 有一个自带的调试页面。
"""

from __future__ import annotations

import os
import sys
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from agent.agent import DEFAULT_MAX_STEPS, new_messages, run_agent  # noqa: E402
from agent.llm_client import PROVIDER, PROVIDERS, LLMError, get_settings  # noqa: E402
from agent.prompts import LIBS_AGENT_SYSTEM_PROMPT  # noqa: E402
from agent.tool_registry import TOOLS, TOOL_NAMES  # noqa: E402

__all__ = ["app", "create_app", "SessionStore"]

# ---------------------------------------------------------------------------
# 会话表
# ---------------------------------------------------------------------------
MAX_SESSIONS = 32


@dataclass
class Session:
    """一段对话。

    `messages` 就是 Agent 循环用的那个列表（会随每一轮增长），
    `lock` 保证同一会话不会有两个请求同时改它。
    """

    id: str
    messages: list[dict]
    created_at: str
    lock: threading.Lock = field(default_factory=threading.Lock)
    n_turns: int = 0
    last_used: float = field(default_factory=time.time)
    last_error: str | None = None


class SessionStore:
    """进程内的会话表。

    超额时按「最久没用过」淘汰，而不是拒绝新会话 ——
    实验室场景下，客户端换了个 session_id 就整个用不了，比丢一段旧对话难查得多。
    """

    def __init__(self, max_sessions: int = MAX_SESSIONS) -> None:
        self._items: OrderedDict[str, Session] = OrderedDict()
        self._guard = threading.Lock()
        self.max_sessions = max_sessions

    def get_or_create(self, session_id: str | None = None) -> Session:
        with self._guard:
            if session_id:
                sess = self._items.get(session_id)
                if sess is not None:
                    self._items.move_to_end(session_id)
                    sess.last_used = time.time()
                    return sess
                # 客户端带来的 id 不认识了（服务重启过）→ 用它建一个新的，
                # 而不是报错。客户端不必知道服务端重启过。
                sess = Session(id=session_id, messages=new_messages(),
                               created_at=_now())
                self._items[session_id] = sess
                self._evict()
                return sess

            sid = uuid.uuid4().hex[:12]
            sess = Session(id=sid, messages=new_messages(), created_at=_now())
            self._items[sid] = sess
            self._evict()
            return sess

    def drop(self, session_id: str) -> bool:
        with self._guard:
            return self._items.pop(session_id, None) is not None

    def info(self) -> list[dict]:
        with self._guard:
            return [
                {
                    "session_id": s.id,
                    "n_messages": len(s.messages),
                    "n_turns": s.n_turns,
                    "created_at": s.created_at,
                    "idle_s": round(time.time() - s.last_used, 1),
                    "busy": s.lock.locked(),
                    "last_error": s.last_error,
                }
                for s in self._items.values()
            ]

    def _evict(self) -> None:
        while len(self._items) > self.max_sessions:
            self._items.popitem(last=False)


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ---------------------------------------------------------------------------
# 请求 / 响应模型
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000,
                         description="用户这句话")
    session_id: str | None = Field(
        None, description="上一轮返回的 session_id；不传就新建一个会话")
    max_steps: int = Field(
        DEFAULT_MAX_STEPS, ge=1, le=20,
        description="最多允许模型「思考 + 调工具」几轮，防止来回绕圈")


class ToolStep(BaseModel):
    round: int
    tool: str
    ok: bool | None = None
    seconds: float | None = None
    args_preview: str | None = None


class ChatResponse(BaseModel):
    ok: bool = True
    session_id: str
    answer: str
    stop: str = Field(..., description="final = 正常答完；max_steps = 轮数用尽被中断")
    steps: list[ToolStep] = Field(default_factory=list)
    n_tools: int = 0
    elapsed_s: float = 0.0
    provider: str = ""
    model: str = ""
    note: str = ""


# ---------------------------------------------------------------------------
# 应用
# ---------------------------------------------------------------------------
DEBUG_PAGE = r"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>LIBS AI Agent —— 调试页</title>
<style>
body{font-family:system-ui,-apple-system,"Segoe UI",sans-serif;max-width:840px;margin:24px auto;padding:0 16px;color:#2C2C2A;line-height:1.6}
h2{font-weight:500;font-size:18px;margin:0 0 4px}
#meta{font-size:13px;color:#888780;margin-bottom:12px}
textarea{width:100%;height:64px;padding:8px;border:1px solid #ccc;border-radius:8px;font:inherit;box-sizing:border-box}
button{padding:8px 16px;margin-right:8px;border-radius:8px;border:1px solid #534AB7;background:#EEEDFE;color:#26215C;font:inherit;cursor:pointer}
button:disabled{opacity:.5;cursor:default}
#log{border:1px solid #ddd;border-radius:8px;padding:12px;margin-top:12px;min-height:200px;background:#fafafa}
#log div{margin:4px 0;white-space:pre-wrap}
.u{color:#185FA5;font-weight:500}
.a{color:#04342C}
.t{color:#854F0B;font-size:13px}
.e{color:#A32D2D}
</style></head><body>
<h2>LIBS AI Agent —— 调试页</h2>
<div id="meta">正在读取服务信息…</div>
<div class="t">这是 Step 10 自带的调试页（正经客户端是 Step 11 的 Qt 窗口）。一次提问可能要 20 秒以上，撞上限流会自动等待。</div>
<textarea id="q" placeholder="例：看看 data-304-1 那组数据的采集质量"></textarea>
<div style="margin-top:8px">
  <button id="send">发送</button>
  <button id="reset">清空会话</button>
</div>
<div id="log"></div>
<script>
var log=document.getElementById('log'), q=document.getElementById('q'), sid=null;
function add(cls,text){var d=document.createElement('div');d.className=cls;d.textContent=text;log.appendChild(d);log.scrollTop=log.scrollHeight;}
fetch('/health').then(function(r){return r.json();}).then(function(h){
  document.getElementById('meta').textContent='服务商 '+h.provider+'　模型 '+h.model+'　工具 '+h.n_tools+' 个';
});
document.getElementById('reset').onclick=function(){
  if(sid){fetch('/sessions/'+sid,{method:'DELETE'});}
  sid=null;log.innerHTML='';add('t','（会话已清空）');
};
document.getElementById('send').onclick=async function(){
  var m=q.value.trim(); if(!m) return; q.value='';
  add('u','你 > '+m);
  var b=document.getElementById('send'); b.disabled=true; add('t','（处理中…）');
  try{
    var r=await fetch('/chat',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({message:m,session_id:sid})});
    var j=await r.json();
    if(!r.ok){
      var d=j.detail||{};
      add('e','失败：'+(d.message||JSON.stringify(j)));
      return;
    }
    sid=j.session_id;
    add('t','（调了 '+j.n_tools+' 个工具，耗时 '+j.elapsed_s+' 秒）');
    add('a','Agent > '+j.answer);
  }catch(e){ add('e','请求失败：'+e); }
  finally{ b.disabled=false; }
};
q.addEventListener('keydown',function(e){ if(e.ctrlKey&&e.key==='Enter'){document.getElementById('send').click();} });
</script></body></html>
"""


def create_app(store: SessionStore | None = None) -> FastAPI:
    """建一个应用实例。

    做成工厂函数是为了测试：`step10_demo.py` 每次都能拿到一个干净的会话表，
    不会被上一次测试留下的会话干扰。
    """
    api = FastAPI(
        title="LIBS AI Agent API",
        description="激光诱导击穿光谱的 AI 分析服务。数字全部来自 tools/ 的纯计算，"
                    "大模型只负责决定调哪个工具、并解释结果。",
        version="0.1.0",
    )
    sessions = store or SessionStore()

    # 本地单机服务，默认只监听 127.0.0.1，所以这里放开跨域是安全的 ——
    # 方便直接用浏览器 / 网页 / Qt WebEngine 调试。要对外监听请先想清楚。
    api.add_middleware(
        CORSMiddleware,
        allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
    )

    # ---------------- GET / - 调试页面 ----------------
    @api.get("/", response_class=HTMLResponse, include_in_schema=False)
    def index() -> str:
        return DEBUG_PAGE

    # ---------------- GET /health ----------------
    @api.get("/health")
    def health() -> dict:
        """健康检查。**不调用大模型**，所以毫秒级返回，可以被监控高频轮询。

        这里刻意**不调 get_settings()**：它在 Key 没配时会 print 一大段人话然后
        sys.exit。健康检查是会被反复轮询的，每次刷一屏提示既吵，
        又得靠捕获 SystemExit 来兜 —— 不如直接读环境变量。两行的事。

        注意：这里只检查「Key 填了没有」，**不验证 Key 是否有效** ——
        那要花一次真实请求，而你的账号每分钟只有 3 次，不该浪费在健康检查上。
        """
        preset = PROVIDERS.get(PROVIDER, {})
        key = (os.getenv(preset.get("api_key_env", "")) or "").strip()
        return {
            "ok": True,
            "provider": preset.get("label", PROVIDER),
            "model": (os.getenv("LLM_MODEL") or preset.get("model")),
            "base_url": (os.getenv("LLM_BASE_URL") or preset.get("base_url")),
            "api_key_configured": bool(key) and not key.startswith("your_"),
            "n_tools": len(TOOL_NAMES),
            "tools": TOOL_NAMES,
            "n_sessions": len(sessions.info()),
            "server_time": _now(),
        }

    # ---------------- GET /tools ----------------
    @api.get("/tools")
    def list_tools() -> dict:
        """把工具说明书原样给出（客户端可以做「工具面板」）。

        注意：`description` 里写着每个工具的**边界**，那是给模型看的纪律。
        客户端要展示给用户时，建议只显示 name，别把整段纪律铺在界面上。
        """
        return {
            "n_tools": len(TOOLS),
            "tools": [
                {
                    "name": t["function"]["name"],
                    "description": t["function"]["description"],
                    "parameters": t["function"]["parameters"],
                }
                for t in TOOLS
            ],
        }

    # ---------------- POST /chat ----------------
    @api.post("/chat", response_model=ChatResponse)
    def post_chat(req: ChatRequest) -> ChatResponse:
        sess = sessions.get_or_create(req.session_id)

        if not sess.lock.acquire(timeout=1.0):
            raise HTTPException(
                status_code=409,
                detail={
                    "kind": "busy",
                    "message": "这个会话还有一次提问正在处理中，等它答完再发下一句。",
                    "hint": "想同时问两件事，就开一个新的 session_id（不传 session_id 即可）。",
                },
            )

        t0 = time.time()
        n_before = len(sess.messages)
        steps: list[ToolStep] = []
        started: dict[str, float] = {}
        events: list[dict] = []

        def on_event(ev: dict) -> None:
            """把 Agent 的每一步记下来，最后放进响应的 steps 字段。

            为什么不用流式把这些实时推出去？
              见文件开头第 ② 条：一次提问动不动 20 秒以上，
              流式对这个场景是负担不是帮助。客户端拿到 steps 一次性渲染就够了。
            """
            events.append(ev)
            if ev.get("type") == "tool_call":
                started[ev["tool"]] = time.time()
            elif ev.get("type") == "tool_result":
                began = started.pop(ev["tool"], t0)
                args = ""
                for e in reversed(events):
                    if e.get("type") == "tool_call" and e.get("tool") == ev["tool"]:
                        args = str(e.get("args") or "")[:200]
                        break
                steps.append(ToolStep(
                    round=len(steps) + 1,
                    tool=ev["tool"],
                    ok=ev.get("ok"),
                    seconds=round(time.time() - began, 2),
                    args_preview=args,
                ))

        try:
            result = run_agent(
                req.message, sess.messages,
                max_steps=req.max_steps, on_event=on_event,
            )
        except LLMError as exc:
            # ★ 第 ⑤ 条：失败的那一轮必须回滚，否则半条消息会毒化下一轮。
            del sess.messages[n_before:]
            sess.last_error = exc.message.splitlines()[0]
            raise HTTPException(status_code=502, detail=exc.as_dict())
        except Exception as exc:  # noqa: BLE001 —— 兜住一切，别让进程挂了
            del sess.messages[n_before:]
            sess.last_error = f"{type(exc).__name__}: {exc}"
            raise HTTPException(
                status_code=500,
                detail={"kind": "internal", "message": f"{type(exc).__name__}: {exc}",
                        "hint": "看服务端控制台的完整堆栈。"},
            )
        finally:
            sess.lock.release()

        sess.n_turns += 1
        sess.last_used = time.time()
        sess.last_error = None

        cfg = get_settings()
        return ChatResponse(
            session_id=sess.id,
            answer=result["answer"],
            stop=result["stop"],
            steps=steps,
            n_tools=len(steps),
            elapsed_s=round(time.time() - t0, 2),
            provider=cfg["label"],
            model=cfg["model"],
            note=(
                "本轮轮数用尽被中断，问题请拆小一点。"
                if result["stop"] == "max_steps" else ""
            ),
        )

    # ---------------- 会话管理 ----------------
    @api.get("/sessions")
    def list_sessions() -> dict:
        rows = sessions.info()
        return {"n_sessions": len(rows), "sessions": rows}

    @api.delete("/sessions/{session_id}")
    def drop_session(session_id: str) -> dict:
        if not sessions.drop(session_id):
            raise HTTPException(404, detail={"kind": "not_found",
                                             "message": f"没有这个会话：{session_id}"})
        return {"ok": True, "dropped": session_id}

    return api


app = create_app()


def main() -> int:
    """直接 `python -m server.app` 启动（等价于用 uvicorn 命令行）。"""
    import uvicorn

    host = os.getenv("LIBS_API_HOST", "127.0.0.1")
    port = int(os.getenv("LIBS_API_PORT", "8000"))
    reload_ = (os.getenv("LIBS_API_RELOAD") or "").strip() not in ("", "0", "false")

    cfg: dict[str, Any] = {}
    try:
        cfg = get_settings()
    except SystemExit:
        print("[警告] .env 里的 Key 没配好，服务能起来，但一问就会返回 502。")

    print("=" * 64)
    print(" LIBS AI Agent —— HTTP 服务")
    print("=" * 64)
    if cfg:
        print(f" 服务商  {cfg['label']}（LLM_PROVIDER={PROVIDER}）")
        print(f" 模型    {cfg['model']}")
    print(f" 工具    {len(TOOL_NAMES)} 个")
    print(f" 监听    http://{host}:{port}")
    print(f" 调试页  http://127.0.0.1:{port}/        （浏览器直接打开就能聊）")
    print(f" 接口文档 http://127.0.0.1:{port}/docs    （FastAPI 自动生成）")
    if host == "0.0.0.0":
        print(" ⚠ 正在监听所有网卡 —— 同一局域网里任何人都能调用你付费的 Key。")
    print("=" * 64)

    uvicorn.run("server.app:app" if reload_ else app,
                host=host, port=port, reload=reload_)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
