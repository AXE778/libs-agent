r"""通用大模型客户端（豆包 / Kimi / 商汤 都支持）—— Step 2 正式版本。

================================ 这个文件解决什么问题 ================================

你的项目以后大概率会换模型（豆包涨价换 Kimi、Kimi 限流太紧再换商汤……）。
所以一开始就不要把代码写死在某一家厂商身上。

好消息：火山方舟（豆包）、Moonshot（Kimi）、商汤日日新（SenseNova）都提供
「OpenAI 兼容」的 Chat Completions 接口——请求路径、请求体、返回结构和 OpenAI
一模一样，区别只有三处：服务地址(base_url)、模型名(model)、密钥(api_key)。

所以做法是：用 **官方 openai SDK**，把 base_url 指向不同厂商。
换厂商 = 改 `.env` 里的一个单词，**代码一行都不用动**。

实测依据（2026-09-21，故意用假密钥探测，trust_env=False 直连）：
    https://ark.cn-beijing.volces.com/api/v3  -> 401
    https://api.moonshot.cn/v1                -> 401
    https://token.sensenova.cn/v1             -> 401
三家都是走到「鉴权」这一步才失败，说明协议完全通用，只差一把真钥匙。

★★ 本机必读：系统代理会掐断这些接口（而且报错完全看不出是代理的锅）★★
    本机 Windows 系统代理是开着的（注册表 ProxyEnable=1，ProxyServer=127.0.0.1:7890）。
    而 openai SDK 默认用 httpx，httpx 又默认 trust_env=True —— 会**静默**把系统代理
    套到每一个请求上，代码里一个字都看不出来。

    2026-09-21 实测，走代理时：
        api.moonshot.cn      -> ConnectError: [SSL: UNEXPECTED_EOF_WHILE_READING]
        token.sensenova.cn   -> ConnectError: [SSL: UNEXPECTED_EOF_WHILE_READING]
    把同一个域名改成直连（trust_env=False）：
        api.moonshot.cn      -> HTTP 401（正常，因为用的是假 Key）
        token.sensenova.cn   -> HTTP 401（同上）

    注意「端口在听」不等于「代理能用」：本机 127.0.0.1:7890 确实有进程在监听，
    但 TLS 握手一进代理就被打断。这种故障看起来像「网络不通」，实际不是。

    所以 build_client() 默认**显式绕开系统代理**（trust_env=False）。
    想强制走代理，在 .env 里填 LLM_PROXY=http://127.0.0.1:7890。
    另外提醒：Clash 的「绕过名单（ProxyOverride）」对 Python 无效 ——
    Python 只读 ProxyServer，不读 ProxyOverride，在名单里加 *.moonshot.cn 也没用。

运行方式（在 libs_agent 目录下）：
    .\.venv\Scripts\python.exe -m agent.llm_client
"""

from __future__ import annotations

import os
import sys
import time

from dotenv import load_dotenv

# load_dotenv() 把 libs_agent/.env 里的键值对放进当前进程的环境变量。
# 这样 API Key 只以文本形式存在于 .env 里，源码中永远看不到明文。
load_dotenv()

# ---------------------------------------------------------------------------
# 1) 各家厂商的「预设」—— 加一家新厂商，只要往这里加一条
# ---------------------------------------------------------------------------
# 每个字段的意思：
#   label / short   给人看的名字
#   base_url        OpenAI 兼容接口的根地址（传给 SDK 的 base_url）
#   model           默认模型名
#   api_key_env     去 .env 里读哪个变量当密钥
#   console         去哪儿申请 / 管理 Key
#   hint            这个平台最容易踩的坑
#   direct          是否**绕开系统代理直连**。国内三家都填 True，
#                   原因见文件头「本机必读」——本机系统代理会把 TLS 掐断。
PROVIDERS: dict[str, dict] = {
    "ark": {
        "label": "豆包（火山方舟 Ark）",
        "short": "豆包",
        "base_url": "https://ark.cn-beijing.volces.com/api/v3",
        "model": "doubao-seed-2-1-pro-260628",
        "api_key_env": "ARK_API_KEY",
        "console": "https://console.volcengine.com/ark",
        "hint": "模型要先在控制台「开通管理」里开通，否则会报模型不存在。",
        "direct": True,
    },
    "kimi": {
        "label": "Kimi（Moonshot AI）",
        "short": "Kimi",
        "base_url": "https://api.moonshot.cn/v1",
        # 2026-09-21 实测：国内账号默认只开放 kimi-k2.6 / kimi-k2.7-code，
        # kimi-k3（旗舰）需要账号有对应权限，否则会报「模型不存在」。
        # 所以默认用 kimi-k2.6 —— 256K 上下文，够用且更便宜。
        "model": "kimi-k2.6",
        "api_key_env": "KIMI_API_KEY",
        "console": "https://platform.moonshot.cn/console/api-keys",
        "hint": "账号能用哪些模型，跑 check_models.py 一看便知；想用旗舰可填 LLM_MODEL=kimi-k3。",
        "direct": True,
    },
    "sensenova": {
        "label": "商汤日日新（SenseNova）",
        "short": "商汤",
        "base_url": "https://token.sensenova.cn/v1",
        # 2026-09-21 查证：这个平台的对话模型栏其实是个「聚合网关」，
        # 除自家的 sensenova-6.8-flash-lite，还挂着 deepseek-v4-pro /
        # deepseek-v4-flash / glm-5.2 / kimi-k3。
        # 默认选 sensenova-6.8-flash-lite：轻量、限流最宽松 —— 本项目一轮问答
        # 要连调好几次工具，限流宽松比单次能力强更重要（对比 Kimi 的每分钟 3 次）。
        "model": "sensenova-6.8-flash-lite",
        "api_key_env": "SENSENOVA_API_KEY",
        "console": "https://platform.sensenova.cn/console/keys",
        "hint": "sensenova-u1-fast 是文生图专用，不能当对话模型填；"
                "免费档限流较紧，连续调用容易 429，其中 flash-lite / deepseek-v4-flash 最宽。",
        "direct": True,
    },
}

# 选哪家：读 .env 里的 LLM_PROVIDER，没写就默认用豆包。
PROVIDER = (os.getenv("LLM_PROVIDER") or "ark").strip().lower()


# ---------------------------------------------------------------------------
# 2) 导入 SDK（用 try/except 给「没装依赖」一句人话，而不是一屏红色堆栈）
# ---------------------------------------------------------------------------
try:
    from openai import (
        APIConnectionError,
        APIStatusError,
        AuthenticationError,
        NotFoundError,
        OpenAI,
        RateLimitError,
    )
except ImportError:
    print("[依赖缺失] 没有找到 openai SDK。")
    print("请在虚拟环境里执行：pip install openai")
    sys.exit(1)


class LLMError(Exception):
    """大模型调用失败，消息是给人看的中文。

    ★ 为什么要有这个类（而不是继续 print + sys.exit）：
      llm_client 原来的做法是「打印一段人话 + sys.exit(1)」—— 对命令行脚本没问题，
      但放进 Web 服务就是灾难：**一次请求失败会把整个服务进程杀掉**，
      而且错误信息打在服务器的控制台上，HTTP 调用方什么都看不到。

      所以加了 `on_error="raise"` 模式：同样的人话原封不动放进异常里抛出去，
      由调用方（server/app.py）决定怎么转成 HTTP 响应。
      默认仍是 "exit"，命令行脚本和既有的 demo 行为一点不变。
    """

    def __init__(self, kind: str, message: str, *, detail: str = "") -> None:
        super().__init__(message)
        self.kind = kind          # 机器可读的种类：auth / model / rate_limit / network / api
        self.message = message    # 给人看的中文说明（可能多行）
        self.detail = detail      # 原始异常文本，排错用

    def as_dict(self) -> dict:
        return {"kind": self.kind, "message": self.message, "detail": self.detail}


def get_settings() -> dict:
    """把「选中的厂商预设」和「.env 里的覆盖值」合成最终配置。

    优先级：.env 里的 LLM_BASE_URL / LLM_MODEL  >  上面 PROVIDERS 里的预设。
    平时不用管，只在想临时换模型版本时才用得上。
    """
    preset = PROVIDERS.get(PROVIDER)
    if preset is None:
        print(f"[配置错误] .env 里的 LLM_PROVIDER={PROVIDER!r} 不是已知的服务商。")
        print("可选值：" + " / ".join(PROVIDERS))
        sys.exit(1)

    api_key = (os.getenv(preset["api_key_env"]) or "").strip()
    if not api_key or api_key.startswith("your_"):
        print(f"[配置错误] 没有读到有效的 {preset['api_key_env']}。")
        print(f"当前选中的服务商是 {preset['label']}（LLM_PROVIDER={PROVIDER}）。")
        print(f"请打开 libs_agent/.env，把 {preset['api_key_env']}= 后面换成你自己的 Key。")
        print(f"获取地址：{preset['console']}")
        sys.exit(1)

    return {
        "label": preset["label"],
        "short": preset["short"],
        "base_url": (os.getenv("LLM_BASE_URL") or preset["base_url"]).strip(),
        "model": (os.getenv("LLM_MODEL") or preset["model"]).strip(),
        "api_key": api_key,
        "console": preset["console"],
        "hint": preset["hint"],
        # direct=True 表示「绕开系统代理直连」，build_client() 会用到
        "direct": bool(preset.get("direct", True)),
    }


def build_client(settings: dict):
    """创建客户端。

    openai SDK 的构造方式和它调 OpenAI 时完全一样，只是 base_url 指向别家。
    这就是「OpenAI 兼容」的红利：SDK 不用换，只换地址。

    ★ 但这里多做了一件事：**显式接管「要不要走代理」**。
      原因见文件头「本机必读」：SDK 默认的 httpx 客户端 trust_env=True，会自动套上
      Windows 系统代理，而本机那个代理会把国内大模型接口的 TLS 掐断
      （SSL: UNEXPECTED_EOF_WHILE_READING）—— 报错里完全看不出跟代理有关，极难排查。
      所以默认直连；只有你在 .env 里显式写了 LLM_PROXY 才走代理。
    """
    kwargs: dict = {
        "api_key": settings["api_key"],
        "base_url": settings["base_url"],
    }
    http_client = _build_http_client(settings)
    if http_client is not None:
        kwargs["http_client"] = http_client
    return OpenAI(**kwargs)


# ---------------------------------------------------------------------------
# 3) 代理策略：默认直连，只有显式配置才走代理
# ---------------------------------------------------------------------------
def _build_http_client(settings: dict):
    """按 .env 的配置决定「走不走代理」，返回 httpx.Client（或 None 表示交回 SDK）。

    三种情况：
      1. .env 写了 LLM_PROXY=http://...  -> 用这个代理（跨厂商通用，临时救急）
      2. 预设 direct=True（国内三家都是） -> 绕开系统代理直连
      3. 其它（以后接 OpenAI / Anthropic 这类必须挂梯子的）
         -> 返回 None，交回 SDK 默认行为（跟随系统代理）
    """
    try:
        import httpx
    except ImportError:
        # httpx 是 openai SDK 的硬依赖，正常装不上不了；真走到这儿就退回默认行为。
        print("[提示] 没有 httpx，无法控制代理，将使用 SDK 默认行为。")
        return None

    # 给一个「连接快失败、长回答不被打断」的超时：
    # 连接 15 秒还没建立就当网络不通；但读完一整个长回答允许 300 秒。
    timeout = httpx.Timeout(300.0, connect=15.0)

    proxy = (os.getenv("LLM_PROXY") or "").strip()
    if proxy:
        return httpx.Client(proxy=proxy, timeout=timeout)

    if settings.get("direct"):
        # ★ 关键的一行：trust_env=False = 既不看环境变量，也不看系统注册表代理
        return httpx.Client(trust_env=False, timeout=timeout)

    return None


def describe_transport(settings: dict) -> str:
    """一句话说清「这次请求是怎么出去的」—— 排查代理问题时先看这行。"""
    proxy = (os.getenv("LLM_PROXY") or "").strip()
    if proxy:
        return f"走显式代理 {proxy}"
    if settings.get("direct"):
        return "直连（已绕开系统代理 trust_env=False）"
    return "SDK 默认行为（跟随系统代理）"


def chat(messages: list[dict], tools: list[dict] | None = None,
         *, on_error: str = "exit", **extra):
    """统一的对话入口 —— Agent 循环就调这个函数。

    Args:
        messages: 对话历史列表。元素形如
                  {"role": "user", "content": "..."}
                  {"role": "assistant", "content": "..."}
                  {"role": "tool", "tool_call_id": "...", "content": "..."}
        tools:    工具定义（JSON Schema 列表）。
        on_error: 出错怎么办。`"exit"`（默认）= 打印人话后 sys.exit(1)，
                  命令行脚本用这个；`"raise"` = 抛 LLMError，**服务端必须用这个**
                  （否则一次请求失败会把 uvicorn 进程带走）。
        extra:    其他透传参数，例如 temperature。

    Returns:
        模型返回的 message 对象，**不是纯文本**。
        因为 Agent 需要看 message.tool_calls 判断「它要不要调工具」，
        只取 content 是不够的。

    ★ 关于限流（本机实测 2026-09-21）：
      这个 Kimi 账号是 **每分钟最多 3 次请求**（organization max RPM: 3）。
      而带工具的一次提问至少 3 次请求（列文件 → 读谱 → 汇总回答），
      所以**必然会撞 429**。这里做了自动等待重试，别把它当成程序卡住。
      要根治只能去控制台提额，或换一个 RPM 更高的账号/模型
      （商汤的 sensenova-6.8-flash-lite 限流比 Kimi 宽得多，见 PROVIDERS）。

    ★ 关于代理：出网方式由 build_client() 决定，见文件头「本机必读」。
      默认绕开 Windows 系统代理直连；想强制走代理就设 .env 的 LLM_PROXY。
    """
    settings = get_settings()
    client = build_client(settings)

    payload: dict = {"model": settings["model"], "messages": messages, **extra}
    if tools:
        payload["tools"] = tools

    completion = _create_with_retry(client, payload, settings, on_error=on_error)
    return completion.choices[0].message


# 限流重试参数：最多重试几次、每次至少等多少秒
_MAX_RETRIES = 4
_MIN_WAIT_S = 21.0


def _retry_after_seconds(exc) -> float:
    """从响应头里读 retry-after；读不到就用 0。"""
    resp = getattr(exc, "response", None)
    if resp is None:
        return 0.0
    raw = None
    try:
        raw = resp.headers.get("retry-after")
    except Exception:
        return 0.0
    try:
        return float(raw)
    except (TypeError, ValueError):
        return 0.0


def _create_with_retry(client, payload: dict, settings: dict, *, on_error: str = "exit"):
    """真正发请求的地方，带限流自动重试。

    每一种失败的「人话」只写一份，由 _fail 决定是打印还是抛异常。
    写两份的话迟早会改了一处忘了另一处。
    """

    def _fail(kind: str, message: str, detail: str = "") -> None:
        if on_error == "raise":
            raise LLMError(kind, message, detail=detail)
        print(message)
        if detail:
            print(f"原始错误：{detail}")
        sys.exit(1)

    for attempt in range(_MAX_RETRIES + 1):
        try:
            return client.chat.completions.create(**payload)

        except AuthenticationError:
            _fail(
                "auth",
                "[鉴权失败] Key 无效或已被删除。\n"
                f"当前服务商：{settings['label']}，"
                f"请到 {settings['console']} 重新复制 Key。",
            )

        except NotFoundError:
            _fail(
                "model",
                f"[模型不存在] 服务商不认这个模型：{settings['model']}\n"
                "原因通常是：模型名写错了，或者你的账号没有开通这个模型。\n"
                "下一步：运行  python check_models.py  看你的 Key 到底能用哪些模型，\n"
                "        再把真实存在的名字填进 .env 的 LLM_MODEL= 。\n"
                f"提示：{settings['hint']}",
            )

        except RateLimitError as exc:
            if attempt >= _MAX_RETRIES:
                _fail(
                    "rate_limit",
                    f"[限流] 已重试 {_MAX_RETRIES} 次仍被限流。\n"
                    "这个账号每分钟请求数很少，建议去控制台提额，或改用别的模型。",
                    detail=str(exc),
                )
            wait = max(_retry_after_seconds(exc), _MIN_WAIT_S)
            # 这行是**日志**，不是错误 —— 服务端也保留，否则运维会以为服务卡死了
            print(f"[限流] 触发每分钟请求上限，等 {wait:.0f} 秒后自动重试"
                  f"（第 {attempt + 1}/{_MAX_RETRIES} 次）…", flush=True)
            time.sleep(wait)
            continue

        except APIConnectionError as exc:
            _fail("network", _network_hint(settings, exc), detail=str(exc))

        except APIStatusError as exc:
            _fail("api", f"[接口报错] HTTP {exc.status_code}", detail=str(exc))

        except Exception as exc:  # noqa: BLE001 —— 兜底，交给 _fail 决定怎么处理
            _fail("unknown", f"[未知错误] {type(exc).__name__}", detail=str(exc))


def _network_hint(settings: dict, exc) -> str:
    """连不上时给一段能直接照做的中文，而不是只丢一个英文异常名。

    ★ 为什么专门为 SSL 写一段：
      本机最容易遇到的「连不上」根本不是网络问题，而是系统代理把 TLS 掐了
      （SSL: UNEXPECTED_EOF_WHILE_READING）。这个报错长得像网络故障，
      让人去 ping / 换 DNS，白折腾半天。所以这里直接把结论摆出来。
    """
    text = str(exc).upper()
    lines = [f"[网络不通] 连不上 {settings['base_url']}"]
    if "SSL" in text or "EOF" in text or "CERTIFICATE" in text:
        lines += [
            "症状是 TLS 握手被打断（UNEXPECTED_EOF_WHILE_READING / SSL 类错误）。",
            "在本机，这几乎总是**系统代理**在中间掐连接，不是网络本身不通。",
            f"当前出网方式：{describe_transport(settings)}",
            "排查：",
            "  1) 确认 .env 里没有写 LLM_PROXY= —— 写了就会强制走代理；",
            "     想直连就把它注释掉（客户端默认已经 trust_env=False 绕开系统代理）。",
            "  2) 确实需要走代理，就填 LLM_PROXY=http://127.0.0.1:7890，",
            "     并确认那个代理进程真的能转发（端口在监听 ≠ 能用）。",
            "  3) 最快的自检：临时把 Windows「设置 → 网络 → 代理」关掉再跑一次。",
        ]
    else:
        lines += [
            "排查：确认本机能上外网；公司/校园网络可能拦了这个域名。",
            f"当前出网方式：{describe_transport(settings)}",
        ]
    return "\n".join(lines)


def to_message_dict(message) -> dict:
    """把模型返回的 assistant message 转成「能原样放回 messages」的 dict。

    为什么不能只挑 content 和 tool_calls 两个字段？
      Kimi 的思考模型（kimi-k3 / kimi-k2.7-code）有「保留式思考」机制：
      官网明确要求，多轮对话和工具调用时必须把 reasoning_content（思考过程）
      原样回传，否则推理链断裂、工具调用质量下降。
      所以这里用 model_dump() 全量转一遍，把厂商额外塞进来的字段也带上。

    Step 6 组装 Agent 循环时会用到它。
    """
    data = message.model_dump(exclude_none=True)
    data["role"] = "assistant"
    return data


def ask_once(question: str) -> str:
    """单轮问答：问一句、答一句，返回纯文本。Step 2 的验收就靠它。"""
    message = chat([{"role": "user", "content": question}])
    return message.content or ""


if __name__ == "__main__":
    cfg = get_settings()
    print(f"[服务商] {cfg['label']}  (LLM_PROVIDER={PROVIDER})")
    print(f"[模型]   {cfg['model']}")
    print(f"[服务]   {cfg['base_url']}")
    print(f"[出网]   {describe_transport(cfg)}")
    print("[提问]   你好，你是谁？")
    print("-" * 60)
    print("[回答]   " + ask_once("你好，你是谁？"))
    print()
    print("=" * 60)
    print("说明：这里是「连通性自检」——问一句、答一句，问完就结束。")
    print("      想和它连续对话（能记住上文），请运行：")
    print()
    print("          python -m agent.chat")
    print("=" * 60)
