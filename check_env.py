r"""环境自检脚本 —— Step 1 完成后跑一次，用来确认"地基"是好的。

运行方式（先激活虚拟环境）：
    .\.venv\Scripts\Activate.ps1
    python check_env.py

判定标准：
  * 所有 [OK]      -> 环境没问题，可以进入 Step 2。
  * 出现 [FAIL]    -> 按提示修复，退出码为 1。
  * 只有 [TODO]    -> 环境没问题，只是还没填 API Key，属于正常状态。
"""

from __future__ import annotations

import os
import sys
import urllib.request

# 需要检查的三方库：(import 名, pip 包名, 显示名)
# pip 包名单独写出来，是为了在模块没有 __version__ 时，
# 用 importlib.metadata 从"安装记录"里反查版本号。
REQUIRED_PACKAGES = [
    ("openai", "openai", "openai SDK"),
    ("dotenv", "python-dotenv", "python-dotenv"),
    ("numpy", "numpy", "numpy"),
    ("pandas", "pandas", "pandas"),
    ("scipy", "scipy", "scipy"),
    ("fastapi", "fastapi", "fastapi"),
    ("uvicorn", "uvicorn", "uvicorn"),
    ("pydantic", "pydantic", "pydantic"),
]

# 可选依赖：装了才显示，没装不影响使用。
OPTIONAL_PACKAGES = [
    ("volcenginesdkarkruntime", "volcengine-python-sdk", "火山方舟SDK(可选)"),
]

# 服务商 -> (显示名, 该服务商要读的环境变量名)
# ★ 2026-09-21 改成「从 agent/llm_client.py 的 PROVIDERS 直接推导」。
#   原先这里手抄了一份 ark/kimi 的对照表，加商汤时就会漏改 —— 于是脚本报
#   「不是已知服务商」，而真实原因只是这份抄件过时了，很容易白查半天。
#   让预设表只有一个出处，就不会再有这种漂移。
def _load_provider_keys() -> tuple[dict, list]:
    try:
        from agent.llm_client import PROVIDERS as _P
    except Exception:  # 连 llm_client 都导不进来时，至少还能查依赖问题
        _P = {
            "ark": {"label": "豆包（火山方舟）", "api_key_env": "ARK_API_KEY"},
            "kimi": {"label": "Kimi（Moonshot）", "api_key_env": "KIMI_API_KEY"},
        }
    return {k: (v["label"], v["api_key_env"]) for k, v in _P.items()}, list(_P)


PROVIDER_KEYS, KNOWN_PROVIDERS = _load_provider_keys()


def get_version(module, pip_name: str) -> str:
    """取三方库的版本号。

    为什么要写得这么啰嗦？因为踩过坑：
      1. 有些库（比如火山方舟 SDK）根本没有 __version__ 属性；
      2. 它的 dist-info 元数据也不完整，用标准库 importlib.metadata 去查会直接抛
         TypeError（"NoneType" 拼字符串），而不是老老实实报"查不到"。
    所以这里先试属性，再直接去 site-packages 里读"包名-版本号.dist-info"的目录名，
    这条路不依赖任何元数据解析，最稳。
    """
    ver = getattr(module, "__version__", None)
    if isinstance(ver, str) and ver.strip() and ver.strip().lower() not in ("none", "unknown"):
        return ver.strip()

    try:
        import sysconfig

        site_dir = sysconfig.get_paths()["purelib"]
        prefix = pip_name.replace("-", "_")
        for name in os.listdir(site_dir):
            if name.startswith(prefix + "-") and name.endswith(".dist-info"):
                return name[len(prefix) + 1: -len(".dist-info")]
    except Exception:
        pass
    return "已安装"


def main() -> int:
    failed = False

    print("=" * 60)
    print(" LIBS AI Agent —— 环境自检")
    print("=" * 60)

    # ---------- 1. Python 版本 ----------
    v = sys.version_info
    if v >= (3, 11):
        print(f"[OK]   Python {v.major}.{v.minor}.{v.micro}")
    else:
        failed = True
        print(f"[FAIL] Python {v.major}.{v.minor}.{v.micro} —— 本项目需要 3.11 及以上")

    # ---------- 2. 是否在虚拟环境里 ----------
    # 在虚拟环境中运行时，sys.prefix 指向 venv 目录，与 sys.base_prefix 不同。
    if sys.prefix != sys.base_prefix:
        print(f"[OK]   虚拟环境   {sys.prefix}")
    else:
        print("[WARN] 当前不在虚拟环境中。请先执行 .\\.venv\\Scripts\\Activate.ps1")

    # ---------- 3. 三方依赖 ----------
    print("-" * 60)
    for mod_name, pip_name, label in REQUIRED_PACKAGES:
        try:
            mod = __import__(mod_name)
            print(f"[OK]   {label:<18} {get_version(mod, pip_name)}")
        except ImportError as exc:
            failed = True
            print(f"[FAIL] {label:<18} 未安装 ({exc})")

    for mod_name, pip_name, label in OPTIONAL_PACKAGES:
        try:
            mod = __import__(mod_name)
            print(f"[OK]   {label:<18} {get_version(mod, pip_name)}")
        except ImportError:
            pass

    # ---------- 4. .env 配置 ----------
    print("-" * 60)
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if not os.path.exists(env_path):
        failed = True
        print("[FAIL] 找不到 .env 文件。请把 .env.example 复制成 .env")
    else:
        print("[OK]   找到 .env")
        try:
            from dotenv import load_dotenv
            load_dotenv(env_path)

            provider = (os.getenv("LLM_PROVIDER") or "ark").strip().lower()
            label, key_env = PROVIDER_KEYS.get(provider, ("", ""))

            if not key_env:
                failed = True
                print(f"[FAIL] LLM_PROVIDER={provider!r} 不是已知服务商，"
                      f"可选：{' / '.join(KNOWN_PROVIDERS)}")
            else:
                print(f"[OK]   LLM_PROVIDER {provider}  ->  {label}")
                key = os.getenv(key_env, "")
                if not key or key.startswith("your_"):
                    print(f"[TODO] {key_env} 还没填。打开 .env，把等号后面换成真实 Key 才能进 Step 2。")
                else:
                    print(f"[OK]   {key_env} 已配置（{key[:6]}...，安全起见只显示前 6 位）")

                # 显示当前生效的模型与地址（预设表在 agent/llm_client.py）
                try:
                    from agent.llm_client import PROVIDERS
                    preset = PROVIDERS.get(provider)
                    if preset:
                        print(f"[OK]   默认模型   {os.getenv('LLM_MODEL') or preset['model']}")
                        print(f"[OK]   服务地址   {os.getenv('LLM_BASE_URL') or preset['base_url']}")

                        # ---------- 代理自检（本机踩过的坑）----------
                        # 这里刻意不调 get_settings()：它在 Key 没填时会 sys.exit，
                        # 而自检脚本应当把所有问题一次列完，不能中途退出。
                        proxy = (os.getenv("LLM_PROXY") or "").strip()
                        direct = bool(preset.get("direct", True))
                        if proxy:
                            print(f"[OK]   出网方式   走显式代理 {proxy}")
                        elif direct:
                            print("[OK]   出网方式   直连（trust_env=False，已绕开系统代理）")
                        else:
                            print("[WARN] 出网方式   跟随 SDK 默认（会读系统代理）。"
                                  "若报 SSL: UNEXPECTED_EOF_WHILE_READING，就是栽在这里。")

                        # 系统代理开着但本服务商直连 —— 说明「绕过」这件事正在起作用，
                        # 明确报出来，省得以后看到系统代理就以为请求走了它。
                        sys_proxy = (urllib.request.getproxies().get("https")
                                     or urllib.request.getproxies().get("http"))
                        if sys_proxy and direct and not proxy:
                            print(f"[OK]   系统代理   本机开着（{sys_proxy}），"
                                  "但本服务商不受它影响")
                except Exception:
                    pass
        except ImportError:
            pass

    print("=" * 60)
    if failed:
        print("结论：有项目未通过，请按上面的 [FAIL] 逐条修复。")
        return 1
    print("结论：环境就绪。下一步填好 .env 里的 Key，然后运行 python -m agent.llm_client")
    return 0


if __name__ == "__main__":
    sys.exit(main())
