r"""列出当前 Key 到底能用哪些模型 —— 换厂商 / 换模型时的第一步诊断。

运行方式：
    .\.venv\Scripts\python.exe check_models.py

为什么需要这个脚本：
    「模型名写对了没」和「账号有没有开通权限」，这两件事从文档上看不出来
    —— 文档写的是「平台支持哪些模型」，不是「你这个账号能用哪些模型」。
    而 /v1/models 这个接口会直接回答后者。

    报错 `模型不存在 / Not found the model` 时，先跑这个脚本，
    再把结果里真实存在的模型名填进 .env 的 LLM_MODEL= 即可。
"""

from __future__ import annotations

import sys

from agent.llm_client import PROVIDER, build_client, get_settings


def main() -> int:
    cfg = get_settings()

    print("=" * 60)
    print(f" 服务商   {cfg['label']}  (LLM_PROVIDER={PROVIDER})")
    print(f" 服务地址 {cfg['base_url']}")
    print("=" * 60)

    client = build_client(cfg)

    try:
        page = client.models.list()
    except Exception as exc:  # noqa: BLE001
        # 有些服务商不开放 /models 接口，或者 Key 权限不足。
        print(f"[查询失败] {type(exc).__name__}")
        print(str(exc)[:500])
        print()
        print("排查建议：")
        print("  1. 确认 .env 里选的服务商（LLM_PROVIDER）和填的 Key 是同一家的")
        print("  2. 确认 Key 是从对应控制台新建的、且没有被删除")
        print("  3. 去控制台「模型列表 / 开通管理」页看账号能用哪些模型，手动填进 LLM_MODEL")
        return 1

    ids = sorted({m.id for m in page.data})

    if not ids:
        print("[提示] 该账号当前没有任何可用模型，请先去控制台开通。")
        return 1

    print(f"[可用模型] 共 {len(ids)} 个：")
    for model_id in ids:
        print(f"    - {model_id}")

    print()
    print("用法：把上面任意一个名字填进 .env 的 LLM_MODEL= 即可，例如")
    print(f"    LLM_MODEL={ids[0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
