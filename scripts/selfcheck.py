r"""
scripts/selfcheck.py —— 离线自检：不联网、不花钱，验证「数据能读、工具能跑」。

clone 下来先跑这个。它**完全不调用大模型**（所以不需要 API Key），
只把 9 个工具挨个跑一遍，告诉你哪一步断了、断在哪。

为什么走 agent.tool_registry.execute() 而不是直接调 tools/ 里的函数？
    因为 execute() 就是模型实际走的那条路 —— 它返回什么、出错怎么包装，
    这里看到的和模型看到的**完全一样**。自检要回答的问题是
    「大模型能不能用这套工具」，不是「函数本身对不对」。

用法：
    .\.venv\Scripts\python.exe scripts\selfcheck.py

退出码：0 = 关键路径全通；1 = 有硬失败（数据读不了 / 工具抛异常）。
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.tool_registry import TOOLS, execute   # noqa: E402
from config import paths as P                    # noqa: E402

OK, SKIP, FAIL = "OK", "跳过", "失败"


def _brief(payload, n=170) -> str:
    s = json.dumps(payload, ensure_ascii=False, default=str)
    return s if len(s) <= n else s[:n] + " …"


def _run(name: str, args: dict, note: str = "") -> tuple[str, dict]:
    """跑一个工具并打印一行结果。返回 (状态, 原始返回)。"""
    out = execute(name, args)
    ok = bool(out.get("ok"))
    err = str(out.get("error") or "")

    # 「谱线库没自备」不算失败 —— 那是可选数据，缺了不影响其它功能
    status = OK if ok else (
        SKIP if "找不到谱线库" in err else FAIL)

    mark = {OK: "[ OK ]", SKIP: "[跳过]", FAIL: "[失败]"}[status]
    detail = _brief(out.get("result")) if ok else _brief(err, 200)
    print(f"  {mark} {name}")
    if note:
        print(f"         {note}")
    print(f"         {detail}")
    return status, out


def main() -> int:
    print("=" * 72)
    print(" LIBS AI Agent —— 离线自检（不联网、不需要 API Key）")
    print("=" * 72)

    # ---------------------------------------------------------------- 环境
    print("\n[环境]")
    print(f"  Python      : {sys.version.split()[0]}  ({sys.executable})")
    print(f"  项目根目录  : {P.PROJECT_ROOT}")
    print(f"  数据根目录  : {P.DATA_ROOT}")
    print(f"  数据来源    : {P.DATA_ROOT_SOURCE}")
    print(f"  已注册工具  : {len(TOOLS)} 个")

    results: list[str] = []

    # ------------------------------------------------------- 1) 有没有数据
    print("\n[1] 数据总览 —— list_available_spectra")
    status, overview = _run("list_available_spectra", {})
    results.append(status)
    if status == FAIL:
        print("\n  ✗ 读不到任何数据。两种可能：")
        print(f"    ① 数据根目录不对 —— 现在指到 {P.DATA_ROOT}")
        print("       想换目录：设环境变量 LIBS_DATA_ROOT，或改 config/paths.py")
        print("    ② 还没有示例数据 —— 跑一下生成脚本：")
        print("       .\\.venv\\Scripts\\python.exe tools\\make_sample_data.py")
        return 1

    res = overview.get("result") or {}
    groups = res.get("groups") or []
    if not groups:
        print("\n  ✗ 目录扫到了，但没有可用的二维扫描组。")
        return 1

    # 挑一个组继续往下测：优先挑文件名解析得出来的那个
    target = groups[0]
    for g in groups:
        if g.get("grid"):
            target = g
            break
    gdir = target.get("dir") or ""
    material = target.get("material") or ""
    print(f"\n  → 后续测试以 {os.path.basename(gdir)} 这组（材料 {material}）为样本")

    # --------------------------------------------------------- 2) 找文件
    print("\n[2] 按条件找文件 —— list_available_spectra(query=...)")
    status, found = _run("list_available_spectra", {"query": material, "limit": 5})
    results.append(status)

    files = ((found.get("result") or {}).get("files")) or []
    if not files:
        print("\n  ✗ 没找到具体文件，后面的单文件测试没法做。")
        return 1
    one = files[0]["path"]

    # --------------------------------------------------------- 3) 二维网格
    print("\n[3] 还原二维网格 —— describe_material_grid")
    status, _ = _run("describe_material_grid", {"material": material})
    results.append(status)

    # --------------------------------------------------------- 4) 读一条谱
    print("\n[4] 读一条谱 —— load_spectrum_summary")
    status, _ = _run("load_spectrum_summary", {"path": one})
    results.append(status)

    # ------------------------------------------------------ 5) 单条质量
    print("\n[5] 单条质量检查 —— check_spectrum_quality")
    status, _ = _run("check_spectrum_quality", {"path": one})
    results.append(status)

    # ------------------------------------------------------ 6) 整组质量
    print("\n[6] 整组质量检查 —— check_group_quality")
    status, _ = _run("check_group_quality", {"material": material, "max_files": 12})
    results.append(status)

    # ---------------------------------------------------------- 7) 画图
    print("\n[7] 画图 —— plot_spectra")
    status, _ = _run("plot_spectra", {"paths": [f["path"] for f in files[:3]]})
    results.append(status)

    # ------------------------------------------------------ 8) 元素匹配
    print("\n[8] 元素谱线匹配 —— analyze_element_lines")
    status, _ = _run("analyze_element_lines", {"path": one},
                     note="（需要自备谱线库；没有会「跳过」而不是「失败」）")
    results.append(status)

    # -------------------------------------------------------- 9) 预处理
    print("\n[9] 预处理 —— preprocess_spectrum")
    status, _ = _run("preprocess_spectrum", {"paths": [one]},
                     note="（会真的写出新文件到 outputs/preprocessed/，原始谱不动）")
    results.append(status)

    # -------------------------------------------------------- 10) 报告
    print("\n[10] 汇总报告 —— generate_report")
    status, _ = _run("generate_report", {"material": material, "max_files": 12})
    results.append(status)

    # ------------------------------------------------------------- 汇总
    n_ok = results.count(OK)
    n_skip = results.count(SKIP)
    n_fail = results.count(FAIL)
    print("\n" + "=" * 72)
    print(f" 合计 {len(results)} 项：OK {n_ok} · 跳过 {n_skip} · 失败 {n_fail}")
    print("=" * 72)

    if n_fail:
        print(" 有硬失败 —— 先解决上面标 [失败] 的项。")
        return 1

    print(" 关键路径全通。接下来可以：")
    print("   1) 让它去聊（要 API Key，见 .env.example）：")
    print('      .\\.venv\\Scripts\\python.exe main.py "有哪些数据"')
    print("   2) 起 HTTP 服务：")
    print("      .\\.venv\\Scripts\\python.exe -m server.app")
    if n_skip:
        print("   ⚠ 有跳过的项：补上谱线库（tools/element_analysis.py 里写了怎么自备）")
        print("     后，元素匹配与报告里的元素部分才可用。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
