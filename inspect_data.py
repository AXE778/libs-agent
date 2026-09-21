r"""数据文件体检 —— 接手任何一份「真实数据」之前，先跑这个。

用法（在项目根目录）：
    .\.venv\Scripts\python.exe inspect_data.py "D:\某个\光谱.csv"
    .\.venv\Scripts\python.exe inspect_data.py "D:\a.csv" "D:\b.txt"

它会告诉你：
    1. 文件是不是纯文本（还是二进制 / 被加密软件加密过）
    2. 用什么编码能读懂（UTF-8 / GBK / UTF-16 …）
    3. 用什么分隔符（逗号 / 制表符 / 空格 / 分号）
    4. 有多少行、多少列、表头长什么样
    5. 每列的数值范围 —— 用来判断「哪个列是波长、哪个列是强度」

为什么不能跳过这一步：
    「真实数据」的格式千奇百怪：仪器自带软件会导出一堆奇怪的表头、
    中文编码可能是 GBK、有的文件干脆是加密的。
    先花 10 秒体检，比写完代码再报错省 1 小时。
"""

from __future__ import annotations

import csv
import io
import os
import sys

# 各编码的「试读顺序」：带 BOM 的 UTF-8 放最前，然后 UTF-8、GBK（中文 Windows 常见）、
# 最后用 latin-1 兜底（它永远不报错，用来判断「是不是根本就不是文本」）。
CANDIDATE_ENCODINGS = ["utf-8-sig", "utf-8", "gbk", "utf-16", "latin-1"]

# 已知的「加密 / 专有格式」魔数（文件开头几个字节）
KNOWN_MAGIC = {
    b"%TSD-Header-###%": "企业加密软件(DLP透明加密)标记 —— Python 直接读会拿到密文",
    b"\x89HDF": "HDF5 / NetCDF4 二进制",
    b"CDF": "NetCDF3 二进制",
    b"PK\x03\x04": "ZIP 压缩包（xlsx 也是这个）",
    b"\x7fELF": "Linux 可执行文件",
    b"MZ": "Windows 可执行文件",
}

SNIFF_CHARS = [",", "\t", ";", " ", "|"]


def classify_bytes(head: bytes) -> tuple[str, str]:
    """判断文件开头是文本还是二进制，返回 (类型, 说明)。"""
    for magic, desc in KNOWN_MAGIC.items():
        if head.startswith(magic):
            return "binary", desc

    # 没有已知魔数时，靠「可打印字符占比」判断：
    # 真正的文本文件里，可打印字符通常 >95%；二进制文件会低很多。
    try:
        text_like = head.decode("utf-8")
        printable = sum(1 for c in text_like if c.isprintable() or c in "\r\n\t")
        ratio = printable / max(len(text_like), 1)
        if ratio > 0.9:
            return "text", f"看起来是文本（可打印字符 {ratio:.0%}）"
        return "binary", f"疑似二进制（可打印字符仅 {ratio:.0%}）"
    except UnicodeDecodeError:
        printable = sum(1 for b in head if 32 <= b < 127 or b in (9, 10, 13))
        ratio = printable / max(len(head), 1)
        if ratio > 0.9:
            return "text", f"看起来是文本（ASCII 占比 {ratio:.0%}）"
        return "binary", f"疑似二进制（ASCII 占比仅 {ratio:.0%}）"


def guess_delimiter(lines: list[str]) -> str:
    """猜分隔符：取前几行里出现次数最稳定且最多的那个字符。"""
    sample = [ln for ln in lines[:20] if ln.strip()]
    if not sample:
        return ","

    best, best_score = ",", -1.0
    for ch in SNIFF_CHARS:
        counts = [ln.count(ch) for ln in sample]
        if not counts or max(counts) == 0:
            continue
        # 好分隔符的特征：每行出现的次数一致（方差小），且次数不太少
        avg = sum(counts) / len(counts)
        spread = max(counts) - min(counts)
        score = avg - spread * 2
        if score > best_score:
            best, best_score = ch, score
    return best


def numeric_range(values: list[str]) -> str:
    """对一列的前若干个值尝试转 float，给出范围；转不了就说明它是文本列。"""
    nums = []
    for v in values[:500]:
        v = v.strip()
        if not v:
            continue
        try:
            nums.append(float(v))
        except ValueError:
            pass
    if not nums:
        return "非数值（文本）"
    return f"min={min(nums):.4g}  max={max(nums):.4g}  (采样 {len(nums)} 个)"


def inspect(path: str) -> None:
    print("=" * 70)
    print(f"文件：{path}")
    if not os.path.exists(path):
        print("  [错误] 文件不存在")
        return

    size = os.path.getsize(path)
    print(f"大小：{size / 1024:.1f} KB")

    with open(path, "rb") as f:
        head = f.read(65536)

    kind, desc = classify_bytes(head)
    print(f"类型：{kind}  —— {desc}")

    # 开头 60 字节的原始样子，方便肉眼确认
    print(f"开头 60 字节：{head[:60]!r}")

    if kind == "binary":
        print("  [结论] 这不是纯文本，load_spectrum() 的常规做法读不了。")
        print("         如果是加密标记，需要先把文件解密 / 导出成明文格式。")
        return

    # ---------- 文本：继续深挖 ----------
    encoding = "latin-1"
    text = ""
    for enc in CANDIDATE_ENCODINGS:
        try:
            text = head.decode(enc)
            encoding = enc
            break
        except (UnicodeDecodeError, LookupError):
            continue
    print(f"编码：{encoding}")

    lines = text.splitlines()
    print(f"总行数（按 64KB 采样）：{len(lines)}")
    print(f"第 1 行：{lines[0][:200]!r}" if lines else "  (空文件)")

    delim = guess_delimiter(lines)
    delim_show = "\\t" if delim == "\t" else delim
    print(f"分隔符：{delim_show!r}")

    rows = [ln for ln in lines if ln.strip()]
    print(f"非空行数：{len(rows)}")

    # 交给 csv 模块做正式解析（它能正确处理引号）
    try:
        parsed = list(csv.reader(io.StringIO("\n".join(rows[:200])), delimiter=delim))
    except Exception as exc:  # noqa: BLE001
        print(f"  [警告] CSV 解析失败：{exc}")
        return

    if not parsed:
        return

    widths = {}
    for r in parsed:
        widths[len(r)] = widths.get(len(r), 0) + 1
    main_width = max(widths, key=widths.get)
    print(f"列数：{main_width}  （出现过的列数：{dict(sorted(widths.items()))}）")

    print("前 3 行内容：")
    for r in parsed[:3]:
        print("   " + " | ".join(c[:28] for c in r[:8]) + ("  …" if len(r) > 8 else ""))

    # 逐列看数值范围
    print("各列性质：")
    for col in range(min(main_width, 6)):
        values = [r[col] for r in parsed if len(r) > col]
        sample_vals = [v for v in values[1:] if v.strip()] or values
        print(f"   col{col}: 表头={values[0][:20]!r}  {numeric_range(sample_vals)}")

    print()
    print("  [结论] 如果正好是 2 列、且都是数值 → 直接可以喂给 load_spectrum()。")


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2
    for path in argv[1:]:
        try:
            inspect(path)
        except Exception as exc:  # noqa: BLE001
            print(f"  [异常] {type(exc).__name__}: {exc}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
