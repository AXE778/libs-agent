# -*- coding: utf-8 -*-
r"""
tools.spectrum_loader —— 把「一条光谱」从磁盘读进内存。

设计约束（见 tools/__init__.py）：
  这里只做纯计算 / 纯读取，不 print、不调用大模型、不 import agent。

支持的真实格式（都是本机 LIBS 仪器导出的）：
  1) 无表头两列：180.619,654.0            → 上位机软件导出的 DATA-*.csv
  2) 有表头两列：wavelength,intensity     → 快速采集导出的 fast-*.csv
  3) 有其它表头名：wave,intensity / 波长,强度 等，靠「表头是不是数字」自动判断
  4) 任意分隔符：逗号 / 制表符 / 分号 / 空白
  5) 任意常见编码：utf-8-sig(BOM) / utf-8 / gbk / latin-1
"""

from __future__ import annotations

import csv
import io
import os
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = ["Spectrum", "load_spectrum", "summarize", "SpectrumLoadError"]

# 列名别名：从表头里认出「哪一列是波长、哪一列是强度」
_WAVELENGTH_ALIASES = {
    "wavelength", "wave", "wavelength_nm", "wavenm", "nm",
    "波长", "波长nm", "波长(nm)", "波长/nm", "lamda", "lambda",
}
_INTENSITY_ALIASES = {
    "intensity", "intensities", "counts", "count", "signal", "value",
    "强度", "计数", "信号", "intensity_counts", "single_point_intensity",
    "mean_intensity",
}

_NUMBER_RE = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$")


class SpectrumLoadError(Exception):
    """读光谱失败时抛出，消息是给人看的中文。"""


@dataclass
class Spectrum:
    """一条光谱。

    属性
    ----
    wavelength : np.ndarray  波长轴，单位 nm，长度 N
    intensity  : np.ndarray  强度，单位「计数 counts」，长度 N
    meta       : dict        来源信息（路径、编码、分隔符、原表头……）
    """

    wavelength: np.ndarray
    intensity: np.ndarray
    meta: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return int(self.wavelength.size)

    def __repr__(self) -> str:  # 便于在终端里快速看一眼
        if len(self) == 0:
            return "Spectrum(空)"
        return (
            f"Spectrum(N={len(self)}, "
            f"{self.wavelength[0]:.3f}~{self.wavelength[-1]:.3f} nm, "
            f"强度 {self.intensity.min():.0f}~{self.intensity.max():.0f})"
        )


# --------------------------------------------------------------------------
# 内部小工具
# --------------------------------------------------------------------------
def _read_text(path: str) -> tuple[str, str]:
    """把文件读成文本，自动试编码。返回 (文本, 用到的编码)。"""
    if not os.path.exists(path):
        raise SpectrumLoadError(f"文件不存在：{path}")
    if os.path.isdir(path):
        raise SpectrumLoadError(f"这是个文件夹不是文件：{path}")

    with open(path, "rb") as fh:
        raw = fh.read()

    if not raw:
        raise SpectrumLoadError(f"文件是空的：{path}")

    # 加密层 / 二进制文件的早期报警
    if raw.startswith(b"%TSD-Header-###%"):
        raise SpectrumLoadError(
            f"这个文件被本机的加密层包住了（文件头是 %TSD-Header-###%）：{path}\n"
            "  原因：文件是被 PowerShell 之类非白名单进程复制/写出的。\n"
            "  解决：用 Python 重新复制一份，例如\n"
            "      import shutil; shutil.copy(原路径, 目标路径)"
        )
    if raw[:8] == b"\x89HDF\r\n\x1a\n" or raw[:3] == b"CDF":
        raise SpectrumLoadError(
            f"这是 HDF5 / NetCDF 立方体，不是两列光谱文件：{path}\n"
            "  那种数据要用专门的立方体读取器（后续步骤再做）。"
        )

    for enc in ("utf-8-sig", "utf-8", "gbk", "latin-1"):
        try:
            return raw.decode(enc), enc
        except UnicodeDecodeError:
            continue
    raise SpectrumLoadError(f"试了 utf-8/gbk/latin-1 都解不开：{path}")


def _split_line(line: str, delim: str | None) -> list[str]:
    return [p.strip() for p in (line.split(delim) if delim else line.split())]


def _is_number(s: str) -> bool:
    return bool(_NUMBER_RE.match(s.strip()))


def _guess_delimiter(lines: list[str]) -> str | None:
    """结合前若干行猜分隔符；返回 ',' / '\t' / ';' / None(空白)。"""
    best, best_score = None, -1
    for d in (",", "\t", ";", None):
        score = 0
        for ln in lines[:20]:
            parts = _split_line(ln, d)
            if len(parts) >= 2 and all(_is_number(p) for p in parts[:2]):
                score += 1
        if score > best_score:
            best, best_score = d, score
    return best


def _pick_columns(header: list[str], n_data_cols: int) -> tuple[int, int]:
    """从表头认列。认不出就用第 0、1 列。"""
    low = [h.strip().lower().replace(" ", "") for h in header]
    wi = ii = None
    for idx, h in enumerate(low):
        if wi is None and h in _WAVELENGTH_ALIASES:
            wi = idx
        if ii is None and h in _INTENSITY_ALIASES:
            ii = idx
    if wi is None:
        wi = 0
    if ii is None:
        ii = 1 if wi != 1 else 0
    if max(wi, ii) >= n_data_cols:
        raise SpectrumLoadError(
            f"表头说波长在第 {wi} 列、强度在第 {ii} 列，但数据只有 {n_data_cols} 列"
        )
    return wi, ii


# --------------------------------------------------------------------------
# 主入口
# --------------------------------------------------------------------------
def load_spectrum(
    path: str,
    *,
    wavelength_col: int | None = None,
    intensity_col: int | None = None,
    delimiter: str | None = None,
    comment: str = "#",
    name: str | None = None,
) -> Spectrum:
    """读一条两列光谱。

    参数
    ----
    path            : 光谱文件路径（.csv / .txt / .dat …）
    wavelength_col  : 手动指定波长列号（从 0 开始）。默认自动判断。
    intensity_col   : 手动指定强度列号。默认自动判断。
    delimiter       : 手动指定分隔符。默认自动判断。
    comment         : 以该字符开头的行当注释跳过。
    name            : 覆盖 meta["name"]。

    返回
    ----
    Spectrum
    """
    text, encoding = _read_text(path)

    # 去掉空行 / 注释行
    lines = []
    for ln in text.splitlines():
        s = ln.strip()
        if not s or s.startswith(comment):
            continue
        lines.append(s)
    if not lines:
        raise SpectrumLoadError(f"去掉注释和空行后没有内容了：{path}")

    if delimiter is None:
        delimiter = _guess_delimiter(lines)

    # 第一行是表头还是数据？
    first = _split_line(lines[0], delimiter)
    has_header = not (len(first) >= 2 and _is_number(first[0]) and _is_number(first[1]))

    header: list[str] = []
    body = lines
    if has_header:
        header = first
        body = lines[1:]
    if not body:
        raise SpectrumLoadError(f"文件只有表头、没有数据：{path}")

    rows: list[tuple[float, float]] = []
    skipped = 0
    for ln in body:
        parts = _split_line(ln, delimiter)
        if len(parts) < 2:
            skipped += 1
            continue
        try:
            x = float(parts[0]); y = float(parts[1])
        except ValueError:
            skipped += 1
            continue
        rows.append((x, y))

    if len(rows) < 2:
        raise SpectrumLoadError(
            f"只解析出 {len(rows)} 行有效数字，太少：{path}\n"
            "  检查一下分隔符和列数（可用 inspect_data.py 先看一眼）。"
        )

    arr = np.asarray(rows, dtype=float)

    if wavelength_col is None or intensity_col is None:
        wi, ii = _pick_columns(header, arr.shape[1]) if header else (0, 1)
        if wavelength_col is not None:
            wi = wavelength_col
        if intensity_col is not None:
            ii = intensity_col
    else:
        wi, ii = wavelength_col, intensity_col
    if max(wi, ii) >= arr.shape[1]:
        raise SpectrumLoadError(f"指定了第 {max(wi, ii)} 列，但只有 {arr.shape[1]} 列")
    if wi == ii:
        raise SpectrumLoadError("波长列和强度列不能是同一列")

    wavelength = arr[:, wi]
    intensity = arr[:, ii]

    # 波长必须递增；如果反了就翻过来
    flipped = False
    if wavelength.size > 2 and wavelength[0] > wavelength[-1]:
        wavelength = wavelength[::-1].copy()
        intensity = intensity[::-1].copy()
        flipped = True

    meta = {
        "path": os.path.abspath(path),
        "name": name or os.path.splitext(os.path.basename(path))[0],
        "encoding": encoding,
        "delimiter": {",": "逗号", "\t": "制表符", ";": "分号", None: "空白"}[delimiter],
        "has_header": has_header,
        "header": header,
        "wavelength_col": wi,
        "intensity_col": ii,
        "skipped_lines": skipped,
        "flipped": flipped,
        "file_size": os.path.getsize(path),
    }
    return Spectrum(wavelength=wavelength, intensity=intensity, meta=meta)


def summarize(spec: Spectrum) -> dict[str, Any]:
    """给一条光谱算一份摘要，供 Agent / 报告使用。纯计算，不 print。"""
    wl, it = spec.wavelength, spec.intensity
    if wl.size == 0:
        raise SpectrumLoadError("光谱是空的，算不出摘要")

    d = np.diff(wl)
    peak_idx = int(np.argmax(it))
    base = float(np.percentile(it, 5))
    return {
        "name": spec.meta.get("name", "?"),
        "n_points": int(wl.size),
        "wavelength_min_nm": float(wl.min()),
        "wavelength_max_nm": float(wl.max()),
        "wavelength_span_nm": float(wl.max() - wl.min()),
        "step_mean_nm": float(d.mean()) if d.size else 0.0,
        "step_min_nm": float(d.min()) if d.size else 0.0,
        "step_max_nm": float(d.max()) if d.size else 0.0,
        "is_uniform": bool(d.size and (d.max() - d.min()) < 1e-6 * max(1.0, float(d.mean()))),
        "intensity_min": float(it.min()),
        "intensity_max": float(it.max()),
        "intensity_mean": float(it.mean()),
        "intensity_median": float(np.median(it)),
        "intensity_std": float(it.std()),
        "noise_p05": base,
        "peak_wavelength_nm": float(wl[peak_idx]),
        "peak_intensity": float(it[peak_idx]),
        "snr_rough": float((it.max() - base) / base) if base > 0 else float("inf"),
        "source": spec.meta.get("path", "?"),
        "delimiter": spec.meta.get("delimiter", "?"),
        "encoding": spec.meta.get("encoding", "?"),
        "has_header": spec.meta.get("has_header", None),
    }
