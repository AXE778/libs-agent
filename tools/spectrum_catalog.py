r"""
tools.spectrum_catalog —— 盘点磁盘上有哪些光谱。

设计约束（见 tools/__init__.py）：
  纯计算 / 纯读取，不 print、不调用大模型、不 import agent。
  只读文件名和目录结构，**不读文件内容**——所以就算有 1200 个文件也是毫秒级。

为什么需要它？
  Agent 要「自己找数据」。用户说「看看 304 的」，模型得先把
  `<你的数据目录>\data-304-1\DATA-....csv` 这种真实路径列出来，
  第二步才能拿路径去读谱。这一步不做，模型只能瞎猜路径。
"""

from __future__ import annotations

import os
import re
from typing import Any

__all__ = [
    "CatalogError",
    "overview",
    "list_materials",
    "find_spectra",
    "scan_grid",
    "resolve_material",
]

# 仪器导出的文件名样式：DATA-0908165817-X0-Y0-1.csv
#   tag=DATA  stamp=0908165817(日期+时间)  x/y=测点坐标  shot=第几次打点
_FILENAME_RE = re.compile(
    r"^(?P<tag>[A-Za-z]+)-(?P<stamp>\d+)-X(?P<x>\d+)-Y(?P<y>\d+)-(?P<shot>\d+)$"
)


class CatalogError(Exception):
    """盘点失败时抛，消息是给人看的中文。"""


def _check_root(data_root: str) -> str:
    if not data_root:
        raise CatalogError("没有指定数据根目录")
    if not os.path.isdir(data_root):
        raise CatalogError(f"数据根目录不存在：{data_root}")
    return os.path.abspath(data_root)


def _parse_name(stem: str) -> dict[str, Any] | None:
    """把 `DATA-0908165817-X0-Y0-1` 拆成结构化字段；不是这个样式就返回 None。"""
    m = _FILENAME_RE.match(stem)
    if not m:
        return None
    d = m.groupdict()
    return {
        "tag": d["tag"],
        "stamp": d["stamp"],
        "x": int(d["x"]),
        "y": int(d["y"]),
        "shot": int(d["shot"]),
    }


def _material_of(name: str) -> str:
    """从目录名猜材料牌号，例如 data-304-新-up5nm -> 304、data-6061-1 -> 6061。

    牌号可能是 3 位（304/316）或 4 位（5052/6061/7075），
    所以用 (?<!\\d)\\d{3,4}(?!\\d) —— 前后不能紧挨着别的数字，
    否则 `data-304-1` 里的 "1" 或长串数字会被误抓。
    """
    m = re.search(r"(?<!\d)(\d{3,4})(?!\d)", name)
    return m.group(1) if m else name


def overview(data_root: str) -> dict[str, Any]:
    """数据总览 —— 只回答「有哪些组数据、每组多大」，不列文件。

    为什么单独有这个？
      因为「不筛选就把 2000 个文件路径全列出来」既没用（模型看不完），
      又会把上下文撑爆。总览信息量高、体积小，是更好的第一步。

    返回：{n_groups, total_files, materials:[...], groups:[{dir, material, n_files,
          grid, n_points, shots_per_point}]}
    """
    mats = list_materials(data_root)
    groups = []
    for m in mats:
        if m["n_data_files"] == 0 and not m["grid"]:
            kind = "非数据目录"
        elif m["grid"]:
            kind = "二维扫描"
        else:
            kind = "无坐标信息"
        groups.append({
            "dir": m["name"],
            "material": m["material"],
            "kind": kind,
            "n_files": m["n_data_files"],
            "grid": m["grid"]["shape"] if m["grid"] else None,
            "n_points": m["grid"]["n_points"] if m["grid"] else None,
            "shots_per_point": m["shots_per_point"],
            "n_measurement_rounds": m["n_stamps"],
        })
    materials = sorted({g["material"] for g in groups if g["grid"]})

    # 按材料牌号汇总。为什么让 Python 算好而让模型自己加？
    #   因为「让模型做算术」= 编造风险。实测它会把 6061 的 6 组 325 个文件
    #   说成「5 组 250 个」。凡是能算的，都在这里算好。
    agg: dict[str, dict[str, Any]] = {}
    for g in groups:
        if not g["grid"]:
            continue
        b = agg.setdefault(g["material"], {
            "material": g["material"], "n_groups": 0,
            "n_files": 0, "n_points_total": 0, "dirs": [],
        })
        b["n_groups"] += 1
        b["n_files"] += g["n_files"]
        b["n_points_total"] += g["n_points"] or 0
        b["dirs"].append(g["dir"])

    return {
        "root": os.path.abspath(data_root),
        "n_groups": len(groups),
        "total_files": sum(g["n_files"] for g in groups),
        "materials": materials,
        "by_material": [agg[k] for k in sorted(agg)],
        "groups": groups,
    }


def list_materials(data_root: str) -> list[dict[str, Any]]:
    """列出数据根目录下的每个子目录（= 一组测量）。

    返回：[{name, path, material, n_csv, n_parseable, grid, tags, stamps}]
    """
    root = _check_root(data_root)
    out: list[dict[str, Any]] = []

    for name in sorted(os.listdir(root)):
        d = os.path.join(root, name)
        if not os.path.isdir(d):
            continue

        csvs, parsed, xs, ys, shots, stamps, tags = [], [], set(), set(), set(), set(), set()
        for fn in sorted(os.listdir(d)):
            if not os.path.isfile(os.path.join(d, fn)):
                continue
            ext = os.path.splitext(fn)[1].lower()
            if ext not in (".csv", ".txt", ".dat"):
                continue
            csvs.append(fn)
            info = _parse_name(os.path.splitext(fn)[0])
            if info:
                parsed.append(info)
                xs.add(info["x"]); ys.add(info["y"])
                shots.add(info["shot"])
                stamps.add(info["stamp"]); tags.add(info["tag"])

        grid = None
        if xs and ys:
            grid = {
                "x_values": sorted(xs),
                "y_values": sorted(ys),
                "shape": f"{len(xs)}×{len(ys)}",
                "n_points": len(xs) * len(ys),
            }

        out.append({
            "name": name,
            "path": d,
            "material": _material_of(name),
            "n_data_files": len(csvs),
            "n_parseable": len(parsed),
            "n_stamps": len(stamps),
            "shots_per_point": sorted(shots),
            "grid": grid,
            "tags": sorted(tags),
        })
    if not out:
        raise CatalogError(f"{root} 下没有任何子目录")
    return out


def resolve_material(data_root: str, query: str) -> list[dict[str, Any]]:
    """按关键字找子目录。`304` 会命中 data-304-1 / data-304-2 / data-304-新 …"""
    root = _check_root(data_root)
    q = (query or "").strip().lower()
    dirs = [n for n in sorted(os.listdir(root)) if os.path.isdir(os.path.join(root, n))]
    if not q:
        return [{"name": n, "path": os.path.join(root, n), "material": _material_of(n)} for n in dirs]

    hits = [n for n in dirs if q in n.lower()]
    if not hits:  # 退一步：只比材料牌号
        hits = [n for n in dirs if q == _material_of(n).lower()]
    if not hits:
        raise CatalogError(
            f"没有名字里含「{query}」的子目录。现有：{', '.join(dirs)}"
        )
    return [{"name": n, "path": os.path.join(root, n), "material": _material_of(n)} for n in hits]


def find_spectra(
    data_root: str,
    query: str | None = None,
    *,
    limit: int = 40,
    full_path: bool = True,
) -> dict[str, Any]:
    """找光谱文件。

    query 为空 → 列出全部（受 limit 限制）；
    query 可以是子目录名或材料牌号，如 `304` / `data-304-1` / `新`。

    返回 {n_matched, n_returned, truncated, files:[{path, name, dir, material, x, y, shot}]}
    """
    root = _check_root(data_root)
    if limit <= 0:
        limit = 40

    if query:
        dirs = [d["path"] for d in resolve_material(root, query)]
    else:
        dirs = [os.path.join(root, n) for n in sorted(os.listdir(root))
                if os.path.isdir(os.path.join(root, n))]

    files: list[dict[str, Any]] = []
    for d in dirs:
        for fn in sorted(os.listdir(d)):
            if os.path.splitext(fn)[1].lower() not in (".csv", ".txt", ".dat"):
                continue
            if not os.path.isfile(os.path.join(d, fn)):
                continue
            info = _parse_name(os.path.splitext(fn)[0]) or {}
            files.append({
                "path": os.path.join(d, fn) if full_path else fn,
                "name": fn,
                "dir": os.path.basename(d),
                "material": _material_of(os.path.basename(d)),
                "x": info.get("x"),
                "y": info.get("y"),
                "shot": info.get("shot"),
            })

    total = len(files)
    return {
        "n_matched": total,
        "n_returned": min(total, limit),
        "truncated": total > limit,
        "files": files[:limit],
    }


def scan_grid(data_root: str, material: str) -> dict[str, Any]:
    """把一组的测点还原成二维网格，看哪些位置有数据、哪些缺。

    这是「二维元素成像」的入口：先知道网格长什么样，才谈得上重建。
    """
    root = _check_root(data_root)
    (target,) = resolve_material(root, material)[:1]
    d = target["path"]

    pts: dict[tuple[int, int], list[int]] = {}
    for fn in sorted(os.listdir(d)):
        info = _parse_name(os.path.splitext(fn)[0])
        if not info:
            continue
        pts.setdefault((info["x"], info["y"]), []).append(info["shot"])

    if not pts:
        raise CatalogError(f"{d} 里没有能解析出 X/Y 坐标的文件名")

    xs = sorted({k[0] for k in pts})
    ys = sorted({k[1] for k in pts})
    missing = [(x, y) for x in xs for y in ys if (x, y) not in pts]
    shots = sorted({s for v in pts.values() for s in v})

    return {
        "dir": target["name"],
        "material": target["material"],
        "x_values": xs,
        "y_values": ys,
        "grid_shape": f"{len(xs)}×{len(ys)}",
        "n_expected_points": len(xs) * len(ys),
        "n_measured_points": len(pts),
        "missing_points": missing,
        "shots_per_point": shots,
        "counts_per_point": {f"X{x}-Y{y}": len(v) for (x, y), v in sorted(pts.items())},
    }
