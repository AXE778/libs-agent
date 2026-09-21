# -*- coding: utf-8 -*-
r"""
把若干条光谱画成一张 HTML 图（纯 Python 画内联 SVG，不需要 matplotlib）。

为什么单独有这个脚本？
  本机 pip 装 matplotlib 很慢；而且报告/演示时「一张能直接双击打开的网页」
  比 PNG 更方便（可以缩放到看到窄峰）。所以留一条不依赖第三方绘图库的出路。

用法：
    python plot_spectra_html.py                       # 画 data/ 里的默认三条
    python plot_spectra_html.py a.csv b.csv           # 画指定的文件
    python plot_spectra_html.py --out out\my.html a.csv

注意：画图属于「展示」，不属于「计算」，所以放在项目根目录，
不放进 tools/（tools/ 的规矩是不 print、不出图，只吐结构化结果）。
"""

from __future__ import annotations

import argparse
import html
import io
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from tools.spectrum_loader import load_spectrum, summarize  # noqa: E402

DEFAULT_FILES = [
    ("data/DATA-0908170219-X1-Y3-1.csv", "316 不锈钢 · 测点 X1-Y3"),
    ("data/DATA-0908165817-X0-Y0-1.csv", "304 不锈钢 · 测点 X0-Y0"),
    ("data/fast-260707-1621-206.csv", "单点重复谱 · 700 V / 300 ns"),
]

COLORS = ["#185FA5", "#0F6E56", "#993C1D", "#534AB7", "#854F0B"]

W, H = 1000, 250
ML, MR, MT, MB = 78, 18, 34, 40
PW, PH = W - ML - MR, H - MT - MB


def panel_svg(sp, color: str) -> str:
    ymin, ymax = 0.0, float(sp.intensity.max()) * 1.08
    x0, x1 = float(sp.wavelength[0]), float(sp.wavelength[-1])
    pts = " ".join(
        f"{ML + (float(x) - x0) / (x1 - x0) * PW:.1f},"
        f"{MT + (1 - (float(y) - ymin) / (ymax - ymin)) * PH:.1f}"
        for x, y in zip(sp.wavelength, sp.intensity)
    )

    yticks = []
    for frac in (0.0, 0.5, 1.0):
        val = ymin + (ymax - ymin) * frac
        py = MT + (1 - frac) * PH
        yticks.append(
            f'<line x1="{ML}" y1="{py:.1f}" x2="{ML + PW}" y2="{py:.1f}" stroke="#e6e4dd" stroke-width="1"/>'
            f'<text x="{ML - 8}" y="{py + 4:.1f}" text-anchor="end" font-size="11" fill="#5F5E5A">{val:,.0f}</text>'
        )
    xticks = []
    t = int(x0 // 100 + 1) * 100
    while t <= x1:
        px = ML + (t - x0) / (x1 - x0) * PW
        xticks.append(
            f'<line x1="{px:.1f}" y1="{MT + PH}" x2="{px:.1f}" y2="{MT + PH + 4}" stroke="#888780" stroke-width="1"/>'
            f'<text x="{px:.1f}" y="{MT + PH + 17}" text-anchor="middle" font-size="11" fill="#5F5E5A">{t}</text>'
        )
        t += 100

    s = summarize(sp)
    note = ""
    if float(sp.intensity.max()) >= 65535:
        py = MT + (1 - (65535 - ymin) / (ymax - ymin)) * PH
        note = (
            f'<line x1="{ML}" y1="{py:.1f}" x2="{ML + PW}" y2="{py:.1f}" stroke="#E24B4A" '
            f'stroke-width="1" stroke-dasharray="5 4"/>'
            f'<text x="{ML + PW - 6}" y="{py - 5:.1f}" text-anchor="end" font-size="11" fill="#A32D2D">'
            f'65535 = 16 位 ADC 天花板（打饱和）</text>'
        )
    px = ML + (s["peak_wavelength_nm"] - x0) / (x1 - x0) * PW
    return f"""
  <div class="panel">
    <h2><span class="dot" style="background:{color}"></span>{html.escape(sp.meta.get("name", "?"))}</h2>
    <svg viewBox="0 0 {W} {H}" width="100%" role="img">
      <rect x="{ML}" y="{MT}" width="{PW}" height="{PH}" fill="#FAFAF7" stroke="#D3D1C7" stroke-width="0.5"/>
      {''.join(yticks)}
      {''.join(xticks)}
      <line x1="{px:.1f}" y1="{MT}" x2="{px:.1f}" y2="{MT + PH}" stroke="#B4B2A9" stroke-width="1" stroke-dasharray="3 3"/>
      {note}
      <polyline points="{pts}" fill="none" stroke="{color}" stroke-width="0.8" stroke-linejoin="round"/>
      <text x="{ML - 8}" y="{MT - 12}" font-size="11" fill="#5F5E5A" text-anchor="end">强度</text>
      <text x="{ML + PW}" y="{MT + PH + 33}" font-size="11" fill="#5F5E5A" text-anchor="end">波长 (nm)</text>
    </svg>
    <div class="stat">
      点数 <b>{s['n_points']}</b> ·
      波长 <b>{s['wavelength_min_nm']:.3f}–{s['wavelength_max_nm']:.3f}</b> nm ·
      平均步长 <b>{s['step_mean_nm']:.4f}</b> nm ·
      强度 <b>{s['intensity_min']:.0f}–{s['intensity_max']:.0f}</b> ·
      最强峰 <b>{s['peak_wavelength_nm']:.2f}</b> nm
    </div>
  </div>"""


STYLE = """<style>
  body { margin:0; padding:24px 28px 40px; background:#FBFAF7; color:#2C2C2A;
         font-family:"Microsoft YaHei","Segoe UI",system-ui,sans-serif; }
  h1 { font-size:19px; font-weight:500; margin:0 0 4px; }
  .sub { font-size:13px; color:#5F5E5A; margin-bottom:22px; line-height:1.7; }
  .panel { background:#fff; border:0.5px solid #D3D1C7; border-radius:12px;
           padding:14px 18px 10px; margin-bottom:16px; }
  .panel h2 { font-size:14px; font-weight:500; margin:0 0 6px; display:flex; align-items:center; gap:8px; }
  .dot { width:9px; height:9px; border-radius:50%; display:inline-block; }
  .stat { font-size:12px; color:#5F5E5A; border-top:0.5px dashed #D3D1C7; padding-top:8px; margin-top:2px; }
  .stat b { color:#2C2C2A; font-weight:500; }
  code { font-family:Consolas,monospace; font-size:12px; background:#F1EFE8; padding:1px 5px; border-radius:4px; }
</style>"""


def build(paths: list[str], out: str) -> None:
    panels = []
    for i, rel in enumerate(paths):
        p = rel if os.path.isabs(rel) else os.path.join(HERE, rel)
        if not os.path.exists(p):
            print(f"[跳过] 不存在：{p}")
            continue
        color = COLORS[i % len(COLORS)]
        panels.append(panel_svg(load_spectrum(p), color))
    if not panels:
        raise SystemExit("一条都没画成，检查文件路径")

    doc = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head><meta charset="utf-8"><title>真实光谱一览</title>
{STYLE}
</head>
<body>
<h1>真实光谱一览</h1>
<div class="sub">横轴 波长 (nm)，纵轴 强度 (counts)。灰虚线标出最强峰位置。</div>
{''.join(panels)}
</body>
</html>
"""
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with io.open(out, "w", encoding="utf-8") as f:
        f.write(doc)
    print(f"已生成：{out}（用浏览器打开）")


def main() -> int:
    ap = argparse.ArgumentParser(description="把光谱画成 HTML（不需要 matplotlib）")
    ap.add_argument("files", nargs="*", help="光谱文件，默认画 data/ 里的三条")
    ap.add_argument("--out", default=os.path.join(HERE, "outputs", "spectra.html"))
    args = ap.parse_args()
    paths = args.files or [f for f, _ in DEFAULT_FILES if os.path.exists(os.path.join(HERE, f))]
    build(paths, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
