"""SketchAgent image-to-sketch one-shot entry.

Pipeline:
  color_segment (split foliage/trunk by color)
  -> make_masked (masked RGB images)
  -> img2sketch_cv.extract_strokes (contour + Canny strokes)
  -> registry: detect_stroke_duplication (dedup)
  -> registry: order_strokes (bottom-to-top, left-to-right)
  -> render (contour blue, Canny orange) + animation

Usage:
  python main.py [--threshold 3]
"""
import argparse
import json
import os
import re

import numpy as np
import cairosvg
from PIL import Image

from color_segment import segment
from make_masked import make_masked
import img2sketch_cv as cv
from combine_sketches import extract_groups
from make_animation import make_animation
from registry import Registry
from tools.stroke_duplication import DetectStrokeDuplicationTool
from tools.stroke_ordering import OrderStrokesTool
import utils


# System prompt describing the agent's cleanup task (English).
SYSTEM_PROMPT = """You are a sketch cleanup agent. Your task is:
1. Read the current sketch (contour outlines + Canny edges, which contain many duplicate strokes).
2. Call detect_stroke_duplication to find which stroke pairs are near-duplicates, and remove one of each pair.
3. Call order_strokes to reorder the remaining strokes bottom-to-top, then left-to-right (trunk before foliage, outline before detail).
4. Output the cleaned, ordered sketch.
"""

CONTOUR_COLOR = "#2f6feb"  # blue
CANNY_COLOR = "#e0562f"    # orange


def _save_mask(mask, path):
    Image.fromarray((mask * 255).astype(np.uint8), mode="L").save(path)


def _render_colored(strokes, colors, out_path, res=50, cell_size=12, stroke_width=7.0):
    """Render strokes with per-stroke colors (and save a PNG)."""
    canvas_size = (res + 1) * cell_size
    all_cp = cv.build_control_points(strokes)
    svg = utils.format_svg(all_cp, dim=(canvas_size, canvas_size), stroke_width=stroke_width)
    groups = [re.sub(r'stroke="[^"]*"', f'stroke="{colors[i]}"', g) for i, g in enumerate(extract_groups(svg))]
    svg = '<svg width="612" height="612" xmlns="http://www.w3.org/2000/svg">\n' + "\n".join(groups) + "\n</svg>"
    with open(out_path, "w") as f:
        f.write(svg)
    cairosvg.svg2png(url=out_path, write_to=out_path.replace(".svg", ".png"), background_color="white")
    return svg, len(strokes)


def main():
    p = argparse.ArgumentParser(description="SketchAgent image-to-sketch")
    p.add_argument("--threshold", type=float, default=3.0, help="dedup distance threshold in pixels")
    args = p.parse_args()

    # 1) color segmentation
    foliage, trunk = segment("input/tree.jpg")
    os.makedirs("results/seg", exist_ok=True)
    _save_mask(foliage, "results/seg/foliage_mask.png")
    _save_mask(trunk, "results/seg/trunk_mask.png")

    # 2) masked RGB images
    foliage_rgb, trunk_rgb = make_masked(
        "input/tree.jpg", "results/seg/foliage_mask.png", "results/seg/trunk_mask.png", "results/masked"
    )

    # 3) extract strokes, tagged with (part, layer, color)
    trunk_c = cv.extract_strokes(trunk_rgb, mode="contour", contour_parts=24, corner_angle=40, max_seg_len=40)
    trunk_e = cv.extract_strokes(trunk_rgb, mode="edges", corner_angle=40, max_strokes=40, max_seg_len=40)
    fol_c = cv.extract_strokes(foliage_rgb, mode="contour", contour_parts=8, corner_angle=180, max_seg_len=100000)
    fol_e = cv.extract_strokes(foliage_rgb, mode="edges", corner_angle=180, max_strokes=30, max_seg_len=100000)

    meta = []
    for s in trunk_c:
        meta.append({"points": s, "part": 0, "layer": 0, "color": CONTOUR_COLOR})
    for s in trunk_e:
        meta.append({"points": s, "part": 0, "layer": 1, "color": CANNY_COLOR})
    for s in fol_c:
        meta.append({"points": s, "part": 1, "layer": 0, "color": CONTOUR_COLOR})
    for s in fol_e:
        meta.append({"points": s, "part": 1, "layer": 1, "color": CANNY_COLOR})

    # 4) registry with both tools
    registry = Registry(DetectStrokeDuplicationTool(), OrderStrokesTool())

    # 5) dedup
    point_lists = [[[float(x), float(y)] for x, y in m["points"]] for m in meta]
    result = json.loads(registry.execute(
        "detect_stroke_duplication",
        json.dumps({"strokes": point_lists, "threshold": args.threshold}),
    ))
    to_remove = set()
    for pair in result["duplicate_pairs"]:
        i, j = pair["i"], pair["j"]
        if i not in to_remove and j not in to_remove:
            to_remove.add(j)
    meta = [m for idx, m in enumerate(meta) if idx not in to_remove]
    print(f"dedup: {result['duplicate_count']} duplicate pairs removed -> {len(meta)} strokes")

    # 6) order (bottom-to-top, left-to-right; trunk before foliage, outline before detail)
    order_input = [
        {"points": [[float(x), float(y)] for x, y in m["points"]], "part": m["part"], "layer": m["layer"]}
        for m in meta
    ]
    order_result = json.loads(registry.execute("order_strokes", json.dumps({"strokes": order_input})))
    ordered = order_result["ordered"]

    ordered_strokes = [np.array(o["points"]) for o in ordered]
    ordered_colors = [meta[o["index"]]["color"] for o in ordered]

    # 7) render + animation
    _render_colored(ordered_strokes, ordered_colors, "results/masked/combined.svg")
    make_animation("results/masked/combined.svg", "results/masked/combined_animation.html", "Tree, stroke by stroke")
    print(f"done: {len(ordered_strokes)} strokes -> results/masked/combined.svg / combined_animation.html")


if __name__ == "__main__":
    main()
