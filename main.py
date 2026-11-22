"""SketchAgent image-to-sketch one-shot entry.

Pipeline (shared by the CLI and gallery.py via run_pipeline):
  SAM3 semantic segmentation (tree / trunk; foliage = tree minus trunk)
  -> make_masked (masked RGB images)
  -> img2sketch_cv.extract_strokes (contour + Canny strokes)
  -> registry: detect_stroke_duplication (dedup)
  -> registry: order_strokes (trunk before foliage; contour chained continuously; detail long/bottom/left)
  -> render (blue contour + orange Canny)

Usage:
  python main.py [--image input/tree.jpg] [--threshold 3]
"""
import argparse
import json
import os
import re

import numpy as np
import cairosvg
from PIL import Image
from scipy.ndimage import binary_fill_holes

from color_segment import segment, refine_trunk_mask
from make_masked import make_masked
import sam3_segment
import img2sketch_cv as cv
from combine_sketches import extract_groups
from make_animation import make_animation
from registry import Registry
from tools.stroke_duplication import DetectStrokeDuplicationTool
from tools.stroke_ordering import OrderStrokesTool
import utils


CONTOUR_COLOR = "#2f6feb"  # blue
CANNY_COLOR = "#e0562f"    # orange


def _save_mask(mask, path):
    Image.fromarray((mask * 255).astype(np.uint8), mode="L").save(path)


def run_pipeline(image_path, threshold=3.0, tree_mask=None, foliage_mask=None, trunk_mask=None, workdir="results/tmp"):
    """Run the full image -> sketch pipeline for one image.

    Returns (svg_text, stroke_count). `foliage_mask` / `trunk_mask` are SAM3 semantic
    masks (bool ndarrays); when omitted, it falls back to color segmentation. `tree_mask`
    optionally restricts both masks to the tree region. This is the single source of
    truth shared by the CLI entry point and the gallery.
    """
    os.makedirs(workdir, exist_ok=True)
    foliage_mask_path = os.path.join(workdir, "foliage_mask.png")
    trunk_mask_path = os.path.join(workdir, "trunk_mask.png")

    # 1) foliage/trunk segmentation (SAM3 masks, or color fallback)
    if foliage_mask is None or trunk_mask is None:
        foliage, trunk = segment(image_path, tree_mask=tree_mask)
    else:
        foliage = np.asarray(foliage_mask, dtype=bool)
        trunk = np.asarray(trunk_mask, dtype=bool)
        if tree_mask is not None:
            tree = np.asarray(tree_mask, dtype=bool)
            foliage = foliage & tree
            trunk = trunk & tree
    foliage = binary_fill_holes(foliage)  # fill holes so the contour is the clean outer silhouette
    _save_mask(foliage, foliage_mask_path)
    _save_mask(trunk, trunk_mask_path)

    # 2) masked RGB images
    foliage_rgb, trunk_rgb = make_masked(image_path, foliage_mask_path, trunk_mask_path, workdir)

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
        json.dumps({"strokes": point_lists, "threshold": threshold}),
    ))
    to_remove = set()
    for pair in result["duplicate_pairs"]:
        i, j = pair["i"], pair["j"]
        if i not in to_remove and j not in to_remove:
            to_remove.add(j)
    meta = [m for idx, m in enumerate(meta) if idx not in to_remove]

    # 6) order (long before short, then bottom-to-top, then left-to-right)
    order_input = [
        {"points": [[float(x), float(y)] for x, y in m["points"]], "part": m["part"], "layer": m["layer"]}
        for m in meta
    ]
    order_result = json.loads(registry.execute("order_strokes", json.dumps({"strokes": order_input})))
    ordered = order_result["ordered"]
    ordered_strokes = [np.array(o["points"]) for o in ordered]
    ordered_colors = [meta[o["index"]]["color"] for o in ordered]

    # 7) render CV strokes (blue contour + orange Canny)
    canvas = (50 + 1) * 12
    all_cp = cv.build_control_points(ordered_strokes)
    cv_svg = utils.format_svg(all_cp, dim=(canvas, canvas), stroke_width=7.0)
    cv_groups = [re.sub(r'stroke="[^"]*"', f'stroke="{ordered_colors[i]}"', g) for i, g in enumerate(extract_groups(cv_svg))]
    body = [re.sub(r'id="s\d+"', f'id="s{i}"', g) for i, g in enumerate(cv_groups, 1)]
    svg = '<svg width="612" height="612" xmlns="http://www.w3.org/2000/svg">\n' + "\n".join(body) + "\n</svg>"
    return svg, len(cv_groups)


def sam3_tree_masks(image, processor, trunk_color_tol=50.0):
    """Return (tree, foliage, trunk) masks from SAM3.

    Only SAM3's trunk prompt is used for the part split; the coarse SAM3 trunk mask
    is then refined by growing into connected pixels of similar color (so branches
    SAM3 missed are re-attached). Foliage is everything in the tree that is not trunk.
    """
    masks = sam3_segment.segment(image, processor, ["tree", "trunk"])
    tree = masks["tree"]
    trunk = refine_trunk_mask(image, masks["trunk"], tree_mask=tree, color_tol=trunk_color_tol)
    foliage = tree & ~trunk
    return tree, foliage, trunk


def main():
    p = argparse.ArgumentParser(description="SketchAgent image-to-sketch")
    p.add_argument("--image", type=str, default="input/tree.jpg", help="input image path")
    p.add_argument("--threshold", type=float, default=3.0, help="dedup distance threshold in pixels")
    args = p.parse_args()

    # 0) SAM3 segmentation: tree + trunk; foliage = tree minus trunk
    processor = sam3_segment.build_processor()
    image = Image.open(args.image).convert("RGB")
    tree_mask, foliage_mask, trunk_mask = sam3_tree_masks(image, processor)

    svg, n = run_pipeline(
        args.image, args.threshold,
        tree_mask=tree_mask, foliage_mask=foliage_mask, trunk_mask=trunk_mask,
        workdir="results/masked",
    )

    os.makedirs("results/masked", exist_ok=True)
    with open("results/masked/combined.svg", "w") as f:
        f.write(svg)
    cairosvg.svg2png(url="results/masked/combined.svg", write_to="results/masked/combined.png", background_color="white")
    make_animation("results/masked/combined.svg", "results/masked/combined_animation.html", "Tree, stroke by stroke")
    print(f"done ({n} strokes) -> results/masked/combined.svg / combined_animation.html")


if __name__ == "__main__":
    main()
