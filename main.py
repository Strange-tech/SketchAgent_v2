"""SketchAgent image-to-sketch one-shot entry.

Pipeline (shared by the CLI and gallery.py via run_pipeline):
  SAM3 semantic segmentation (tree / trunk; foliage = tree minus trunk)
  -> make_masked (masked RGB images)
  -> img2sketch_cv.extract_strokes (contour + Canny strokes)
  -> registry: detect_stroke_duplication (dedup)
  -> registry: order_strokes (long before short, then bottom-to-top, then left-to-right)
  -> agent_ensure_trunk (LLM trunk/branch strokes, filtered to the trunk mask)
  -> merge CV strokes (blue contour + orange Canny) + agent trunk (brown)

Usage:
  python main.py [--image input/tree.jpg] [--threshold 3]
"""
import argparse
import ast
import json
import os
import re

import numpy as np
import cairosvg
from PIL import Image, ImageOps
import anthropic
from dotenv import load_dotenv

from color_segment import segment
from make_masked import make_masked
import sam3_segment
import img2sketch_cv as cv
from combine_sketches import extract_groups
from make_animation import make_animation
from registry import Registry
from tools.stroke_duplication import DetectStrokeDuplicationTool
from tools.stroke_ordering import OrderStrokesTool
from tools.compile_xml import CompileXmlTool
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


# System prompt guiding the agent to enforce the tree structure.
TREE_SYSTEM_PROMPT = """You are a tree sketch refinement agent. A tree must have exactly ONE trunk, with several branches splitting off from it. The trunk is the single main vertical stem connecting the base of the tree to the crown. If a sketch is missing a trunk, has a fragmented/disconnected trunk, or has multiple disconnected trunks, fix it so there is one continuous trunk with branches."""


def _strokes_to_xml(strokes, res=50, cell_size=12):
    return cv.strokes_to_xml(strokes, res, cell_size)


def agent_ensure_trunk(strokes_xml, model="deepseek-chat", max_attempts=3):
    """Ask the LLM to output ONLY the trunk + branch strokes (not the foliage).

    Each attempt is an independent single-turn request, so a thinking-mode model
    never requires its ``thinking`` block to be passed back (the cause of the
    "content[].thinking ... must be passed back" 400 error). If the output XML
    fails CompileXmlTool validation, the specific error plus the previous output
    is embedded in the next fresh request, up to `max_attempts`.
    """
    load_dotenv()
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    compiler = CompileXmlTool()
    base_prompt = (
        "Here is a tree sketch. Identify the trunk and its branches. "
        "Output ONLY the strokes that form the trunk and branches (a single continuous trunk "
        "with several branches splitting off). Do NOT include any foliage (crown) strokes. "
        "Return them in the same <strokes> format, with every coordinate in single quotes like 'x25y1'."
    )
    feedback = ""
    for attempt in range(1, max_attempts + 1):
        user_text = base_prompt if not feedback else feedback + "\n\n" + base_prompt
        messages = [
            {"role": "assistant", "content": strokes_xml},
            {"role": "user", "content": [{"type": "text", "text": user_text}]},
        ]
        try:
            response = client.messages.create(
                model=model,
                max_tokens=16000,
                system=TREE_SYSTEM_PROMPT,
                messages=messages,
                stop_sequences=["</answer>"],
            )
            text = utils.extract_text_from_response(response).strip()
        except Exception as e:
            print(f"agent_ensure_trunk attempt {attempt} failed: {e}")
            continue
        if not text:
            continue

        result = compiler.validate(text)
        if result.get("valid"):
            return text

        # carry the specific error + (bounded) previous output into the next fresh request
        error = result.get("error", "unknown error")
        prev = text if len(text) <= 1500 else text[-1500:]
        feedback = (
            f"Your previous answer was invalid: {error}. "
            "Fix the format and output ONLY the valid <strokes> block: a <strokes> root containing "
            "one or more <sN> elements, each with a <points> child (comma-separated single-quoted "
            "coordinates like 'x3y25', 'x4y24') and a <t_values> child. No markdown fences, no extra text.\n\n"
            f"Your previous (invalid) output was:\n{prev}"
        )
    return ""


def _parse_strokes_pixels(xml_text, res=50, cell_size=12):
    """Parse an LLM <strokes> output into pixel-coordinate point lists.

    Returns [] for empty or malformed input (never raises).
    """
    if not xml_text:
        return []
    try:
        strokes_list_str, _ = utils.parse_xml_string(xml_text, res)
        if strokes_list_str is None:
            return []
        strokes_list_str = re.sub(r"(?<!['\w])(x\d+y\d+)(?!['\w])", r"'\1'", strokes_list_str)
        strokes_list = ast.literal_eval(strokes_list_str)
    except Exception:
        return []
    out = []
    for stroke in strokes_list:
        pts = []
        for cell in stroke:
            m = re.match(r"x(\d+)y(\d+)", cell)
            if not m:
                continue
            gx, gy = int(m.group(1)), int(m.group(2))
            # grid y=1 -> pixel bottom, grid y=res -> pixel top (matches cells_to_pixels)
            pts.append([(gx + 0.5) * cell_size, (res + 0.5 - gy) * cell_size])
        if len(pts) >= 2:
            out.append(np.array(pts))
    return out


def _mask_to_canvas(mask, canvas_size=612):
    """Resize a binary mask to the square letterboxed canvas (matches load_gray_square).

    The CV strokes live in the 612x612 letterboxed space; the color-segmentation
    mask lives in the original image space, so we apply the same contain+center-pad
    transform to align them before filtering.
    """
    h, w = mask.shape
    scale = min(canvas_size / h, canvas_size / w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    img = Image.fromarray((mask * 255).astype(np.uint8), mode="L").resize((nw, nh), Image.NEAREST)
    canvas = Image.new("L", (canvas_size, canvas_size), 0)
    canvas.paste(img, ((canvas_size - nw) // 2, (canvas_size - nh) // 2))
    return np.asarray(canvas) > 127


def _filter_trunk_strokes(strokes, trunk_mask, min_overlap=0.20):
    """Keep only strokes whose points mostly fall inside the trunk mask.

    The agent is asked to output ONLY trunk/branch strokes, but it often also
    copies foliage strokes verbatim. The color-segmentation trunk mask separates
    them: trunk strokes overlap it, foliage strokes do not.
    """
    if trunk_mask is None:
        return strokes
    h, w = trunk_mask.shape
    kept = []
    for s in strokes:
        xs = np.clip(s[:, 0].astype(int), 0, w - 1)
        ys = np.clip(s[:, 1].astype(int), 0, h - 1)
        if trunk_mask[ys, xs].mean() >= min_overlap:
            kept.append(s)
    return kept


def _render_strokes_groups(strokes, color, res=50, cell_size=12, stroke_width=7.0):
    """Render pixel-coordinate strokes into colored <g> groups (for merging)."""
    if not strokes:
        return []
    canvas = (res + 1) * cell_size
    all_cp = cv.build_control_points(strokes)
    svg = utils.format_svg(all_cp, dim=(canvas, canvas), stroke_width=stroke_width)
    return [re.sub(r'stroke="[^"]*"', f'stroke="{color}"', g) for g in extract_groups(svg)]


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

    # 6) order (bottom-to-top, left-to-right; trunk before foliage, outline before detail)
    order_input = [
        {"points": [[float(x), float(y)] for x, y in m["points"]], "part": m["part"], "layer": m["layer"]}
        for m in meta
    ]
    order_result = json.loads(registry.execute("order_strokes", json.dumps({"strokes": order_input})))
    ordered = order_result["ordered"]
    ordered_strokes = [np.array(o["points"]) for o in ordered]
    ordered_colors = [meta[o["index"]]["color"] for o in ordered]

    # 7) agent trunk (prompt-only): keep CV strokes + add the agent's trunk/branch strokes
    strokes_xml = _strokes_to_xml(ordered_strokes)
    trunk_xml = agent_ensure_trunk(strokes_xml)
    trunk_strokes = _parse_strokes_pixels(trunk_xml)
    trunk_strokes = _filter_trunk_strokes(trunk_strokes, _mask_to_canvas(trunk))  # drop foliage the agent wrongly copied
    if not trunk_strokes:
        print(f"  (no trunk strokes for {image_path})")
    trunk_groups = _render_strokes_groups(trunk_strokes, "#8b5a2b")  # brown trunk

    # 8) merge CV strokes (blue contour + orange Canny) + agent trunk (brown), then render
    canvas = (50 + 1) * 12
    all_cp = cv.build_control_points(ordered_strokes)
    cv_svg = utils.format_svg(all_cp, dim=(canvas, canvas), stroke_width=7.0)
    cv_groups = [re.sub(r'stroke="[^"]*"', f'stroke="{ordered_colors[i]}"', g) for i, g in enumerate(extract_groups(cv_svg))]
    all_groups = cv_groups + trunk_groups
    body = [re.sub(r'id="s\d+"', f'id="s{i}"', g) for i, g in enumerate(all_groups, 1)]
    svg = '<svg width="612" height="612" xmlns="http://www.w3.org/2000/svg">\n' + "\n".join(body) + "\n</svg>"
    return svg, len(all_groups)


def sam3_tree_masks(image, processor):
    """Return (tree, foliage, trunk) masks from SAM3.

    Only SAM3's trunk prompt is used for the part split; foliage is derived as
    everything in the tree that is not trunk, because the SAM3 'foliage' text
    prompt is unreliable across foliage colors/shapes.
    """
    masks = sam3_segment.segment(image, processor, ["tree", "trunk"])
    tree = masks["tree"]
    trunk = masks["trunk"]
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
