"""Naturalize CV-extracted strokes with the LLM (the SketchAgent agent).

Reads a strokes.xml produced by img2sketch_cv.py, feeds those strokes to the
agent as its previous answer, and asks it to refine them into a more natural,
hand-drawn sketch (keeping shape, fixing stroke order, smoothing t-values).
"""
import argparse
import ast
import os
import re

import numpy as np

import cairosvg
from dotenv import load_dotenv
from PIL import Image
import anthropic

import utils
from prompts import system_prompt


NATURALIZE_PROMPT = """You are given a set of strokes that mix contour outline strokes and Canny edge strokes. Many strokes are duplicated (two strokes tracing nearly the same line), and some short strokes are parts of the same continuous line.

Process them:
1. Remove duplicate strokes: where two strokes trace nearly the same line, keep only one.
2. Merge short strokes that are clearly one continuous line (the end of one continues at the start of the next) into a single stroke, concatenating their points in order.
3. Reorder the final strokes so they draw bottom-to-top, then left-to-right (y=1 is the bottom of the canvas and y increases upward; x=1 is the left edge).

Format rules (strict):
- Write every coordinate in single quotes, like <points>'x25y1', 'x25y9'</points>.
- Do NOT invent or move any coordinate: only remove/merge/reorder whole strokes, concatenating their existing points.
- Re-number the strokes s1, s2, s3, ... in the new order.
- Each <t_values> must be the same length as its <points>.

Return the result in the same <strokes> format inside <answer> tags.
"""


def _quote_coords(s):
    # LLM sometimes writes xNyM without quotes; make it a valid Python literal.
    return re.sub(r"(?<!['\w])(x\d+y\d+)(?!['\w])", r"'\1'", s)


def render_strokes(xml_text, out_dir, name, res, cell_size, stroke_width):
    strokes_list_str, t_values_str = utils.parse_xml_string(xml_text, res)
    if strokes_list_str is None:
        return None
    strokes_list_str = _quote_coords(strokes_list_str)
    strokes_list = ast.literal_eval(strokes_list_str)
    # LLM's merged strokes can have degenerate (repeated) t_values; use uniform arc-length
    t_values = [list(np.linspace(0.0, 1.0, len(s))) for s in strokes_list]
    cells_to_pixels = utils.cells_to_pixels(res, cell_size, header_size=cell_size)
    control_points = utils.get_control_points(strokes_list, t_values, cells_to_pixels)
    canvas_size = (res + 1) * cell_size
    svg = utils.format_svg(control_points, dim=(canvas_size, canvas_size), stroke_width=stroke_width)

    svg_path = os.path.join(out_dir, f"{name}.svg")
    with open(svg_path, "w") as f:
        f.write(svg)
    cairosvg.svg2png(url=svg_path, write_to=os.path.join(out_dir, f"{name}.png"), background_color="white")

    grid_img, _ = utils.create_grid_image(res=res, cell_size=cell_size, header_size=cell_size)
    cairosvg.svg2png(url=svg_path, write_to=os.path.join(out_dir, f"{name}_canvas.png"))
    fg = Image.open(os.path.join(out_dir, f"{name}_canvas.png"))
    grid_img.paste(Image.open(os.path.join(out_dir, f"{name}_canvas.png")), (0, 0), fg)
    grid_img.save(os.path.join(out_dir, f"{name}_canvas.png"))
    return svg


def main():
    p = argparse.ArgumentParser(description="Naturalize CV strokes with the LLM")
    p.add_argument("--strokes_xml", type=str, required=True, help="path to strokes.xml from img2sketch_cv.py")
    p.add_argument("--path2save", type=str, default="results/cmp/naturalized")
    p.add_argument("--name", type=str, default=None, help="output folder name (default: <mode>_<image>)")
    p.add_argument("--model", type=str, default="deepseek-v4-flash")
    p.add_argument("--max_tokens", type=int, default=16000)
    p.add_argument("--res", type=int, default=50)
    p.add_argument("--cell_size", type=int, default=12)
    p.add_argument("--stroke_width", type=float, default=7.0)
    args = p.parse_args()

    strokes_xml = open(args.strokes_xml).read()

    full = os.path.abspath(args.strokes_xml)
    leaf = os.path.basename(os.path.dirname(full))
    gp = os.path.basename(os.path.dirname(os.path.dirname(full)))
    name = args.name or (f"{gp}_{leaf}" if gp and gp not in ("results", "cmp") else leaf)

    out_dir = os.path.join(args.path2save, name)
    os.makedirs(out_dir, exist_ok=True)

    load_dotenv()
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    # CV strokes = the agent's previous answer; user now asks it to refine.
    msg_history = [{"role": "assistant", "content": strokes_xml}]

    print("Calling LLM to naturalize...")
    response = client.messages.create(
        model=args.model,
        max_tokens=args.max_tokens,
        system=system_prompt.format(res=args.res),
        messages=msg_history + [{"role": "user", "content": [{"type": "text", "text": NATURALIZE_PROMPT}]}],
        stop_sequences=["</answer>"],
    )
    content = utils.extract_text_from_response(response)

    with open(os.path.join(out_dir, "llm_output.txt"), "w") as f:
        f.write(content)

    svg = render_strokes(content, out_dir, name, args.res, args.cell_size, args.stroke_width)
    if svg is None:
        print(f"ERROR: LLM 输出里没有找到 <strokes> 标记，原始输出已存到 {out_dir}/llm_output.txt")
        return
    print(f"已保存到 {out_dir}/  ({name}.svg / {name}.png / {name}_canvas.png / llm_output.txt)")


if __name__ == "__main__":
    main()
