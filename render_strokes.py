"""Render ordered strokes (JSON with grid (x,y) points) to SVG + PNG."""
import argparse
import json
import os

import cairosvg

import utils


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--strokes", required=True, help="JSON list of {points, t_values}")
    p.add_argument("--outdir", required=True)
    p.add_argument("--res", type=int, default=50)
    p.add_argument("--cell_size", type=int, default=12)
    p.add_argument("--stroke_width", type=float, default=7.0)
    args = p.parse_args()

    strokes = json.load(open(args.strokes))
    os.makedirs(args.outdir, exist_ok=True)

    cells_to_pixels = utils.cells_to_pixels(args.res, args.cell_size, header_size=args.cell_size)
    strokes_all = [[f"x{x}y{y}" for x, y in s["points"]] for s in strokes]
    t_values_all = [s["t_values"] for s in strokes]
    control_points = utils.get_control_points(strokes_all, t_values_all, cells_to_pixels)

    canvas = (args.res + 1) * args.cell_size
    svg = utils.format_svg(control_points, dim=(canvas, canvas), stroke_width=args.stroke_width)

    name = os.path.splitext(os.path.basename(args.strokes))[0]
    svg_path = os.path.join(args.outdir, f"{name}.svg")
    with open(svg_path, "w") as f:
        f.write(svg)
    cairosvg.svg2png(url=svg_path, write_to=os.path.join(args.outdir, f"{name}.png"), background_color="white")
    print(f"渲染 {len(strokes)} 条笔画 -> {svg_path}")


if __name__ == "__main__":
    main()
