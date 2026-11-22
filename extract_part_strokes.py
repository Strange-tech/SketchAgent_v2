"""Stage C: per-part stroke extraction from SAM3 masks (trunk + foliage).

Outputs a JSON list of strokes: {part, layer, points, t_values}
  part  : 0=trunk, 1=foliage
  layer : 0=outline, 1=detail
  points: list of (x, y) grid coords, x in [1,res] left->right, y in [1,res] BOTTOM->top
"""
import argparse
import json
from collections import Counter

import numpy as np
from PIL import Image, ImageOps
from skimage.morphology import skeletonize, binary_erosion, disk
from skimage.feature import canny
from skimage.measure import find_contours

import img2sketch_cv as cv


def load_mask_square(mask_path, canvas_size):
    img = Image.open(mask_path).convert("L")
    img = ImageOps.contain(img, (canvas_size, canvas_size))
    canvas = Image.new("L", (canvas_size, canvas_size), 0)  # black = background padding
    canvas.paste(img, ((canvas_size - img.width) // 2, (canvas_size - img.height) // 2))
    return np.asarray(canvas, dtype=float) / 255.0 > 0.5


def pixel_to_grid_xy(x, y, res, cell_size):
    gx = int(round(x / cell_size - 1.5)) + 1
    gy = int(round((res - 0.5) - y / cell_size)) + 1
    return (max(1, min(res, gx)), max(1, min(res, gy)))


def stroke_to_dict(pixel_pts, res, cell_size, part, layer):
    grid = []
    for x, y in pixel_pts:
        g = pixel_to_grid_xy(x, y, res, cell_size)
        if not grid or grid[-1] != g:
            grid.append(g)
    if len(grid) < 2:
        return None
    pts = np.array(grid, dtype=float)
    d = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(d)])
    total = cum[-1]
    if total > 0:
        t = [float(v / total) for v in cum]
    else:
        t = [float(i) / (len(grid) - 1) for i in range(len(grid))]
    return {"part": part, "layer": layer, "points": grid, "t_values": t}


def _trace(mask, min_len, simplify_tol, corner_angle, res, cell_size, part, layer):
    skel = skeletonize(mask)
    out = []
    for s in cv.trace_strokes(skel, min_len=min_len):
        s = cv.simplify(s, simplify_tol)
        for seg in cv.split_at_corners(s, corner_angle):
            if len(seg) >= 2 and cv._path_length(seg) >= min_len:
                d = stroke_to_dict(seg, res, cell_size, part, layer)
                if d:
                    out.append(d)
    return out


def outline(mask, simplify_tol, parts, res, cell_size):
    contours = find_contours(mask.astype(float), 0.5)
    if not contours:
        return []
    outer = max(contours, key=len)
    pts = np.stack([outer[:, 1], outer[:, 0]], axis=1)
    pts = cv.simplify(pts, simplify_tol)
    out = []
    for seg in cv.split_closed_into_open(pts, n=parts):
        d = stroke_to_dict(seg, res, cell_size, 1, 0)
        if d:
            out.append(d)
    return out


def foliage_detail(image_gray, foliage_mask, sigma, low, high, min_len, simplify_tol, corner_angle, res, cell_size, max_strokes=None):
    """真实树叶细节：原图 Canny 边缘 ∩ 树叶 mask 内部（腐蚀后，排除轮廓边界）。"""
    edges = canny(image_gray, sigma=sigma, low_threshold=low, high_threshold=high)
    inner = binary_erosion(foliage_mask, disk(2))
    edges = edges & inner
    skel = skeletonize(edges)
    segs = []
    for s in cv.trace_strokes(skel, min_len=min_len):
        s = cv.simplify(s, simplify_tol)
        for seg in cv.split_at_corners(s, corner_angle):
            if len(seg) >= 2 and cv._path_length(seg) >= min_len:
                segs.append(seg)
    segs.sort(key=cv._path_length, reverse=True)
    if max_strokes:
        segs = segs[:max_strokes]
    out = []
    for seg in segs:
        d = stroke_to_dict(seg, res, cell_size, 1, 1)
        if d:
            out.append(d)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trunk_mask", required=True)
    p.add_argument("--foliage_mask", required=True)
    p.add_argument("--image", required=True, help="原始图像，用于树叶 detail 的 Canny 边缘")
    p.add_argument("--output", required=True)
    p.add_argument("--res", type=int, default=50)
    p.add_argument("--cell_size", type=int, default=12)
    p.add_argument("--min_len", type=float, default=8.0)
    p.add_argument("--simplify_tol", type=float, default=2.0)
    p.add_argument("--corner_angle", type=float, default=45.0)
    p.add_argument("--sigma", type=float, default=1.5)
    p.add_argument("--low", type=float, default=0.10)
    p.add_argument("--high", type=float, default=0.25)
    p.add_argument("--contour_parts", type=int, default=10)
    p.add_argument("--max_detail_strokes", type=int, default=40, help="树叶 detail 最多保留的最长笔画数")
    args = p.parse_args()

    canvas = (args.res + 1) * args.cell_size
    trunk = load_mask_square(args.trunk_mask, canvas)
    foliage = load_mask_square(args.foliage_mask, canvas)
    image_gray = cv.load_gray_square(args.image, canvas)

    strokes = []
    strokes += _trace(trunk, args.min_len, args.simplify_tol, args.corner_angle, args.res, args.cell_size, 0, 0)
    strokes += outline(foliage, args.simplify_tol, args.contour_parts, args.res, args.cell_size)
    strokes += foliage_detail(image_gray, foliage, args.sigma, args.low, args.high, args.min_len, args.simplify_tol, args.corner_angle, args.res, args.cell_size, args.max_detail_strokes)

    with open(args.output, "w") as f:
        json.dump(strokes, f, indent=2)

    c = Counter((s["part"], s["layer"]) for s in strokes)
    print(f"提取 {len(strokes)} 条: trunk={c.get((0,0),0)}, foliage_outline={c.get((1,0),0)}, foliage_detail={c.get((1,1),0)} -> {args.output}")


if __name__ == "__main__":
    main()
