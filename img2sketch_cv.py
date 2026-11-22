"""Image -> sketch via classical computer vision (no LLM).

Pipeline:
    grayscale -> resize/pad to square canvas -> Canny edges -> skeletonize
    -> trace skeleton into ordered strokes -> simplify (Douglas-Peucker)
    -> fit Bezier control points -> render SVG/PNG via utils.format_svg.

Also dumps the strokes in SketchAgent's <strokes> grid-cell XML format
(strokes.xml) so they can later be handed to the LLM for naturalization.
"""
import argparse
import os

import numpy as np
from PIL import Image, ImageOps
from skimage.feature import canny
from skimage.morphology import skeletonize, binary_dilation, disk
from skimage.measure import approximate_polygon, find_contours, label, regionprops
from skimage.filters import threshold_otsu
import cairosvg

import utils


def load_gray_square(image_path, canvas_size):
    img = Image.open(image_path).convert("L")
    img = ImageOps.contain(img, (canvas_size, canvas_size))
    canvas = Image.new("L", (canvas_size, canvas_size), 255)
    canvas.paste(img, ((canvas_size - img.width) // 2, (canvas_size - img.height) // 2))
    return np.asarray(canvas, dtype=float) / 255.0


def extract_skeleton(gray, sigma, low, high):
    edges = canny(gray, sigma=sigma, low_threshold=low, high_threshold=high)
    return skeletonize(edges)


def _neighbors(p, pts):
    x, y = p
    out = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            q = (x + dx, y + dy)
            if q in pts:
                out.append(q)
    return out


def _path_length(path):
    pts = np.asarray(path, dtype=float)
    if len(pts) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))


def trace_strokes(skel, min_len=6):
    """Trace a 1px skeleton into ordered polylines (x=col, y=row)."""
    coords = [(int(c), int(r)) for r, c in zip(*np.nonzero(skel))]
    pts = set(coords)

    def degree(p):
        return len(_neighbors(p, pts))

    endpoints = [p for p in pts if degree(p) == 1]
    junctions = set(p for p in pts if degree(p) >= 3)

    visited = set()
    strokes = []

    # Walk from each endpoint to the next endpoint/junction.
    for start in endpoints:
        if start in visited:
            continue
        path = [start]
        visited.add(start)
        cur, prev = start, None
        while True:
            nbrs = [n for n in _neighbors(cur, pts) if n not in visited]
            if not nbrs:
                break
            if prev is not None:
                dx, dy = cur[0] - prev[0], cur[1] - prev[1]
                nbrs.sort(key=lambda q: (q[0] - cur[0] - dx) ** 2 + (q[1] - cur[1] - dy) ** 2)
            nxt = nbrs[0]
            path.append(nxt)
            visited.add(nxt)
            prev, cur = cur, nxt
            if cur in junctions or cur in endpoints:
                break
        if len(path) >= 2:
            strokes.append(path)

    # Closed loops: components with no endpoints.
    remaining = pts - visited
    while remaining:
        start = remaining.pop()
        path = [start]
        visited.add(start)
        cur = start
        while True:
            nbrs = [n for n in _neighbors(cur, pts) if n not in visited]
            if not nbrs:
                break
            nxt = nbrs[0]
            path.append(nxt)
            visited.add(nxt)
            cur = nxt
            if cur == start:
                break
        if len(path) >= 4:
            strokes.append(path)
        remaining = pts - visited

    return [s for s in strokes if _path_length(s) >= min_len]


def arc_length_t_values(points):
    pts = np.asarray(points, dtype=float)
    if len(pts) == 1:
        return [0.0]
    d = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(d)])
    total = cum[-1]
    if total == 0:
        return list(np.linspace(0.0, 1.0, len(pts)))
    return [float(v) for v in (cum / total)]


def simplify(points, tolerance):
    pts = np.asarray(points, dtype=float)
    if len(pts) <= 2:
        return pts
    return np.asarray(approximate_polygon(pts, tolerance=tolerance))


def split_at_corners(points, angle_deg=45.0):
    """Split a polyline at sharp turns so each piece fits one smooth Bezier.

    A single cubic Bezier cannot represent a sharp corner (the least-squares
    fit blows up), so we cut the polyline wherever the turn angle exceeds
    ``angle_deg`` and return the pieces.
    """
    pts = np.asarray(points, dtype=float)
    if len(pts) <= 2:
        return [pts]
    segments = []
    seg = [pts[0]]
    for i in range(1, len(pts) - 1):
        v1 = pts[i] - pts[i - 1]
        v2 = pts[i + 1] - pts[i]
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        turn = 0.0
        if n1 > 0 and n2 > 0:
            cosang = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
            turn = float(np.degrees(np.arccos(cosang)))
        seg.append(pts[i])
        if turn > angle_deg:
            if len(seg) >= 2:
                segments.append(np.asarray(seg))
            seg = [pts[i]]
    seg.append(pts[-1])
    if len(seg) >= 2:
        segments.append(np.asarray(seg))
    return [s for s in segments if _path_length(s) > 0.5]


def split_closed_into_open(pts, n=6):
    """Split a (closed) contour into n equal-ARC-LENGTH open arcs.

    Arc-length splitting (rather than equal point count) keeps long, roughly
    straight sides broken into short segments, so each piece fits a clean
    Bezier instead of one wild sweeping curve.
    """
    pts = np.asarray(pts, dtype=float)
    if len(pts) < 3:
        return [pts] if len(pts) >= 2 else []
    if np.linalg.norm(pts[0] - pts[-1]) < 1.0:
        pts = pts[:-1]
    m = len(pts)
    if m < 2:
        return []
    d = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(d)])
    total = cum[-1]
    if total < 1e-6:
        return [pts]
    n = max(2, min(n, m))
    brk = [0]
    for k in range(1, n):
        i = int(np.searchsorted(cum, k * total / n))
        i = max(1, min(m - 1, i))
        if i != brk[-1]:
            brk.append(i)
    if brk[-1] != m - 1:
        brk.append(m - 1)
    brk = sorted(set(brk))
    segs = []
    for k in range(len(brk) - 1):
        a, b = brk[k], brk[k + 1]
        seg = pts[a: b + 1]
        if len(seg) >= 2:
            segs.append(seg)
    if segs and np.linalg.norm(segs[-1][-1] - pts[0]) > 1.0:
        segs[-1] = np.concatenate([segs[-1], pts[:1]], axis=0)  # close loop
    return segs


def compute_mask(gray):
    """Otsu binarize -> keep largest connected component (object silhouette)."""
    t = threshold_otsu(gray)
    mask = gray < t  # foreground = darker than background
    lab = label(mask)
    if lab.max() == 0:
        return np.zeros_like(mask, dtype=bool)
    keep = max(regionprops(lab), key=lambda r: r.area).label
    return lab == keep


def extract_contour_strokes(mask, simplify_tol, parts=8, corner_angle=45.0):
    """Outer boundary of the silhouette, split into open arcs."""
    contours = find_contours(mask.astype(float), 0.5)
    if not contours:
        return []
    outer = max(contours, key=lambda c: len(c))
    pts = np.stack([outer[:, 1], outer[:, 0]], axis=1)  # (row, col) -> (x, y)
    pts = simplify(pts, simplify_tol)
    out = []
    for arc in split_closed_into_open(pts, n=parts):
        for seg in split_at_corners(arc, corner_angle):
            if len(seg) >= 2:
                out.append(seg)
    return out


def extract_canny_strokes(gray, sigma, low, high, min_len, simplify_tol, corner_angle, bridge):
    """Canny edges -> skeleton -> strokes."""
    edges = canny(gray, sigma=sigma, low_threshold=low, high_threshold=high)
    skel = skeletonize(edges)
    if bridge > 0:
        skel = skeletonize(binary_dilation(skel, disk(bridge)))
    strokes = trace_strokes(skel, min_len=min_len)
    strokes = [simplify(s, simplify_tol) for s in strokes]
    split = []
    for s in strokes:
        split.extend(split_at_corners(s, angle_deg=corner_angle))
    return [s for s in split if len(s) >= 2 and _path_length(s) >= min_len], edges, skel


def pixel_to_cell(x, y, res, cell_size):
    gx = int(round(x / cell_size - 1.5)) + 1
    gy = int(round((res - 0.5) - y / cell_size)) + 1
    gx = max(1, min(res, gx))
    gy = max(1, min(res, gy))
    return f"x{gx}y{gy}"


def build_control_points(strokes):
    all_cp = []
    for path in strokes:
        pts = [list(p) for p in path]
        t = arc_length_t_values(pts)
        all_cp.append(utils.estimate_bezier_control_points(pts, t))
    return all_cp


def split_long_strokes(strokes, max_len=40.0):
    """Split any stroke longer than max_len px into shorter segments.

    A long, slightly-curved stroke (e.g. a trunk boundary from Canny) otherwise
    gets a single cubic Bezier whose least-squares control points swing wildly.
    """
    out = []
    for s in strokes:
        s = np.asarray(s, dtype=float)
        if len(s) < 2:
            continue
        d = np.linalg.norm(np.diff(s, axis=0), axis=1)
        total = float(d.sum())
        if total <= max_len:
            out.append(s)
        else:
            n = int(np.ceil(total / max_len))
            cum = np.concatenate([[0.0], np.cumsum(d)])
            brk = [0]
            for k in range(1, n):
                i = int(np.searchsorted(cum, k * total / n))
                i = max(1, min(len(s) - 1, i))
                if i != brk[-1]:
                    brk.append(i)
            if brk[-1] != len(s) - 1:
                brk.append(len(s) - 1)
            brk = sorted(set(brk))
            for k in range(len(brk) - 1):
                a, b = brk[k], brk[k + 1]
                if b - a >= 1:
                    out.append(s[a:b + 1])
    return out


def strokes_to_xml(strokes, res, cell_size):
    parts = ["<strokes>"]
    for i, path in enumerate(strokes, 1):
        # Dedupe consecutive points that fall in the same grid cell, keeping
        # the t-values in sync with the deduped points.
        cells = []
        kept = []
        for x, y in path:
            c = pixel_to_cell(x, y, res, cell_size)
            if not cells or cells[-1] != c:
                cells.append(c)
                kept.append((x, y))
        t = arc_length_t_values(kept)
        parts.append(f"  <s{i}>")
        parts.append(f"    <points>{', '.join(cells)}</points>")
        parts.append(f"    <t_values>{', '.join(f'{v:.2f}' for v in t)}</t_values>")
        parts.append(f"    <id>stroke_{i}</id>")
        parts.append(f"  </s{i}>")
    parts.append("</strokes>")
    return "\n".join(parts)


def render_sketch(image_path, mode="both", res=50, cell_size=12, stroke_width=7.0,
                  contour_parts=12, corner_angle=45.0, sigma=2.0, low=0.10, high=0.25,
                  min_len=15.0, simplify_tol=3.0, max_strokes=None, max_seg_len=100000.0):
    """Extract strokes (contour and/or Canny) from an image and return SVG text."""
    canvas_size = (res + 1) * cell_size
    gray = load_gray_square(image_path, canvas_size)
    contour_strokes = []
    canny_strokes = []
    if mode in ("contour", "both"):
        mask = compute_mask(gray)
        contour_strokes = extract_contour_strokes(mask, simplify_tol, contour_parts, corner_angle)
        contour_strokes = [s for s in contour_strokes if len(s) >= 2 and _path_length(s) >= min_len]
    if mode in ("edges", "both"):
        canny_strokes, _, _ = extract_canny_strokes(gray, sigma, low, high, min_len, simplify_tol, corner_angle, 0)
        canny_strokes = sorted(canny_strokes, key=_path_length, reverse=True)
        if max_strokes:
            canny_strokes = canny_strokes[:max_strokes]
    if mode == "both":
        strokes = list(contour_strokes) + canny_strokes
    elif mode == "contour":
        strokes = sorted(contour_strokes, key=_path_length, reverse=True)
        if max_strokes:
            strokes = strokes[:max_strokes]
    else:
        strokes = canny_strokes
    strokes = split_long_strokes(strokes, max_seg_len)
    all_cp = build_control_points(strokes)
    return utils.format_svg(all_cp, dim=(canvas_size, canvas_size), stroke_width=stroke_width)


def extract_strokes(image_path, mode="both", res=50, cell_size=12,
                    contour_parts=12, corner_angle=45.0, sigma=2.0, low=0.10, high=0.25,
                    min_len=15.0, simplify_tol=3.0, max_strokes=None, max_seg_len=100000.0):
    """Extract stroke polylines (pixel coords) from an image."""
    canvas_size = (res + 1) * cell_size
    gray = load_gray_square(image_path, canvas_size)
    contour_strokes = []
    canny_strokes = []
    if mode in ("contour", "both"):
        mask = compute_mask(gray)
        contour_strokes = extract_contour_strokes(mask, simplify_tol, contour_parts, corner_angle)
        contour_strokes = [s for s in contour_strokes if len(s) >= 2 and _path_length(s) >= min_len]
    if mode in ("edges", "both"):
        canny_strokes, _, _ = extract_canny_strokes(gray, sigma, low, high, min_len, simplify_tol, corner_angle, 0)
        canny_strokes = sorted(canny_strokes, key=_path_length, reverse=True)
        if max_strokes:
            canny_strokes = canny_strokes[:max_strokes]
    if mode == "both":
        strokes = list(contour_strokes) + canny_strokes
    elif mode == "contour":
        strokes = sorted(contour_strokes, key=_path_length, reverse=True)
        if max_strokes:
            strokes = strokes[:max_strokes]
    else:
        strokes = canny_strokes
    return split_long_strokes(strokes, max_seg_len)


def main():
    p = argparse.ArgumentParser(description="Image -> sketch via CV (no LLM)")
    p.add_argument("--image", type=str, required=True)
    p.add_argument("--path2save", type=str, default="results/cv")
    p.add_argument("--res", type=int, default=50)
    p.add_argument("--cell_size", type=int, default=12)
    p.add_argument("--stroke_width", type=float, default=7.0)
    p.add_argument("--sigma", type=float, default=1.5)
    p.add_argument("--low", type=float, default=0.05)
    p.add_argument("--high", type=float, default=0.20)
    p.add_argument("--simplify_tol", type=float, default=2.0)
    p.add_argument("--corner_angle", type=float, default=45.0, help="split strokes at turns larger than this (degrees)")
    p.add_argument("--min_len", type=float, default=6.0)
    p.add_argument("--max_strokes", type=int, default=None, help="keep only the N longest strokes (drops texture noise)")
    p.add_argument("--max_seg_len", type=float, default=40.0, help="split strokes longer than this (px) into shorter segments")
    p.add_argument("--bridge", type=int, default=0, help="bridge gaps in edges (dilate+skeletonize) with this px radius")
    p.add_argument("--mode", type=str, default="edges", choices=["edges", "contour", "both"], help="edges=Canny, contour=silhouette outline, both=outline + internal Canny lines")
    p.add_argument("--contour_parts", type=int, default=8, help="split the closed contour into this many open arcs")
    args = p.parse_args()

    if not os.path.exists(args.image):
        raise SystemExit(f"Image not found: {args.image}")

    canvas_size = (args.res + 1) * args.cell_size
    name = os.path.splitext(os.path.basename(args.image))[0]
    out_dir = os.path.join(args.path2save, name)
    os.makedirs(out_dir, exist_ok=True)

    gray = load_gray_square(args.image, canvas_size)

    contour_strokes = []
    canny_strokes = []

    if args.mode in ("contour", "both"):
        mask = compute_mask(gray)
        contour_strokes = extract_contour_strokes(mask, args.simplify_tol, args.contour_parts, args.corner_angle)
        contour_strokes = [s for s in contour_strokes if len(s) >= 2 and _path_length(s) >= args.min_len]

    if args.mode in ("edges", "both"):
        canny_strokes, edges, skel = extract_canny_strokes(
            gray, args.sigma, args.low, args.high, args.min_len,
            args.simplify_tol, args.corner_angle, args.bridge,
        )
        # intermediate debug images (black lines on white background)
        Image.fromarray(((~edges) * 255).astype(np.uint8)).save(os.path.join(out_dir, f"{name}_canny.png"))
        Image.fromarray(((~skel) * 255).astype(np.uint8)).save(os.path.join(out_dir, f"{name}_skeleton.png"))
        canny_strokes = sorted(canny_strokes, key=_path_length, reverse=True)
        if args.max_strokes:
            canny_strokes = canny_strokes[: args.max_strokes]

    if args.mode == "both":
        # simply stack the full Canny edges + the silhouette contour
        strokes = list(contour_strokes) + canny_strokes
    elif args.mode == "contour":
        strokes = sorted(contour_strokes, key=_path_length, reverse=True)
        if args.max_strokes:
            strokes = strokes[: args.max_strokes]
    else:  # edges
        strokes = canny_strokes

    print(f"提取到 {len(strokes)} 条笔画")

    strokes = split_long_strokes(strokes, args.max_seg_len)

    # render
    all_cp = build_control_points(strokes)
    svg = utils.format_svg(all_cp, dim=(canvas_size, canvas_size), stroke_width=args.stroke_width)
    svg_path = os.path.join(out_dir, f"{name}.svg")
    with open(svg_path, "w") as f:
        f.write(svg)

    png_path = os.path.join(out_dir, f"{name}.png")
    cairosvg.svg2png(url=svg_path, write_to=png_path, background_color="white")

    # canvas version (grid background)
    grid_img, _ = utils.create_grid_image(res=args.res, cell_size=args.cell_size, header_size=args.cell_size)
    canvas_png = os.path.join(out_dir, f"{name}_canvas.png")
    cairosvg.svg2png(url=svg_path, write_to=canvas_png)
    fg = Image.open(canvas_png)
    grid_img.paste(Image.open(canvas_png), (0, 0), fg)
    grid_img.save(canvas_png)

    # grid-cell XML for future LLM naturalization
    xml = strokes_to_xml(strokes, args.res, args.cell_size)
    with open(os.path.join(out_dir, "strokes.xml"), "w") as f:
        f.write(xml)

    print(f"已保存到 {out_dir}/")
    print(f"  {name}.svg / {name}.png / {name}_canvas.png / strokes.xml")


if __name__ == "__main__":
    main()
