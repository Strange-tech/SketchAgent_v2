"""Combine trunk+foliage contour+Canny strokes into one <strokes> XML (with duplicates).

Uniform params for both parts (no trunk-specific hardcoding). The agent then
dedups + merges + reorders.
"""
import re
import numpy as np
import img2sketch_cv as cv

RES = 50
CELL = 12


def extract(image_path, contour_parts=12, corner_angle=60.0, sigma=2.0, low=0.10, high=0.25, min_len=15, simplify_tol=3.0):
    canvas = (RES + 1) * CELL
    gray = cv.load_gray_square(image_path, canvas)
    mask = cv.compute_mask(gray)
    contour = cv.extract_contour_strokes(mask, simplify_tol, contour_parts, corner_angle)
    canny, _, _ = cv.extract_canny_strokes(gray, sigma, low, high, min_len, simplify_tol, corner_angle, 0)
    return contour + canny  # pixel-coord segments


def parse_cell(c):
    m = re.match(r"x(\d+)y(\d+)", c)
    return int(m.group(1)), int(m.group(2))


def main():
    trunk = extract("results/masked/trunk_rgb.png")
    foliage = extract("results/masked/foliage_rgb.png")
    all_segs = trunk + foliage  # trunk first, then foliage
    # keep the longest strokes (drop tiny over-fragmented bits, keep the duplicates)
    all_segs = sorted(all_segs, key=cv._path_length, reverse=True)[:100]

    parts = ["<strokes>"]
    n = 0
    for seg in all_segs:
        cells = []
        for x, y in seg:
            c = cv.pixel_to_cell(x, y, RES, CELL)
            if not cells or cells[-1] != c:
                cells.append(c)
        if len(cells) < 2:
            continue
        n += 1
        pts = np.array([parse_cell(c) for c in cells], dtype=float)
        d = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        cum = np.concatenate([[0.0], np.cumsum(d)])
        if cum[-1] > 0:
            t = [float(v / cum[-1]) for v in cum]
        else:
            t = [float(i) / (len(cells) - 1) for i in range(len(cells))]
        parts.append(f"  <s{n}>")
        parts.append(f"    <points>{', '.join(cells)}</points>")
        parts.append(f"    <t_values>{', '.join(f'{v:.2f}' for v in t)}</t_values>")
        parts.append(f"    <id>stroke_{n}</id>")
        parts.append(f"  </s{n}>")
    parts.append("</strokes>")

    out = "\n".join(parts)
    open("results/masked/combined_strokes.xml", "w").write(out)
    print(f"合并 {n} 条笔画(含重复) -> results/masked/combined_strokes.xml")


if __name__ == "__main__":
    main()
