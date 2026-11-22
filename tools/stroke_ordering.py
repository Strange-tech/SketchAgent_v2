"""Reorder strokes into a human-like drawing order (pixel coordinates).

Coordinates: x is left-to-right, y is top-to-bottom, so "bottom" means a LARGER
y value. Ordering (most significant first):
  1. part: trunk (0) before foliage (1)
  2. layer: contour/outline (0) before detail (1)
  3. within contour: a continuous chain — after the first stroke, each next stroke
     starts where the previous ended, so the silhouette is traced rather than
     jumped around
  4. within detail: long before short, then bottom-to-top, then left-to-right

Detail strokes are also oriented bottom-to-top (left-to-right for horizontal strokes).
"""
import json


def _stroke_length(points):
    """Total polyline length (sum of segment distances) in pixels."""
    if len(points) < 2:
        return 0.0
    total = 0.0
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        total += ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
    return total


def _dist(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def _fix_direction(stroke):
    pts = stroke["points"]
    if len(pts) < 2:
        return stroke
    dx = pts[-1][0] - pts[0][0]
    dy = pts[-1][1] - pts[0][1]
    if abs(dy) >= abs(dx):
        # vertical-ish: draw bottom-to-top (start at larger y)
        reverse = pts[0][1] < pts[-1][1]
    else:
        # horizontal-ish: draw left-to-right (start at smaller x)
        reverse = pts[0][0] > pts[-1][0]
    if reverse:
        stroke["points"] = list(reversed(pts))
    return stroke


def _chain_strokes(strokes):
    """Reorder strokes into a continuous chain (greedy nearest-neighbour).

    Starts from the bottom-most stroke, then repeatedly appends the stroke whose
    start (or end, flipped) is closest to the current chain's end, so each stroke
    is adjacent to the previous one.
    """
    if len(strokes) <= 1:
        return strokes

    remaining = list(strokes)

    def bottom_anchor(s):
        return max(p[1] for p in s["points"])

    start = min(remaining, key=lambda s: (-bottom_anchor(s), min(p[0] for p in s["points"])))
    remaining.remove(start)
    chain = [start]
    cur = start["points"][-1]

    while remaining:
        best, best_d, best_flip = None, float("inf"), False
        for s in remaining:
            d_start = _dist(cur, s["points"][0])
            if d_start < best_d:
                best, best_d, best_flip = s, d_start, False
            d_end = _dist(cur, s["points"][-1])
            if d_end < best_d:
                best, best_d, best_flip = s, d_end, True
        if best_flip:
            best["points"] = list(reversed(best["points"]))
        chain.append(best)
        remaining.remove(best)
        cur = best["points"][-1]

    return chain


class OrderStrokesTool:
    def definition(self):
        return {
            "name": "order_strokes",
            "description": (
                "Reorder strokes into a human-like drawing order: trunk (part=0) before "
                "foliage (part=1), outline/contour (layer=0) before detail (layer=1). "
                "Contour strokes are chained so each stroke starts where the previous one "
                "ended. Detail strokes are ordered long-before-short, then bottom-to-top, "
                "then left-to-right. Coordinates are pixels (x left-to-right, y "
                "top-to-bottom; bottom = larger y)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "strokes": {
                        "type": "array",
                        "description": "List of strokes; each is an object {points, part, layer}.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "points": {
                                    "type": "array",
                                    "items": {
                                        "type": "array",
                                        "items": {"type": "number"},
                                        "minItems": 2,
                                        "maxItems": 2,
                                    },
                                },
                                "part": {"type": "integer", "description": "0=trunk, 1=foliage"},
                                "layer": {"type": "integer", "description": "0=contour/outline, 1=detail"},
                            },
                            "required": ["points"],
                        },
                    },
                },
                "required": ["strokes"],
            },
        }

    def execute(self, args):
        try:
            params = json.loads(args)
        except json.JSONDecodeError as e:
            return "Error: Arguments are not valid JSON: " + str(e)

        strokes = params.get("strokes", [])
        for idx, s in enumerate(strokes):
            s["index"] = idx

        # group by (part, layer): trunk-contour, trunk-detail, foliage-contour, foliage-detail
        groups = {}
        for s in strokes:
            key = (s.get("part", 0), s.get("layer", 0))
            groups.setdefault(key, []).append(s)

        ordered = []
        for key in sorted(groups.keys()):
            _, layer = key
            grp = groups[key]
            if layer == 0:
                # contour/outline: chain into a continuous path
                ordered.extend(_chain_strokes(grp))
            else:
                # detail: long -> short, bottom -> top, left -> right
                grp.sort(key=lambda s: (
                    -_stroke_length(s["points"]),
                    -max(p[1] for p in s["points"]),
                    min(p[0] for p in s["points"]),
                ))
                ordered.extend([_fix_direction(s) for s in grp])

        return json.dumps({"ordered": ordered}, indent=2)
