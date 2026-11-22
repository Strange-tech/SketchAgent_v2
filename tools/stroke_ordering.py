"""Reorder strokes into a human-like drawing order (pixel coordinates).

Coordinates: x is left-to-right, y is top-to-bottom, so "bottom" means a LARGER
y value. Ordering key: part (0=trunk before 1=foliage), layer (0=outline before
1=detail), then bottom-to-top, then left-to-right. Each stroke is also oriented
bottom-to-top (left-to-right for horizontal strokes).
"""
import json


def _bottom_anchor(points):
    """(y, x) of the bottom-most point: largest y, tie-break smallest x (left)."""
    x, y = max(points, key=lambda p: (p[1], -p[0]))
    return y, x


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


class OrderStrokesTool:
    def definition(self):
        return {
            "name": "order_strokes",
            "description": (
                "Reorder strokes into a human-like drawing order: part (0=trunk "
                "before 1=foliage), then layer (0=outline before 1=detail), then "
                "bottom-to-top, then left-to-right. Coordinates are pixels (x "
                "left-to-right, y top-to-bottom; bottom = larger y). Each stroke is "
                "also oriented bottom-to-top (left-to-right for horizontal strokes)."
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
                                "layer": {"type": "integer", "description": "0=outline, 1=detail"},
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

        keyed = []
        for s in strokes:
            ay, ax = _bottom_anchor(s["points"])
            keyed.append((s.get("part", 0), s.get("layer", 0), -ay, ax, s))
        keyed.sort(key=lambda k: (k[0], k[1], k[2], k[3]))

        ordered = [_fix_direction(k[4]) for k in keyed]
        return json.dumps({"ordered": ordered}, indent=2)
