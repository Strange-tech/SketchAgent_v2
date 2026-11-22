"""Reorder strokes into a human-like drawing order (pixel coordinates).

Coordinates: x is left-to-right, y is top-to-bottom, so "bottom" means a LARGER
y value. Ordering key (most significant first):
  1. length: long strokes before short strokes
  2. bottom-to-top: lower strokes before higher strokes
  3. left-to-right: left strokes before right strokes

Each stroke is also oriented bottom-to-top (left-to-right for horizontal strokes),
so the pen moves the way a human would.
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
                "Reorder strokes into a human-like drawing order: longest strokes "
                "first, then bottom-to-top, then left-to-right. Coordinates are "
                "pixels (x left-to-right, y top-to-bottom; bottom = larger y). Each "
                "stroke is also oriented bottom-to-top (left-to-right for horizontal "
                "strokes)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "strokes": {
                        "type": "array",
                        "description": "List of strokes; each is an object {points}.",
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
            pts = s["points"]
            length = _stroke_length(pts)
            bottom_y = max(p[1] for p in pts)
            left_x = min(p[0] for p in pts)
            # sort ascending on (-length, -bottom_y, left_x) == long, bottom, left first
            keyed.append((-length, -bottom_y, left_x, s))
        keyed.sort(key=lambda k: (k[0], k[1], k[2]))

        ordered = [_fix_direction(k[3]) for k in keyed]
        return json.dumps({"ordered": ordered}, indent=2)
