"""Human-like stroke ordering (plan.md Stage A).

Reorders strokes the way a person draws a tree:
  1. part    : trunk (0) before foliage (1)
  2. layer   : outline (0) before detail (1)   [foliage only]
  3. spatial : bottom-to-top, then left-to-right
  4. direction: every stroke drawn bottom-to-top

Coordinate convention (grid "xNyM"):
  x in [1, res] = left -> right
  y in [1, res] = BOTTOM -> TOP  (y=1 is the bottom of the canvas)
"""
import argparse
import json


def bottom_anchor(points):
    """(y, x) of the bottom-most point: smallest y, tie-break smallest x."""
    x, y = min(points, key=lambda p: (p[1], p[0]))
    return y, x


def fix_direction(stroke):
    """Reverse the stroke if it runs top-to-bottom, so it draws bottom-to-top."""
    pts = stroke["points"]
    if len(pts) >= 2 and pts[0][1] > pts[-1][1]:
        stroke["points"] = list(reversed(pts))
        t = stroke.get("t_values")
        if t is not None and len(t) == len(pts):
            stroke["t_values"] = [1.0 - v for v in reversed(t)]
    return stroke


def humanize_order(strokes):
    """Return a new list of strokes in human drawing order (input unchanged)."""
    keyed = []
    for s in strokes:
        ay, ax = bottom_anchor(s["points"])
        keyed.append((s.get("part", 0), s.get("layer", 0), ay, ax, s))
    keyed.sort(key=lambda k: (k[0], k[1], k[2], k[3]))
    return [fix_direction(k[4]) for k in keyed]


def main():
    p = argparse.ArgumentParser(description="Reorder strokes into a human-like drawing order")
    p.add_argument("--input", required=True, help="JSON file: list of strokes {part, layer, points, t_values}")
    p.add_argument("--output", required=True, help="output JSON file")
    args = p.parse_args()

    with open(args.input) as f:
        strokes = json.load(f)
    ordered = humanize_order(strokes)
    with open(args.output, "w") as f:
        json.dump(ordered, f, indent=2)
    print(f"已排序 {len(ordered)} 条笔画 -> {args.output}")
    for i, s in enumerate(ordered, 1):
        ay, ax = bottom_anchor(s["points"])
        print(f"  {i:3d}. part={s.get('part', 0)} layer={s.get('layer', 0)} anchor=({ax},{ay})")


if __name__ == "__main__":
    main()
