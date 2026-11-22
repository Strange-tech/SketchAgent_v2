"""Detect near-duplicate stroke curves.

The contour-vs-Canny pipeline produces two strokes tracing the same boundary:
a long silhouette outline and many short Canny segments along the same line.
The symmetric chamfer distance is large for such a "subset" overlap (the long
stroke has many points far from the short segment), so duplication is detected
by COVERAGE instead: the fraction of one stroke's points that lie within a
distance threshold of the other. max(coverage(a,b), coverage(b,a)) catches both
full duplicates and subset overlaps.
"""
import json

import numpy as np


def stroke_distance(a, b):
    """Symmetric average minimum (chamfer) distance between two polylines."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) == 0 or len(b) == 0:
        return float("inf")
    d_ab = np.min(np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2), axis=1)
    d_ba = np.min(np.linalg.norm(b[:, None, :] - a[None, :, :], axis=2), axis=1)
    return float((d_ab.mean() + d_ba.mean()) / 2.0)


def coverage(a, b, threshold):
    """Fraction of a's points that lie within `threshold` distance of b."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) == 0 or len(b) == 0:
        return 0.0
    d_ab = np.min(np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2), axis=1)
    return float((d_ab <= threshold).mean())


def resample(polyline, n=30):
    """Resample a polyline to n uniformly-spaced points along its arc length.

    The pipeline over-segments strokes into 2-3 point segments; coverage on such
    coarse vertices is only 0 / 0.5 / 1.0 and misses overlaps whose endpoints are
    staggered. Resampling to a dense point cloud captures the overlap in between.
    """
    pts = np.asarray(polyline, dtype=float)
    if len(pts) < 2:
        return pts
    d = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(d)])
    total = cum[-1]
    if total < 1e-9:
        return pts
    targets = np.linspace(0.0, total, n)
    out = []
    for t in targets:
        i = int(np.searchsorted(cum, t, side="right")) - 1
        i = max(0, min(len(pts) - 2, i))
        frac = (t - cum[i]) / (cum[i + 1] - cum[i] + 1e-9)
        out.append(pts[i] + frac * (pts[i + 1] - pts[i]))
    return np.array(out)


class DetectStrokeDuplicationTool:
    def definition(self):
        return {
            "name": "detect_stroke_duplication",
            "description": (
                "Detect near-duplicate stroke curves. For each pair it reports the "
                "coverage (fraction of one stroke's points within the distance "
                "threshold of the other). Pairs with max coverage above the coverage "
                "threshold are flagged as duplicates. Also returns a duplication degree."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "strokes": {
                        "type": "array",
                        "description": "List of strokes; each stroke is a list of [x, y] points.",
                        "items": {
                            "type": "array",
                            "items": {
                                "type": "array",
                                "items": {"type": "number"},
                                "minItems": 2,
                                "maxItems": 2,
                            },
                        },
                    },
                    "threshold": {
                        "type": "number",
                        "description": "Distance threshold (px): points within this distance are considered overlapping.",
                    },
                    "coverage_threshold": {
                        "type": "number",
                        "description": "Coverage threshold (0..1): pairs with max coverage above this are duplicates.",
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
        dist_thr = float(params.get("threshold", 8.0))
        cov_thr = float(params.get("coverage_threshold", 0.8))
        if not strokes:
            return json.dumps({"error": "no strokes provided"})

        # dense point cloud per stroke (so coverage isn't dominated by 2-3 vertices)
        resampled = [resample(s, 30) for s in strokes]

        pairs = []
        for i in range(len(resampled)):
            for j in range(i + 1, len(resampled)):
                cov = max(
                    coverage(resampled[i], resampled[j], dist_thr),
                    coverage(resampled[j], resampled[i], dist_thr),
                )
                if cov > cov_thr:
                    pairs.append({"i": i, "j": j, "coverage": round(cov, 3)})
        pairs.sort(key=lambda p: -p["coverage"])

        n = len(strokes)
        total_pairs = n * (n - 1) // 2
        dup_degree = len(pairs) / total_pairs if total_pairs else 0.0

        return json.dumps(
            {
                "total_strokes": n,
                "distance_threshold": dist_thr,
                "coverage_threshold": cov_thr,
                "duplicate_pairs": pairs,
                "duplicate_count": len(pairs),
                "duplication_degree": round(dup_degree, 4),
            },
            indent=2,
        )
