"""测试不同 dedup 阈值，每个阈值输出一张最终图（蓝 contour + 橙 Canny）。"""
import json
import os

from main import (
    Registry, DetectStrokeDuplicationTool, CONTOUR_COLOR, CANNY_COLOR,
    _render_colored, _strokes_to_lists, _save_mask, segment, make_masked, cv,
)


def main():
    # 提取笔画（只做一次，四个阈值共用）
    foliage, trunk = segment("input/tree.jpg")
    os.makedirs("results/seg", exist_ok=True)
    _save_mask(foliage, "results/seg/foliage_mask.png")
    _save_mask(trunk, "results/seg/trunk_mask.png")
    foliage_rgb, trunk_rgb = make_masked(
        "input/tree.jpg", "results/seg/foliage_mask.png", "results/seg/trunk_mask.png", "results/masked"
    )
    trunk_c = cv.extract_strokes(trunk_rgb, mode="contour", contour_parts=24, corner_angle=40, max_seg_len=40)
    trunk_e = cv.extract_strokes(trunk_rgb, mode="edges", corner_angle=40, max_strokes=40, max_seg_len=40)
    fol_c = cv.extract_strokes(foliage_rgb, mode="contour", contour_parts=8, corner_angle=180, max_seg_len=100000)
    fol_e = cv.extract_strokes(foliage_rgb, mode="edges", corner_angle=180, max_strokes=30, max_seg_len=100000)

    strokes = trunk_c + trunk_e + fol_c + fol_e
    colors = ([CONTOUR_COLOR] * len(trunk_c) + [CANNY_COLOR] * len(trunk_e) +
              [CONTOUR_COLOR] * len(fol_c) + [CANNY_COLOR] * len(fol_e))

    registry = Registry(DetectStrokeDuplicationTool())
    point_lists = _strokes_to_lists(strokes)

    for t in (1, 3, 5, 7):
        r = json.loads(registry.execute(
            "detect_stroke_duplication",
            json.dumps({"strokes": point_lists, "threshold": float(t)}),
        ))
        to_remove = set()
        for pair in r["duplicate_pairs"]:
            i, j = pair["i"], pair["j"]
            if i not in to_remove and j not in to_remove:
                to_remove.add(j)
        keep = [i for i in range(len(strokes)) if i not in to_remove]
        out = f"results/threshold_{t}.svg"
        _render_colored([strokes[i] for i in keep], [colors[i] for i in keep], out)
        print(f"threshold={t}: {r['duplicate_count']} 对重复 -> {len(keep)} 笔 -> {out}")


if __name__ == "__main__":
    main()
