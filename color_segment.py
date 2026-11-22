"""Color-based segmentation (Stage B alternative): green -> foliage, grey/brown -> trunk.

HSV 规则：
  树叶 = 绿色系（H 在绿色区间，且有一定饱和度）
  树干 = 低饱和灰（中亮度）或 暖褐（H 偏红橙）
  其余（蓝天/白底）→ 背景，不输出
"""
import argparse
import os

import numpy as np
from PIL import Image
from skimage.color import rgb2hsv
from skimage.morphology import remove_small_objects, binary_opening, rectangle


def segment(img_path, green_lo=0.18, green_hi=0.48, s_min=0.15,
            grey_s_max=0.22, brown_h_hi=0.10, v_lo=0.12, v_hi=0.88, min_area=200):
    img = np.array(Image.open(img_path).convert("RGB")).astype(np.float32) / 255.0
    hsv = rgb2hsv(img)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    # 树叶：绿色系（H 在绿区间，饱和度足够）
    foliage = (h > green_lo) & (h < green_hi) & (s > s_min)

    # 树干：灰（低饱和、中亮度）或 暖褐（H 偏红橙）
    grey = (s < grey_s_max) & (v > v_lo) & (v < v_hi)
    brown = (h < brown_h_hi) & (s > s_min) & (v > v_lo)
    trunk = (grey | brown) & ~foliage

    foliage = remove_small_objects(foliage, min_size=min_area)
    trunk = remove_small_objects(trunk, min_size=min_area)
    return foliage, trunk


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--green_lo", type=float, default=0.18)
    p.add_argument("--green_hi", type=float, default=0.48)
    p.add_argument("--s_min", type=float, default=0.15)
    p.add_argument("--grey_s_max", type=float, default=0.22)
    p.add_argument("--brown_h_hi", type=float, default=0.10)
    p.add_argument("--min_area", type=int, default=200)
    p.add_argument("--open_radius", type=int, default=0, help="形态学开运算半径(px)，去掉树干上的细凸起/树枝；0=关闭")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    foliage, trunk = segment(args.image, args.green_lo, args.green_hi, args.s_min,
                             args.grey_s_max, args.brown_h_hi, min_area=args.min_area)
    if args.open_radius > 0:
        trunk = binary_opening(trunk, rectangle(1, 2 * args.open_radius))

    h, w = foliage.shape
    print(f"图像 {w}x{h}: foliage 覆盖 {foliage.mean():.3f}, trunk 覆盖 {trunk.mean():.3f}")

    Image.fromarray((foliage * 255).astype(np.uint8), mode="L").save(os.path.join(args.outdir, "foliage_mask.png"))
    Image.fromarray((trunk * 255).astype(np.uint8), mode="L").save(os.path.join(args.outdir, "trunk_mask.png"))

    # 可视化：原图 | foliage | trunk
    img = Image.open(args.image).convert("RGB")
    thumbs = [img.resize((h, h))]
    for m in (foliage, trunk):
        thumbs.append(Image.fromarray((m * 255).astype(np.uint8), mode="L").convert("RGB").resize((h, h)))
    viz = Image.new("RGB", (sum(t.width for t in thumbs), h), "white")
    x = 0
    for t in thumbs:
        viz.paste(t, (x, 0)); x += t.width
    viz.save(os.path.join(args.outdir, "color_seg_viz.png"))
    print(f"已保存 {args.outdir}/ : foliage_mask.png, trunk_mask.png, color_seg_viz.png")


if __name__ == "__main__":
    main()
