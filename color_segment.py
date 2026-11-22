"""Color-based segmentation: saturated (non-blue) pixels -> foliage, desaturated -> trunk.

Handles green, yellow, orange and red foliage by using SATURATION (leaves are
vivid colors) instead of a fixed green hue. The trunk is detected as desaturated
(grey/brown) pixels of medium brightness. Blue sky is excluded by hue.
"""
import argparse
import os

import numpy as np
from PIL import Image
from skimage.color import rgb2hsv
from skimage.measure import label
from skimage.morphology import remove_small_objects, binary_opening, rectangle


def segment(img_path, s_min=0.25, v_lo=0.10, v_hi=0.70,
            blue_lo=0.48, blue_hi=0.75, brown_s_max=0.45, min_area=200, tree_mask=None):
    img = np.array(Image.open(img_path).convert("RGB")).astype(np.float32) / 255.0
    hsv = rgb2hsv(img)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    # foliage = saturated, non-blue (covers green/yellow/orange/red), not too dark
    non_blue = ~((h > blue_lo) & (h < blue_hi))
    foliage = non_blue & (s > s_min) & (v > v_lo)

    # trunk = desaturated grey (medium-dark) OR brown (warm hue, medium saturation, dark)
    grey = (s < s_min) & (v > v_lo) & (v < v_hi)
    brown = (h < 0.10) & (s > s_min) & (s < brown_s_max) & (v > v_lo) & (v < v_hi)
    trunk = (grey | brown) & ~foliage

    # restrict to the SAM3 tree region if provided (pre-segmentation)
    if tree_mask is not None:
        tree_mask = np.asarray(tree_mask, dtype=bool)
        foliage = foliage & tree_mask
        trunk = trunk & tree_mask

    foliage = remove_small_objects(foliage, min_size=min_area)
    trunk = remove_small_objects(trunk, min_size=min_area)
    return foliage, trunk


def refine_trunk_mask(image, trunk_mask, tree_mask=None, color_tol=50.0, sat_margin=0.10):
    """Grow a coarse trunk mask into connected pixels of similar color.

    Samples the trunk's representative color and saturation from `trunk_mask`, then
    adds the connected components of similarly-colored, low-saturation pixels that
    touch the trunk seed. The saturation cap keeps the growth on the (desaturated)
    trunk/branches and away from saturated foliage. `tree_mask` bounds the growth to
    the tree region. The seed is never removed, so the result is a superset of
    `trunk_mask`.
    """
    img = np.array(image).astype(np.float32)
    mask = np.asarray(trunk_mask, dtype=bool)
    if not mask.any():
        return mask

    trunk_color = np.median(img[mask], axis=0)  # representative trunk color
    dist = np.sqrt(((img - trunk_color) ** 2).sum(axis=2))

    # low-saturation constraint: grown pixels may be only slightly more saturated
    # than the trunk (trunk/bark is desaturated; foliage is saturated).
    sat = rgb2hsv(img / 255.0)[..., 1]
    sat_cap = float(np.percentile(sat[mask], 90)) + sat_margin

    similar = (dist <= color_tol) & (sat <= sat_cap)

    labels, _ = label(similar, connectivity=2, return_num=True)
    seed_ids = np.unique(labels[mask])
    seed_ids = seed_ids[seed_ids != 0]
    refined = mask | np.isin(labels, seed_ids)  # grow, never shrink

    if tree_mask is not None:
        refined = refined & np.asarray(tree_mask, dtype=bool)

    return refined


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--s_min", type=float, default=0.25, help="saturation threshold: above = foliage, below = trunk")
    p.add_argument("--v_lo", type=float, default=0.10)
    p.add_argument("--v_hi", type=float, default=0.70)
    p.add_argument("--blue_lo", type=float, default=0.48)
    p.add_argument("--blue_hi", type=float, default=0.75)
    p.add_argument("--brown_s_max", type=float, default=0.45)
    p.add_argument("--min_area", type=int, default=200)
    p.add_argument("--open_radius", type=int, default=0, help="morphological opening radius for trunk; 0=off")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    foliage, trunk = segment(args.image, args.s_min, args.v_lo, args.v_hi,
                             args.blue_lo, args.blue_hi, args.brown_s_max, min_area=args.min_area)
    if args.open_radius > 0:
        trunk = binary_opening(trunk, rectangle(1, 2 * args.open_radius))

    h, w = foliage.shape
    print(f"图像 {w}x{h}: foliage 覆盖 {foliage.mean():.3f}, trunk 覆盖 {trunk.mean():.3f}")

    Image.fromarray((foliage * 255).astype(np.uint8), mode="L").save(os.path.join(args.outdir, "foliage_mask.png"))
    Image.fromarray((trunk * 255).astype(np.uint8), mode="L").save(os.path.join(args.outdir, "trunk_mask.png"))

    img = Image.open(args.image).convert("RGB")
    thumbs = [img.resize((h, h))]
    for m in (foliage, trunk):
        thumbs.append(Image.fromarray((m * 255).astype(np.uint8), mode="L").convert("RGB").resize((h, h)))
    viz = Image.new("RGB", (sum(t.width for t in thumbs), h), "white")
    x = 0
    for t in thumbs:
        viz.paste(t, (x, 0))
        x += t.width
    viz.save(os.path.join(args.outdir, "color_seg_viz.png"))
    print(f"已保存 {args.outdir}/ : foliage_mask.png, trunk_mask.png, color_seg_viz.png")


if __name__ == "__main__":
    main()
