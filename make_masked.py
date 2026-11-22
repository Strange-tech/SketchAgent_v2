"""Generate foliage-masked and trunk-masked RGB images from masks."""
import os

import numpy as np
from PIL import Image


def make_masked(tree_img_path, foliage_mask_path, trunk_mask_path, outdir="results/masked"):
    img = np.array(Image.open(tree_img_path).convert("RGB"))
    fol = np.array(Image.open(foliage_mask_path).convert("L")) > 128
    trk = np.array(Image.open(trunk_mask_path).convert("L")) > 128
    os.makedirs(outdir, exist_ok=True)
    frgb = img.copy(); frgb[~fol] = 255
    trgb = img.copy(); trgb[~trk] = 255
    foliage_path = os.path.join(outdir, "foliage_rgb.png")
    trunk_path = os.path.join(outdir, "trunk_rgb.png")
    Image.fromarray(frgb).save(foliage_path)
    Image.fromarray(trgb).save(trunk_path)
    return foliage_path, trunk_path


if __name__ == "__main__":
    foliage_path, trunk_path = make_masked(
        "input/tree.jpg",
        "results/color_seg/foliage_mask.png",
        "results/color_seg/trunk_mask.png",
        "results/masked",
    )
    print(f"saved {foliage_path} + {trunk_path}")
