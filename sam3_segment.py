"""Stage B: SAM3 semantic segmentation (trunk vs foliage).

Uses facebookresearch/sam3 (copied to ./sam3) with a local checkpoint, and
text prompts to segment the image into binary masks per part.

Run with the `layout` conda env (has torch + timm + PIL):
  ~/miniconda3/envs/layout/bin/python sam3_segment.py --image input/tree.jpg --outdir results/seg
"""
import argparse
import os
import sys

import numpy as np
from PIL import Image

SAM3_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sam3")
CHECKPOINT = os.environ.get(
    "SAM3_CHECKPOINT",
    "/home/data/johnli/.cache/modelscope/models/facebook--sam3/snapshots/master/sam3.pt",
)


def build_processor(confidence_threshold=0.5):
    sys.path.insert(0, SAM3_ROOT)
    import torch
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()

    bpe_path = os.path.join(SAM3_ROOT, "sam3", "assets", "bpe_simple_vocab_16e6.txt.gz")
    model = build_sam3_image_model(
        bpe_path=bpe_path,
        checkpoint_path=CHECKPOINT,
        load_from_HF=False,
    )
    return Sam3Processor(model, confidence_threshold=confidence_threshold)


def segment(image, processor, prompts):
    """Return {prompt: bool ndarray (H, W)} of the union of detected instances."""
    state = processor.set_image(image)
    masks = {}
    for prompt in prompts:
        processor.reset_all_prompts(state)
        state = processor.set_text_prompt(prompt=prompt, state=state)
        m = state["masks"]  # [N, 1, H, W] bool
        print(f"  masks shape: {tuple(m.shape)}")
        if m.shape[0] > 0:
            combined = np.squeeze(m.any(dim=0).cpu().numpy())
        else:
            combined = np.zeros((image.height, image.width), dtype=bool)
        masks[prompt] = combined
        print(f"  prompt={prompt!r}: {m.shape[0]} 个实例, combined shape {combined.shape}, 覆盖占比 {combined.mean():.3f}")
    return masks


def main():
    p = argparse.ArgumentParser(description="SAM3 text-prompt segmentation")
    p.add_argument("--image", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--prompts", default="trunk,foliage", help="逗号分隔的文本提示词")
    p.add_argument("--confidence", type=float, default=0.5)
    args = p.parse_args()

    prompts = [x.strip() for x in args.prompts.split(",") if x.strip()]
    os.makedirs(args.outdir, exist_ok=True)

    image = Image.open(args.image).convert("RGB")
    print(f"图像 {image.size}, 提示词 {prompts}, 阈值 {args.confidence}")

    processor = build_processor(args.confidence)
    masks = segment(image, processor, prompts)

    # 保存每类二值掩码（白=前景）
    for name, mask in masks.items():
        safe = name.strip().replace(" ", "_")
        Image.fromarray((mask * 255).astype(np.uint8), mode="L").save(
            os.path.join(args.outdir, f"{safe}_mask.png")
        )

    # 拼接一张可视化：原图 | trunk | foliage
    from PIL import ImageDraw
    names = list(masks.keys())
    h = image.height
    thumbs = [image.resize((h, h))]
    for name in names:
        m = masks[name]
        thumbs.append(Image.fromarray((m * 255).astype(np.uint8), mode="L").convert("RGB").resize((h, h)))
    w = sum(t.width for t in thumbs)
    viz = Image.new("RGB", (w, h), "white")
    x = 0
    for t in thumbs:
        viz.paste(t, (x, 0)); x += t.width
    viz.save(os.path.join(args.outdir, "segmentation_viz.png"))

    print(f"已保存到 {args.outdir}/ : " + ", ".join(f"{n.strip().replace(chr(32), chr(95))}_mask.png" for n in names) + ", segmentation_viz.png")


if __name__ == "__main__":
    main()
