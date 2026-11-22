"""Combine trunk + foliage SVGs (produced by img2sketch_cv.py) into one sketch.

Trunk strokes come first, then foliage strokes. Stroke groups are re-numbered.
"""
import argparse
import re

import cairosvg


def extract_groups(svg_text):
    return re.findall(r'<g id="s\d+"[^>]*>.*?</g>', svg_text, re.S)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trunk_svg", required=True)
    p.add_argument("--foliage_svg", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    groups = extract_groups(open(args.trunk_svg).read()) + extract_groups(open(args.foliage_svg).read())
    body = []
    for i, g in enumerate(groups, 1):
        body.append(re.sub(r'id="s\d+"', f'id="s{i}"', g))

    svg = '<svg width="612" height="612" xmlns="http://www.w3.org/2000/svg">\n' + "\n".join(body) + "\n</svg>"
    with open(args.out, "w") as f:
        f.write(svg)
    cairosvg.svg2png(url=args.out, write_to=args.out.replace(".svg", ".png"), background_color="white")
    print(f"合并 {len(groups)} 条笔画 (trunk={len(extract_groups(open(args.trunk_svg).read()))} + foliage={len(extract_groups(open(args.foliage_svg).read()))}) -> {args.out}")


if __name__ == "__main__":
    main()
