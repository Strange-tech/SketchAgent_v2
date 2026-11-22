"""Run the sketch pipeline (main.run_pipeline) on all images in input/ and build a gallery page.

The gallery is pure visualization: it reuses the exact same pipeline as the CLI
entry point (main.run_pipeline) and only wraps each result in a stroke-by-stroke
replay cell, so the gallery can never drift out of sync with the CLI.
"""
import base64
import io
import os

from PIL import Image

import sam3_segment
from main import run_pipeline, sam3_tree_masks


HEADER = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SketchAgent Tree Gallery</title>
<style>
  body { font-family: system-ui, -apple-system, sans-serif; background: #f3f0e8; color: #1c1a16; margin: 0; padding: 24px; }
  h1 { margin: 0 0 4px; font-size: 26px; }
  .sub { color: #6f6a5f; margin: 0 0 20px; }
  .grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; }
  .cell { background: #fff; border: 1px solid #e8e3d8; border-radius: 10px; padding: 10px; }
  .cell .label { font-size: 13px; color: #555; margin-bottom: 6px; font-weight: 600; }
  .cell svg { width: 100%; height: auto; display: block; background: #fff; border-radius: 4px; }
  .cell svg path { stroke-width: 6; fill: none; stroke-linecap: round; stroke-linejoin: round; }
  .cell .replay { margin-top: 8px; font-size: 13px; padding: 6px 14px; border: 1px solid #ccc; border-radius: 999px; background: #fff; cursor: pointer; }
  .cell .masks { display: grid; grid-template-columns: repeat(3, 1fr); gap: 6px; margin-top: 10px; }
  .cell .masks figure { margin: 0; text-align: center; }
  .cell .masks img { width: 100%; height: auto; display: block; border: 1px solid #e8e3d8; border-radius: 4px; background: #000; }
  .cell .masks figcaption { font-size: 11px; color: #777; margin-top: 3px; }
  @media (max-width: 900px) { .grid { grid-template-columns: repeat(2, 1fr); } }
</style>
</head>
<body>
  <h1>SketchAgent Tree Gallery</h1>
  <p class="sub">Blue = contour outline, orange = Canny edges. Each tree replays stroke-by-stroke.</p>
  <div class="grid">
"""

FOOTER = """  </div>
  <script>
  document.querySelectorAll('.cell').forEach(function(cell) {
    var groups = Array.prototype.slice.call(cell.querySelectorAll('svg g'));
    var paths = Array.prototype.slice.call(cell.querySelectorAll('svg g path'));
    var perStroke = 420;
    var schedule = [], totalDur = 0, lastTick = 0, rafId = null, elapsed = 0, playing = false;

    groups.forEach(function(g) {
      var color = g.getAttribute('stroke') || '#17130e';
      g.querySelectorAll('path').forEach(function(p) {
        var L = p.getTotalLength();
        p.dataset.len = L; p.dataset.color = color;
        p.style.strokeDasharray = L; p.style.strokeDashoffset = L;
      });
    });

    function buildSchedule() {
      schedule = [];
      var pi = 0;
      groups.forEach(function(g) {
        g.querySelectorAll('path').forEach(function(p) {
          schedule.push({el: p, delay: pi * perStroke, dur: perStroke, len: +p.dataset.len});
          pi++;
        });
      });
      totalDur = pi * perStroke;
    }

    function apply(t) {
      schedule.forEach(function(s) {
        var f = Math.max(0, Math.min(1, (t - s.delay) / s.dur));
        s.el.style.strokeDashoffset = s.len * (1 - f);
      });
      paths.forEach(function(p){ p.style.stroke = p.dataset.color; });
    }

    function play() {
      elapsed = 0; buildSchedule(); apply(0);
      playing = true; lastTick = performance.now();
      if (rafId) cancelAnimationFrame(rafId);
      rafId = requestAnimationFrame(tick);
    }

    function tick(now) {
      if (!playing) return;
      elapsed += now - lastTick; lastTick = now;
      if (elapsed >= totalDur) { elapsed = totalDur; apply(elapsed); playing = false; return; }
      apply(elapsed); rafId = requestAnimationFrame(tick);
    }

    cell.querySelector('.replay').addEventListener('click', play);
    play();
  });
  </script>
</body>
</html>
"""


def image_thumb(image_path, max_size=90):
    img = Image.open(image_path).convert("RGB")
    img.thumbnail((max_size, max_size))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode(), img.width, img.height


def build_cell(name, svg_text, thumb_b64, tw, th, rgb_b64, fol_b64, trk_b64):
    inner = svg_text.split(">", 1)[1].rsplit("</svg>", 1)[0].strip()
    x = 612 - tw - 6
    y = 6
    thumb = (
        f'<rect x="{x}" y="{y}" width="{tw}" height="{th}" fill="white" stroke="#c8c3b8" stroke-width="2"/>'
        f'<image href="data:image/png;base64,{thumb_b64}" x="{x}" y="{y}" width="{tw}" height="{th}"/>'
    )
    masks = (
        '<div class="masks">'
        f'<figure><img src="data:image/png;base64,{rgb_b64}" alt="input"><figcaption>input</figcaption></figure>'
        f'<figure><img src="data:image/png;base64,{fol_b64}" alt="foliage mask"><figcaption>foliage</figcaption></figure>'
        f'<figure><img src="data:image/png;base64,{trk_b64}" alt="trunk mask"><figcaption>trunk</figcaption></figure>'
        '</div>'
    )
    return (
        f'    <div class="cell">\n      <div class="label">{name}</div>\n'
        f'      <svg viewBox="0 0 612 612">{inner}{thumb}</svg>\n'
        f'      {masks}\n'
        f'      <button class="replay">Replay</button>\n    </div>'
    )


def main():
    threshold = 3.0
    processor = sam3_segment.build_processor()  # build SAM3 once
    images = sorted([f for f in os.listdir("input") if f.lower().endswith((".jpg", ".jpeg", ".png"))])
    cells = []
    for f in images:
        image = Image.open(f"input/{f}").convert("RGB")
        tree_mask, foliage_mask, trunk_mask = sam3_tree_masks(image, processor)
        svg, n = run_pipeline(
            f"input/{f}", threshold,
            tree_mask=tree_mask, foliage_mask=foliage_mask, trunk_mask=trunk_mask,
            workdir="results/tmp",
        )
        thumb_b64, tw, th = image_thumb(f"input/{f}")                     # small corner thumbnail
        rgb_b64, _, _ = image_thumb(f"input/{f}", max_size=140)           # strip reference image
        fol_b64, _, _ = image_thumb("results/tmp/foliage_mask.png", max_size=140)
        trk_b64, _, _ = image_thumb("results/tmp/trunk_mask.png", max_size=140)
        cells.append(build_cell(f, svg, thumb_b64, tw, th, rgb_b64, fol_b64, trk_b64))
        print(f"{f}: {n} strokes")

    html = HEADER + "\n".join(cells) + "\n" + FOOTER
    out = "results/masked/gallery.html"
    os.makedirs("results/masked", exist_ok=True)
    with open(out, "w") as fh:
        fh.write(html)
    print(f"gallery -> {out} ({len(images)} trees)")


if __name__ == "__main__":
    main()
