"""Generate a stroke-by-stroke animation HTML from an SVG (reuses the sketch_animation JS)."""
import argparse
import re


HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>{title}</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600&family=Inter:wght@400;500;600&display=swap">
<style>
  :root {{ --bg:#f3f0e8; --surface:#ffffff; --surface-2:#e8e3d8; --ink:#1c1a16; --muted:#6f6a5f; --accent:#2f5d8a; --paper:#ffffff; --line:#17130e; --display:"Space Grotesk",system-ui,sans-serif; --body:"Inter",system-ui,sans-serif; }}
  @media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ --bg:#151412; --surface:#201e1b; --surface-2:#2c2a26; --ink:#ece7de; --muted:#9b968a; --accent:#7fa8cc; color-scheme:dark; }} }}
  :root[data-theme="dark"] {{ --bg:#151412; --surface:#201e1b; --surface-2:#2c2a26; --ink:#ece7de; --muted:#9b968a; --accent:#7fa8cc; color-scheme:dark; }}
  * {{ box-sizing:border-box; }}
  html,body {{ height:100%; }}
  body {{ margin:0; background:var(--bg); color:var(--ink); font-family:var(--body); padding:28px max(16px,env(safe-area-inset-right,0px)) 44px max(16px,env(safe-area-inset-left,0px)); }}
  .wrap {{ max-width:680px; margin:0 auto; }}
  header {{ margin-bottom:20px; }}
  .eyebrow {{ font-size:11px; letter-spacing:.16em; text-transform:uppercase; color:var(--muted); font-weight:600; }}
  h1 {{ font-family:var(--display); font-size:clamp(24px,4vw,34px); margin:4px 0 2px; font-weight:600; }}
  .sub {{ color:var(--muted); margin:0; }}
  .stage {{ background:var(--surface); border:1px solid var(--surface-2); border-radius:14px; padding:18px; display:grid; place-items:center; }}
  .paper {{ background:var(--paper); border-radius:4px; box-shadow:0 1px 2px rgba(0,0,0,.05),0 10px 28px rgba(0,0,0,.08); width:100%; max-width:500px; aspect-ratio:1; }}
  svg {{ display:block; width:100%; height:100%; }}
  svg path {{ stroke:var(--line); stroke-width:7; fill:none; stroke-linecap:round; stroke-linejoin:round; }}
  .controls {{ margin-top:16px; display:flex; align-items:center; gap:12px; flex-wrap:wrap; }}
  #replay {{ font-family:var(--body); font-weight:600; font-size:14px; color:#fff; background:var(--accent); border:0; border-radius:999px; padding:10px 22px; cursor:pointer; }}
  #pause {{ font-family:var(--body); font-weight:600; font-size:14px; color:var(--ink); background:var(--surface); border:1px solid var(--surface-2); border-radius:999px; padding:10px 22px; cursor:pointer; }}
  .seg {{ display:inline-flex; border:1px solid var(--surface-2); border-radius:999px; overflow:hidden; background:var(--surface); }}
  .seg button {{ background:transparent; color:var(--ink); border:0; border-radius:0; padding:8px 14px; font-family:var(--body); font-size:13px; cursor:pointer; }}
  .seg button[aria-pressed="true"] {{ background:var(--surface-2); }}
  .readout {{ margin-left:auto; font-variant-numeric:tabular-nums; color:var(--muted); font-size:14px; white-space:nowrap; }}
  .readout b {{ color:var(--ink); }}
  .bar {{ height:4px; background:var(--surface-2); border-radius:999px; margin-top:12px; overflow:hidden; }}
  .bar > div {{ height:100%; width:0; background:var(--accent); border-radius:999px; }}
</style>
</head>
<body>
  <div class="wrap">
    <header>
      <div class="eyebrow">SketchAgent &middot; sequential sketch</div>
      <h1>{title}</h1>
      <p class="sub">{stroke_count} strokes. Blue = contour outline, orange = Canny edges.</p>
    </header>
    <div class="stage">
      <div class="paper">
        <svg viewBox="0 0 612 612" preserveAspectRatio="xMidYMid meet" aria-label="sketch animation">
{svg_content}
        </svg>
      </div>
    </div>
    <div class="controls">
      <button id="replay">Replay</button>
      <button id="pause">Pause</button>
      <div class="seg" role="group" aria-label="speed">
        <button data-speed="0.5" aria-pressed="false">0.5&times;</button>
        <button data-speed="1" aria-pressed="true">1&times;</button>
        <button data-speed="2" aria-pressed="false">2&times;</button>
      </div>
      <div class="readout">stroke <b id="cur">0</b> / <b id="total">0</b></div>
    </div>
    <div class="bar"><div id="prog"></div></div>
  </div>
  <script>
    var groups = Array.prototype.slice.call(document.querySelectorAll('svg g'));
    var paths = Array.prototype.slice.call(document.querySelectorAll('svg g path'));
    var perStroke = 420;
    var speed = 1;
    var playing = false, paused = false;
    var schedule = [];
    var elapsed = 0, totalDur = 0, lastTick = 0, rafId = null;
    var curEl = document.getElementById('cur');
    var progEl = document.getElementById('prog');
    var pauseBtn = document.getElementById('pause');
    document.getElementById('total').textContent = groups.length;
    groups.forEach(function(g) {{
      var color = g.getAttribute('stroke') || 'var(--line)';
      g.querySelectorAll('path').forEach(function(p) {{
        var L = p.getTotalLength();
        p.dataset.len = L;
        p.dataset.color = color;
        p.style.strokeDasharray = L;
        p.style.strokeDashoffset = L;
      }});
    }});
    function buildSchedule() {{
      schedule = [];
      groups.forEach(function(g, gi) {{
        var gp = Array.prototype.slice.call(g.querySelectorAll('path'));
        var delay = gi * perStroke / speed;
        gp.forEach(function(p) {{
          schedule.push({{ el: p, delay: delay, dur: perStroke / speed, len: +p.dataset.len }});
        }});
      }});
      totalDur = groups.length * perStroke / speed;
    }}
    function apply(t) {{
      schedule.forEach(function(s) {{
        var f = Math.max(0, Math.min(1, (t - s.delay) / s.dur));
        s.el.style.strokeDashoffset = s.len * (1 - f);
      }});
      paths.forEach(function(p){{ p.style.stroke = p.dataset.color; }});
      var idx = Math.floor(t / (perStroke / speed));
      curEl.textContent = String(Math.min(groups.length, idx + 1));
      progEl.style.width = (Math.min(1, t / totalDur) * 100).toFixed(1) + '%';
    }}
    function play() {{
      elapsed = 0; buildSchedule(); apply(0);
      playing = true; paused = false; pauseBtn.textContent = 'Pause';
      lastTick = performance.now(); rafId = requestAnimationFrame(tick);
    }}
    function tick(now) {{
      if (!playing) return;
      elapsed += now - lastTick; lastTick = now;
      if (elapsed >= totalDur) {{ elapsed = totalDur; apply(elapsed); playing = false; paused = false; pauseBtn.textContent = 'Pause'; return; }}
      apply(elapsed); rafId = requestAnimationFrame(tick);
    }}
    function togglePause() {{
      if (paused) {{ paused = false; playing = true; lastTick = performance.now(); pauseBtn.textContent = 'Pause'; rafId = requestAnimationFrame(tick); }}
      else if (playing) {{ paused = true; playing = false; if (rafId) cancelAnimationFrame(rafId); pauseBtn.textContent = 'Resume'; }}
    }}
    document.getElementById('replay').addEventListener('click', play);
    pauseBtn.addEventListener('click', togglePause);
    Array.prototype.slice.call(document.querySelectorAll('.seg button')).forEach(function(b) {{
      b.addEventListener('click', function() {{
        speed = +b.dataset.speed;
        document.querySelectorAll('.seg button').forEach(function(x){{ x.setAttribute('aria-pressed', String(x === b)); }});
        play();
      }});
    }});
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {{ buildSchedule(); apply(totalDur); pauseBtn.disabled = true; }}
    else {{ play(); }}
  </script>
</body>
</html>
"""


def make_animation(svg_path, out_path, title="Sketch"):
    svg = open(svg_path).read()
    inner = svg.split(">", 1)[1].rsplit("</svg>", 1)[0].strip()
    n = len(re.findall(r'<g id="s\d+"', inner))
    html = HTML.format(title=title, stroke_count=n, svg_content=inner)
    with open(out_path, "w") as f:
        f.write(html)
    return out_path, n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--svg", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--title", default="Tree, stroke by stroke")
    args = p.parse_args()

    svg = open(args.svg).read()
    inner = svg.split(">", 1)[1].rsplit("</svg>", 1)[0].strip()
    n = len(re.findall(r'<g id="s\d+"', inner))
    html = HTML.format(title=args.title, stroke_count=n, svg_content=inner)
    with open(args.out, "w") as f:
        f.write(html)
    print(f"生成 {args.out} ({n} 笔)")


if __name__ == "__main__":
    main()
