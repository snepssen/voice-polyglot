"""Write a local listening page for the clips a native ear has to judge.

    python review_page.py DATASET_DIR [--all]

Writes DATASET_DIR/review.html next to clips/, listing every clip marked
`review` in manifest.tsv (or every clip, with --all): a player, the script
line, what Parakeet heard, and a Drop toggle. "Copy verdicts" puts the
dropped ids on the clipboard, to paste back into the conversation.
"""

import csv
import html
import json
import sys
from pathlib import Path

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Clip Review</title>
<style>
:root { --bg:#fbfaf7; --fg:#1d1c1a; --muted:#6d6a64; --line:#e4e1da; --accent:#2f6f5e; --drop:#b4442f; --card:#fff; }
@media (prefers-color-scheme: dark) { :root { --bg:#171716; --fg:#ecebe7; --muted:#9a968e; --line:#2c2b29; --accent:#6fbfa6; --drop:#e0775f; --card:#1f1f1d; } }
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--fg); font:15px/1.45 -apple-system, system-ui, sans-serif; }
header { position:sticky; top:0; background:var(--bg); border-bottom:1px solid var(--line); padding:14px 16px; display:flex; gap:12px; align-items:center; flex-wrap:wrap; z-index:1; }
h1 { font-size:17px; margin:0; font-weight:600; }
.count { color:var(--muted); font-variant-numeric:tabular-nums; }
button { font:inherit; border:1px solid var(--line); background:var(--card); color:var(--fg); border-radius:8px; padding:6px 12px; cursor:pointer; }
button.primary { background:var(--accent); border-color:var(--accent); color:#fff; margin-left:auto; }
main { max-width:860px; margin:0 auto; padding:8px 16px 80px; }
.help { color:var(--muted); font-size:13px; margin:12px 0; }
.clip { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:12px 14px; margin:10px 0; display:grid; grid-template-columns:1fr auto; gap:6px 12px; }
.clip.dropped { border-color:var(--drop); opacity:.62; }
.clip.current { outline:2px solid var(--accent); outline-offset:1px; }
.id { font:12px ui-monospace, Menlo, monospace; color:var(--muted); }
.script { font-size:16px; }
.heard { color:var(--muted); font-size:13px; }
.heard b { font-weight:500; }
audio { width:100%; height:32px; grid-column:1 / -1; }
.toggle { align-self:start; grid-row:1 / span 3; grid-column:2; }
.clip.dropped .toggle { border-color:var(--drop); color:var(--drop); }
</style></head>
<body>
<header><h1>__TITLE__</h1><span class="count" id="count"></span>
<button class="primary" id="copy">Copy verdicts</button></header>
<main>
<p class="help">Everything is kept unless you drop it. Drop a clip if a word is misread, slurred, missing, or belongs to a neighbouring line. Keys: <b>space</b> play, <b>j / k</b> next / previous, <b>d</b> drop.</p>
<div id="list"></div>
</main>
<script>
const clips = __CLIPS__;
const KEY = "clip-review-" + __TITLE_JSON__ + "-" + __VERSION__;
let dropped = new Set();
try { dropped = new Set(JSON.parse(localStorage.getItem(KEY) || "[]")); } catch (e) {}
const save = () => { try { localStorage.setItem(KEY, JSON.stringify([...dropped])); } catch (e) {} };
const list = document.getElementById("list");
let current = 0;
clips.forEach((c, i) => {
  const el = document.createElement("section");
  el.className = "clip"; el.id = c.id;
  el.innerHTML = `<div class="id">${c.id} · score ${c.score} · ${c.seconds}s</div>
    <button class="toggle"></button>
    <div class="script"></div><div class="heard">Parakeet heard: <b></b></div>
    <audio controls preload="none" src="clips/${c.id}.wav"></audio>`;
  el.querySelector(".script").textContent = c.text;
  el.querySelector(".heard b").textContent = c.heard || "(nothing)";
  el.querySelector(".toggle").onclick = () => toggle(i);
  el.querySelector("audio").onplay = () => focus(i, false);
  list.appendChild(el);
});
function render() {
  clips.forEach((c, i) => {
    const el = document.getElementById(c.id), d = dropped.has(c.id);
    el.classList.toggle("dropped", d); el.classList.toggle("current", i === current);
    el.querySelector(".toggle").textContent = d ? "Dropped" : "Drop";
  });
  document.getElementById("count").textContent = `${clips.length} clips · ${dropped.size} dropped`;
}
function toggle(i) { const id = clips[i].id; dropped.has(id) ? dropped.delete(id) : dropped.add(id); save(); render(); }
function focus(i, scroll = true) { current = Math.max(0, Math.min(clips.length - 1, i)); render(); if (scroll) document.getElementById(clips[current].id).scrollIntoView({ block: "center", behavior: "smooth" }); }
document.addEventListener("keydown", e => {
  if (e.target.tagName === "BUTTON" && e.key === " ") e.preventDefault();
  const a = document.querySelectorAll("audio")[current];
  if (e.key === "j") focus(current + 1);
  else if (e.key === "k") focus(current - 1);
  else if (e.key === "d") toggle(current);
  else if (e.key === " ") { e.preventDefault(); document.querySelectorAll("audio").forEach(x => x !== a && x.pause()); a.paused ? a.play() : a.pause(); }
});
document.getElementById("copy").onclick = async () => {
  const text = `Reviewed ${clips.length} clips (${__TITLE_JSON__}). Dropped ${dropped.size}: ${[...dropped].sort().join(", ") || "none"}`;
  try { await navigator.clipboard.writeText(text); document.getElementById("copy").textContent = "Copied"; }
  catch (e) { prompt("Copy this:", text); }
  setTimeout(() => document.getElementById("copy").textContent = "Copy verdicts", 1500);
};
render();
</script></body></html>
"""


def main(dataset: str, everything: bool = False) -> None:
    d = Path(dataset)
    rows = list(csv.DictReader(open(d / "manifest.tsv", encoding="utf-8"), delimiter="\t"))
    picked = [r for r in rows if everything or r["status"] == "review"]
    clips = [{k: r[k] for k in ("id", "text", "heard", "score", "seconds")} for r in picked]
    title = f"{d.parent.name.capitalize()} clip review"
    import hashlib
    version = hashlib.sha1("".join(c["id"] + c["text"] + str(c["seconds"]) for c in clips).encode()).hexdigest()[:8]
    page = (PAGE.replace("__TITLE__", html.escape(title))
                .replace("__VERSION__", json.dumps(version))
                .replace("__TITLE_JSON__", json.dumps(title))
                .replace("__CLIPS__", json.dumps(clips, ensure_ascii=False)))
    (d / "review.html").write_text(page, encoding="utf-8")
    print(f"{d / 'review.html'}: {len(clips)} clips")


if __name__ == "__main__":
    main(sys.argv[1], "--all" in sys.argv[2:])
