"""A listening page for choosing an epoch by ear.

    python listen_page.py [CHECKIN_DIR]

Writes CHECKIN_DIR/listen.html (default work/runs/checkin-v2): for every
language, the check-in sentence rendered by every checked-in epoch and by the
voice before training, one click each. Blind mode shuffles the order and
hides the epoch and the Parakeet score, so the choice is made on sound alone.
Mark one render per language as best and any number as bad; "Copy verdicts"
puts the choices on the clipboard to paste back.

Judging rendered quality needs no fluency: clicks, metallic or buzzy timbre,
mumbled or swallowed syllables, wobbling pitch, a stray sound after the last
word, and pacing that rushes or drags are all audible without knowing the
language.

Epochs without an export (milestone_watch exports every 25th) are marked:
heard, but not choosable as the final voice.

Serve the folder over http (the page loads epoch-NNN/<lang>.wav beside it).
The page is static; regenerate it to add epochs that arrive later.
"""

import html
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from checkin import SENTENCES  # noqa: E402

NAMES = {"en": "English", "pl": "Polish", "hu": "Hungarian", "de": "German", "nl": "Dutch",
         "fr": "French", "es": "Spanish", "it": "Italian", "en-short": "English, two words"}

PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Epoch Listening</title>
<style>
:root { --bg:#fbfaf7; --fg:#1d1c1a; --muted:#6d6a64; --line:#e4e1da; --card:#fff; --accent:#2f6f5e; --accent-fg:#fff; --best:#2f6f5e; --bad:#b4442f; --chip:#f2f0ea; }
@media (prefers-color-scheme: dark) { :root { --bg:#171716; --fg:#ecebe7; --muted:#9a968e; --line:#2c2b29; --card:#1f1f1d; --accent:#6fbfa6; --accent-fg:#10201b; --best:#6fbfa6; --bad:#e0775f; --chip:#272725; } }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--fg); font:15px/1.45 -apple-system, system-ui, sans-serif; }
header { position:sticky; top:0; z-index:2; background:var(--bg); border-bottom:1px solid var(--line); padding:12px 16px; display:flex; gap:10px; align-items:center; flex-wrap:wrap; }
h1 { font-size:17px; margin:0 8px 0 0; font-weight:600; }
.status { color:var(--muted); font-size:13px; font-variant-numeric:tabular-nums; }
button { font:inherit; border:1px solid var(--line); background:var(--card); color:var(--fg); border-radius:8px; padding:6px 12px; cursor:pointer; }
button.on { background:var(--accent); color:var(--accent-fg); border-color:var(--accent); }
.spacer { flex:1; }
main { max-width:980px; margin:0 auto; padding:8px 16px 80px; }
.help { color:var(--muted); font-size:13px; margin:12px 0 4px; }
.help b { color:var(--fg); font-weight:600; }
section { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:12px 14px; margin:12px 0; }
section.current { outline:2px solid var(--accent); outline-offset:1px; }
.lang { display:flex; align-items:baseline; gap:10px; }
.lang h2 { font-size:15px; margin:0; font-weight:600; }
.sentence { color:var(--muted); font-size:13px; margin:2px 0 10px; }
.row { display:flex; flex-wrap:wrap; gap:6px; }
.cell { position:relative; display:flex; flex-direction:column; align-items:center; min-width:62px; padding:6px 8px; border-radius:8px; border:1px solid var(--line); background:var(--chip); cursor:pointer; user-select:none; }
.cell .label { font-size:13px; font-weight:600; font-variant-numeric:tabular-nums; }
.cell .score { font-size:11px; color:var(--muted); font-variant-numeric:tabular-nums; }
.cell.playing { border-color:var(--accent); box-shadow:0 0 0 2px var(--accent) inset; }
.cell.focus { outline:2px dashed var(--muted); outline-offset:1px; }
.cell.best { border-color:var(--best); background:color-mix(in srgb, var(--best) 18%, var(--chip)); }
.cell.bad { border-color:var(--bad); opacity:.55; text-decoration:line-through; }
.cell.gone .label::after { content:" ·"; color:var(--muted); }
.cell .mark { font-size:10px; letter-spacing:.04em; text-transform:uppercase; color:var(--best); }
.cell.bad .mark { color:var(--bad); }
.blind .score { visibility:hidden; }
.verdicts { white-space:pre-wrap; font:12px ui-monospace, Menlo, monospace; color:var(--muted); margin-top:16px; }
</style></head>
<body>
<header>
  <h1>Epoch listening</h1>
  <span class="status" id="status"></span>
  <span class="spacer"></span>
  <button id="blind">Blind</button>
  <button id="copy">Copy verdicts</button>
</header>
<main>
<p class="help">Click a render to play it; click again to replay. <b>←/→</b> step through renders and play, <b>↑/↓</b> change language, <b>space</b> replays, <b>b</b> marks best (one per language), <b>x</b> marks bad. Listen for clicks, buzz or metallic tone, swallowed syllables, wobbling pitch, a stray sound after the last word, and rushed or dragging pace — none of that needs the language. Blind mode shuffles each row and hides labels until you turn it off. A <b>·</b> after an epoch means it has no exported voice: it can be heard but not chosen. Every 25th epoch is exported.</p>
<div id="list"></div>
<div class="verdicts" id="verdicts"></div>
</main>
<audio id="player" preload="auto"></audio>
<script>
const DATA = __DATA__;
const KEY = "epoch-listening-v1";
let state = { best: {}, bad: {}, blind: false };
try { Object.assign(state, JSON.parse(localStorage.getItem(KEY) || "{}")); } catch (e) {}
const save = () => { try { localStorage.setItem(KEY, JSON.stringify(state)); } catch (e) {} };
const player = document.getElementById("player");
let cur = { lang: 0, cell: 0 }, playing = null;
const order = {};

function shuffled(n, seed) {
  const a = [...Array(n).keys()]; let s = seed;
  for (let i = n - 1; i > 0; i--) { s = (s * 9301 + 49297) % 233280; const j = Math.floor(s / 233280 * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; }
  return a;
}
function cellsOf(li) {
  const L = DATA.langs[li];
  const idx = state.blind ? (order[L.key] ||= shuffled(DATA.epochs.length, 7 + li * 31)) : [...DATA.epochs.keys()];
  return idx.filter(i => L.scores[i] !== null);
}
function render() {
  document.body.classList.toggle("blind", state.blind);
  document.getElementById("blind").classList.toggle("on", state.blind);
  const list = document.getElementById("list"); list.innerHTML = "";
  DATA.langs.forEach((L, li) => {
    const s = document.createElement("section");
    if (li === cur.lang) s.classList.add("current");
    s.innerHTML = `<div class="lang"><h2></h2></div><div class="sentence"></div><div class="row"></div>`;
    s.querySelector("h2").textContent = L.name;
    s.querySelector(".sentence").textContent = L.text;
    const row = s.querySelector(".row");
    cellsOf(li).forEach((ei, ci) => {
      const id = `${DATA.epochs[ei]}/${L.key}`;
      const c = document.createElement("div");
      c.className = "cell";
      if (!DATA.kept[ei]) c.classList.add("gone");
      if (state.best[L.key] === DATA.epochs[ei]) c.classList.add("best");
      if ((state.bad[L.key] || []).includes(DATA.epochs[ei])) c.classList.add("bad");
      if (playing === id) c.classList.add("playing");
      if (li === cur.lang && ci === cur.cell) c.classList.add("focus");
      const label = state.blind ? String.fromCharCode(65 + ci) : DATA.labels[ei];
      const mark = c.classList.contains("best") ? "best" : c.classList.contains("bad") ? "bad" : "";
      c.innerHTML = `<span class="label">${label}</span><span class="score">${L.scores[ei]}%</span><span class="mark">${mark}</span>`;
      c.onclick = () => { cur = { lang: li, cell: ci }; play(); };
      row.appendChild(c);
    });
    list.appendChild(s);
  });
  const picked = Object.keys(state.best).length;
  document.getElementById("status").textContent = `${DATA.epochs.length} renders × ${DATA.langs.length} sentences · best picked for ${picked}/${DATA.langs.length}`;
  document.getElementById("verdicts").textContent = verdicts();
}
function current() { const L = DATA.langs[cur.lang]; const ei = cellsOf(cur.lang)[cur.cell]; return { L, ep: DATA.epochs[ei] }; }
function play() {
  const { L, ep } = current(); if (!ep) return;
  playing = `${ep}/${L.key}`;
  player.src = `${ep}/${L.key}.wav`; player.currentTime = 0; player.play().catch(() => {});
  render();
  document.querySelectorAll("section")[cur.lang]?.scrollIntoView({ block: "nearest", behavior: "smooth" });
}
player.onended = () => { playing = null; render(); };
function verdicts() {
  const lines = DATA.langs.map(L => {
    const b = state.best[L.key] ? DATA.labels[DATA.epochs.indexOf(state.best[L.key])] : "-";
    const bad = (state.bad[L.key] || []).map(e => DATA.labels[DATA.epochs.indexOf(e)]).join(", ");
    return `${L.key}: best ${b}${bad ? `; bad ${bad}` : ""}`;
  });
  return "Epoch listening verdicts\n" + lines.join("\n");
}
document.addEventListener("keydown", e => {
  if (e.metaKey || e.ctrlKey || e.altKey) return;
  const n = cellsOf(cur.lang).length;
  if (e.key === "ArrowRight") { cur.cell = Math.min(n - 1, cur.cell + 1); play(); }
  else if (e.key === "ArrowLeft") { cur.cell = Math.max(0, cur.cell - 1); play(); }
  else if (e.key === "ArrowDown") { cur.lang = Math.min(DATA.langs.length - 1, cur.lang + 1); cur.cell = Math.min(cur.cell, cellsOf(cur.lang).length - 1); play(); }
  else if (e.key === "ArrowUp") { cur.lang = Math.max(0, cur.lang - 1); cur.cell = Math.min(cur.cell, cellsOf(cur.lang).length - 1); play(); }
  else if (e.key === " ") { play(); }
  else if (e.key === "b") { const { L, ep } = current(); state.best[L.key] = state.best[L.key] === ep ? undefined : ep; if (!state.best[L.key]) delete state.best[L.key]; save(); render(); }
  else if (e.key === "x") { const { L, ep } = current(); const s = new Set(state.bad[L.key] || []); s.has(ep) ? s.delete(ep) : s.add(ep); state.bad[L.key] = [...s]; save(); render(); }
  else return;
  e.preventDefault();
});
document.getElementById("blind").onclick = () => { state.blind = !state.blind; cur.cell = 0; save(); render(); };
document.getElementById("copy").onclick = async () => {
  const t = verdicts();
  try { await navigator.clipboard.writeText(t); document.getElementById("copy").textContent = "Copied"; }
  catch (e) { prompt("Copy this:", t); }
  setTimeout(() => document.getElementById("copy").textContent = "Copy verdicts", 1500);
};
render();
</script>
</body></html>
"""


def main(checkin_dir: str = "work/runs/checkin-v2") -> None:
    d = Path(checkin_dir)
    epochs = sorted((p.name for p in d.glob("epoch-*") if p.is_dir()),
                    key=lambda n: -1 if "before" in n else int(re.search(r"\d+", n).group()))
    labels = ["before" if "before" in e else f"ep {int(re.search(r'\d+', e).group())}" for e in epochs]
    scores: dict[str, dict[str, str]] = {}
    for line in (d / "scores.txt").read_text().splitlines() if (d / "scores.txt").exists() else []:
        parts = line.split()
        if parts:
            scores[parts[0]] = dict(re.findall(r"(\S+) (\d+)%", line))
    langs = []
    for key, text in SENTENCES.items():
        row = []
        for e in epochs:
            if not (d / e / f"{key}.wav").exists():
                row.append(None)
            else:
                row.append(scores.get(e, {}).get(key, "–"))
        langs.append({"key": key, "name": NAMES.get(key, key), "text": text, "scores": row})
    exports = d.parent / "polyglot" / "exports"
    kept = [("before" in e) or any(exports.glob(f"{e}*")) for e in epochs]
    data = {"epochs": epochs, "labels": labels, "langs": langs, "kept": kept}
    (d / "listen.html").write_text(PAGE.replace("__DATA__", json.dumps(data, ensure_ascii=False)), encoding="utf-8")
    print(f"{d / 'listen.html'}: {len(epochs)} renders x {len(langs)} sentences")


if __name__ == "__main__":
    main(*sys.argv[1:2])
