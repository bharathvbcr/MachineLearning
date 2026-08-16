#!/usr/bin/env python
"""Build a single self-contained HTML view of the learning-notes.

Concatenates every NN-*.md (in numeric order), rewrites inter-file `.md` links to
in-page anchors, and renders client-side with marked.js + mermaid.js. If the libs are
vendored under ./vendor/ they are INLINED for a fully offline single file; otherwise the
build falls back to CDN <script src> tags. ASCII charts are fenced code blocks (monospace
<pre>, always offline); Mermaid blocks render to SVG. Adds a sidebar TOC with scroll-spy,
clickable heading anchors, and a print stylesheet (Ctrl/Cmd+P -> PDF).

Run:  python build_html.py   ->   writes ml-notes.html next to this script.
"""
import html
import json
import pathlib
import re

HERE = pathlib.Path(__file__).parent
OUT = HERE / "ml-notes.html"

# Collect NN-name.md files in numeric order (00..25), skip this script's output.
files = sorted(
    [p for p in HERE.glob("*.md") if re.match(r"^\d\d-", p.name)],
    key=lambda p: int(p.name[:2]),
)


def anchor(name: str) -> str:
    return "file-" + name[:-3] if name.endswith(".md") else "file-" + name


# Build TOC entries (number, title from first H1, anchor) and stitched markdown.
toc, chunks = [], []
for p in files:
    text = p.read_text(encoding="utf-8")
    m = re.search(r"^#\s+(.*)$", text, re.MULTILINE)
    title = m.group(1).strip() if m else p.stem
    a = anchor(p.name)
    toc.append((p.name[:2], title, a))
    # Rewrite links like (04-sequence-mixers.md) -> (#file-04-sequence-mixers)
    text = re.sub(r"\((\d\d-[a-z0-9-]+)\.md\)", r"(#file-\1)", text)
    text = re.sub(r"\(00-README\.md\)", r"(#file-00-README)", text)
    chunks.append(f'\n\n<div class="note" id="{a}"></div>\n\n' + text)

stitched = "\n\n<hr class='filebreak'>\n\n".join(chunks)

# Encode the markdown as a JS string literal. json.dumps handles quotes/newlines/unicode;
# replacing </ with <\/ stops the HTML parser from ever seeing a </script> end-tag
# (the backslash is invisible inside a JS string). This preserves raw HTML (the anchor
# <div>s) so marked passes them through instead of showing &lt;div&gt; literals.
md_payload = json.dumps(stitched).replace("</", "<\\/")

toc_html = "\n".join(
    f'<li><a href="#{a}" data-target="{a}"><span class="n">{num}</span> {html.escape(t)}</a></li>'
    for num, t, a in toc
)

# Inline the vendored libs for a truly offline single file; fall back to CDN if absent.
# (Verified neither minified lib contains a literal </script, so direct inlining is safe.)
VENDOR = HERE / "vendor"
libs, offline = [], True
for fname, cdn in [
    ("marked.min.js", "https://cdn.jsdelivr.net/npm/marked/marked.min.js"),
    ("mermaid.min.js", "https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"),
]:
    vp = VENDOR / fname
    if vp.exists():
        libs.append("<script>\n" + vp.read_text(encoding="utf-8") + "\n</script>")
    else:
        offline = False
        libs.append(f'<script src="{cdn}"></script>')
libs_html = "\n".join(libs)
mode_note = ("Self-contained · works fully offline" if offline
             else "ASCII charts show offline · Mermaid needs internet (CDN)")

# Plain template (NOT an f-string) so JS backslashes (\n, \s, \d) and CSS braces stay
# literal. Two sentinels are filled by str.replace afterwards (replacement is literal —
# no backslash reprocessing, so the json-escaped payload survives intact).
TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ML From Scratch — Parameter Golf Learning Notes</title>
__LIBS__
<style>
  :root {
    --bg:#0f1117; --panel:#161922; --fg:#d7dce5; --muted:#8b93a7;
    --accent:#e3b341; --link:#6cb6ff; --border:#262b38; --code:#1b1f2a;
  }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--fg);
    font:16px/1.65 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif; }
  #layout { display:flex; }
  #sidebar { width:320px; min-width:320px; height:100vh; position:sticky; top:0;
    overflow-y:auto; background:var(--panel); border-right:1px solid var(--border);
    padding:18px 14px; }
  #sidebar h2 { font-size:13px; letter-spacing:.08em; text-transform:uppercase;
    color:var(--muted); margin:18px 6px 8px; }
  #sidebar ul { list-style:none; margin:0; padding:0; }
  #sidebar li a { display:block; padding:5px 8px; color:var(--fg);
    text-decoration:none; border-radius:6px; font-size:14px; }
  #sidebar li a:hover { background:#1f2430; }
  #sidebar li a.active { background:#23304a; color:#fff; }
  #sidebar li a.active .n { color:#fff; }
  #sidebar .n { display:inline-block; width:26px; color:var(--accent);
    font-variant-numeric:tabular-nums; font-weight:600; }
  .hanchor { opacity:0; text-decoration:none; color:var(--muted); margin-left:.4em;
    font-weight:400; font-size:.7em; }
  h1:hover .hanchor, h2:hover .hanchor, h3:hover .hanchor { opacity:1; }
  .searchbox { padding:0 6px 6px; }
  #search { width:100%; box-sizing:border-box; padding:7px 9px; font-size:13.5px;
    background:#0f1117; color:var(--fg); border:1px solid var(--border); border-radius:7px; }
  #search:focus { outline:none; border-color:var(--accent); }
  .searchnav { display:flex; align-items:center; gap:6px; margin-top:5px;
    font-size:12px; color:var(--muted); min-height:18px; }
  .searchnav span { flex:1; }
  .searchnav button { background:#1f2430; color:var(--fg); border:1px solid var(--border);
    border-radius:5px; cursor:pointer; font-size:10px; padding:2px 7px; }
  .searchnav button:hover { background:#2a3142; }
  mark.hit { background:#5a4a12; color:#fff; border-radius:2px; }
  mark.hit.current { background:var(--accent); color:#000; }
  #content { flex:1; max-width:900px; margin:0 auto; padding:34px 40px 120px; }
  #content h1 { font-size:30px; border-bottom:2px solid var(--border);
    padding-bottom:.3em; margin-top:.2em; }
  #content h2 { font-size:23px; margin-top:1.7em; border-bottom:1px solid var(--border);
    padding-bottom:.2em; }
  #content h3 { font-size:18px; margin-top:1.4em; color:#eef1f6; }
  a { color:var(--link); }
  hr.filebreak { border:0; border-top:1px dashed var(--border); margin:60px 0; }
  table { border-collapse:collapse; width:100%; margin:1em 0; font-size:14.5px; }
  th,td { border:1px solid var(--border); padding:7px 10px; text-align:left; }
  th { background:#1f2430; }
  tr:nth-child(even) td { background:#13161f; }
  code { background:var(--code); padding:.12em .4em; border-radius:4px;
    font-family:"JetBrains Mono",Consolas,Menlo,monospace; font-size:.88em; }
  pre { background:var(--code); border:1px solid var(--border); border-radius:8px;
    padding:14px 16px; overflow-x:auto; line-height:1.4; }
  pre code { background:none; padding:0; font-size:13px; white-space:pre; }
  blockquote { border-left:4px solid var(--accent); margin:1em 0; padding:.2em 1em;
    background:#15171f; color:#c8cedb; }
  .mermaid { background:#0c0e14; border:1px solid var(--border); border-radius:8px;
    padding:14px; margin:1em 0; text-align:center; }
  .topbar { position:sticky; top:0; background:rgba(15,17,23,.92);
    backdrop-filter:blur(6px); padding:10px 0 8px; margin-bottom:10px;
    border-bottom:1px solid var(--border); font-size:13px; color:var(--muted); }
  @media (max-width:820px) { #sidebar { display:none; } #content { padding:20px; } }
  @media print {
    #sidebar, .topbar { display:none !important; }
    #content { max-width:none; padding:0; }
    body { background:#fff; color:#000; }
    pre, code, th, blockquote, .mermaid { background:#f4f4f4 !important;
      color:#000 !important; border-color:#ccc !important; }
    a { color:#000; text-decoration:none; }
    h1, h2 { page-break-after:avoid; } pre, table, .mermaid { page-break-inside:avoid; }
    hr.filebreak { page-break-after:always; border:0; }
  }
</style>
</head>
<body>
<div id="layout">
  <nav id="sidebar">
    <h2>ML From Scratch</h2>
    <div style="font-size:12.5px;color:var(--muted);padding:0 6px 6px">
      Parameter Golf learning notes · grounded in real runs</div>
    <div class="searchbox">
      <input id="search" type="search" placeholder="Search notes…  ( / )" autocomplete="off" spellcheck="false">
      <div class="searchnav"><span id="searchcount"></span>
        <button id="prevhit" title="Previous (Shift+Enter)">&#9650;</button>
        <button id="nexthit" title="Next (Enter)">&#9660;</button></div>
    </div>
    <ul>__TOC__</ul>
  </nav>
  <main id="content">
    <div class="topbar">__MODE__ · press <code>Ctrl/Cmd+P</code> to save as PDF</div>
    <div id="rendered">Rendering…</div>
  </main>
</div>

<script>
  const md = __MD__;
  mermaid.initialize({ startOnLoad:false, theme:"dark", securityLevel:"loose" });
  marked.setOptions({ gfm:true, breaks:false });
  // Replace ```mermaid fences with raw <div class="mermaid"> BEFORE marked runs — robust
  // across marked versions (the renderer.code API changed and broke the lang check).
  // Use a unique placeholder so marked never parses the diagram source, then swap back.
  const blocks = [];
  const pre = md.replace(/```mermaid\n([\s\S]*?)```/g, (m, body) => {
    blocks.push(body);
    return "\n\nMERMAIDBLOCK" + (blocks.length - 1) + "ENDMERMAID\n\n";
  });
  let out = marked.parse(pre);
  out = out.replace(/MERMAIDBLOCK(\d+)ENDMERMAID/g,
    (m, i) => '<div class="mermaid">' + blocks[+i] + '</div>');
  document.getElementById("rendered").innerHTML = out;
  mermaid.run({ querySelector:".mermaid" }).catch((e)=>console.warn("mermaid:", e));

  // Clickable ¶ anchors on every heading (id derived from text; unique-ified).
  const seen = {};
  document.querySelectorAll("#content h1, #content h2, #content h3").forEach(h => {
    let id = h.textContent.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
    if (seen[id]) id += "-" + (++seen[id]); else seen[id] = 1;
    h.id = id;
    const a = document.createElement("a");
    a.className = "hanchor"; a.href = "#" + id; a.textContent = "¶";
    h.appendChild(a);
  });

  // TOC scroll-spy: the active note is the LAST one whose top has scrolled above ~140px.
  // (A scroll handler is robust where a thin IntersectionObserver band is not: it stays
  //  correct even when a section is far taller than the viewport.)
  const links = [...document.querySelectorAll("#sidebar a[data-target]")];
  const byId = Object.fromEntries(links.map(a => [a.dataset.target, a]));
  const notes = [...document.querySelectorAll(".note")];
  let ticking = false;
  function updateSpy() {
    ticking = false;
    let current = notes[0];
    for (const n of notes) {
      if (n.getBoundingClientRect().top <= 140) current = n; else break;
    }
    links.forEach(l => l.classList.remove("active"));
    const a = current && byId[current.id];
    if (a) {
      a.classList.add("active");
      a.scrollIntoView({ block: "nearest" });
    }
  }
  window.addEventListener("scroll", () => {
    if (!ticking) { ticking = true; requestAnimationFrame(updateSpy); }
  }, { passive: true });
  updateSpy();

  // ---- Client-side full-text search: highlight matches, navigate with Enter / arrows ----
  const content = document.getElementById("rendered");
  const input = document.getElementById("search");
  const countEl = document.getElementById("searchcount");
  let hits = [], curHit = -1, debounce;

  function clearHits() {
    // Unwrap every <mark class="hit"> back to plain text, then merge split text nodes.
    document.querySelectorAll("mark.hit").forEach(m => {
      const t = document.createTextNode(m.textContent);
      m.replaceWith(t);
    });
    content.normalize();
    hits = []; curHit = -1; countEl.textContent = "";
  }

  function runSearch(q) {
    clearHits();
    if (q.length < 2) return;
    const needle = q.toLowerCase();
    // Walk visible text nodes only; skip SVG (mermaid), pre/code stay searchable.
    const walker = document.createTreeWalker(content, NodeFilter.SHOW_TEXT, {
      acceptNode(n) {
        if (!n.nodeValue.trim()) return NodeFilter.FILTER_REJECT;
        const p = n.parentNode.nodeName;
        if (p === "SCRIPT" || p === "STYLE") return NodeFilter.FILTER_REJECT;
        if (n.parentElement.closest("svg")) return NodeFilter.FILTER_REJECT;
        return n.nodeValue.toLowerCase().includes(needle)
          ? NodeFilter.FILTER_ACCEPT : NodeFilter.FILTER_REJECT;
      }
    });
    const targets = [];
    let node;
    while ((node = walker.nextNode())) targets.push(node);
    for (const t of targets) {
      const text = t.nodeValue, low = text.toLowerCase(), frag = document.createDocumentFragment();
      let i = 0, idx;
      while ((idx = low.indexOf(needle, i)) !== -1) {
        if (idx > i) frag.appendChild(document.createTextNode(text.slice(i, idx)));
        const mk = document.createElement("mark");
        mk.className = "hit";
        mk.textContent = text.slice(idx, idx + needle.length);
        frag.appendChild(mk); hits.push(mk);
        i = idx + needle.length;
      }
      if (i < text.length) frag.appendChild(document.createTextNode(text.slice(i)));
      t.parentNode.replaceChild(frag, t);
    }
    if (hits.length) gotoHit(0);
    else countEl.textContent = "no matches";
  }

  function gotoHit(n) {
    if (!hits.length) return;
    if (curHit >= 0 && hits[curHit]) hits[curHit].classList.remove("current");
    curHit = (n + hits.length) % hits.length;
    const m = hits[curHit];
    m.classList.add("current");
    m.scrollIntoView({ block: "center", behavior: "smooth" });
    countEl.textContent = (curHit + 1) + " / " + hits.length;
  }

  input.addEventListener("input", () => {
    clearTimeout(debounce);
    debounce = setTimeout(() => runSearch(input.value.trim()), 160);
  });
  input.addEventListener("keydown", e => {
    if (e.key === "Enter") { e.preventDefault(); gotoHit(curHit + (e.shiftKey ? -1 : 1)); }
    else if (e.key === "ArrowDown") { e.preventDefault(); gotoHit(curHit + 1); }
    else if (e.key === "ArrowUp") { e.preventDefault(); gotoHit(curHit - 1); }
    else if (e.key === "Escape") { input.value = ""; clearHits(); input.blur(); }
  });
  document.getElementById("nexthit").onclick = () => gotoHit(curHit + 1);
  document.getElementById("prevhit").onclick = () => gotoHit(curHit - 1);
  // Press "/" anywhere to focus search (unless already typing in a field).
  document.addEventListener("keydown", e => {
    if (e.key === "/" && document.activeElement !== input &&
        !/^(INPUT|TEXTAREA)$/.test(document.activeElement.nodeName)) {
      e.preventDefault(); input.focus();
    }
  });
</script>
</body>
</html>
"""

DOC = (TEMPLATE
       .replace("__LIBS__", libs_html)
       .replace("__MODE__", mode_note)
       .replace("__TOC__", toc_html)
       .replace("__MD__", md_payload))

OUT.write_text(DOC, encoding="utf-8")
print(f"Wrote {OUT}  ({len(files)} notes, {len(DOC)//1024} KB, "
      f"{'OFFLINE/self-contained' if offline else 'CDN fallback'})")
