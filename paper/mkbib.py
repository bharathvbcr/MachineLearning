#!/usr/bin/env python3
"""Generate paper/refs.tex: a thebibliography preserving the Markdown numbering.

Not BibTeX, because the repository publishes both the Markdown paper and the
PDF and BibTeX renumbers. Under `plain` the entry the Markdown calls [12] prints
as [34]; under `unsrt` it takes citation order, which is also not the Markdown
order (the first entry cited is [6]). Either way a reader moving between the two
forms finds the same bracket number pointing at different papers -- the
cross-artifact drift section 7.4 is about.

Emitting \\bibitem{refN} in order 1..N makes LaTeX's numbering agree with the
Markdown by construction.

Usage: python3 mkbib.py ../PAPER_2026-08_Recipe_Dependent_Rankings.md > refs.tex
"""
from __future__ import annotations
import re, sys
from pathlib import Path

# Backslash first: everything after it introduces one.
TEX_ESCAPES = (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"),
               ("$", r"\$"), ("#", r"\#"), ("_", r"\_"), ("{", r"\{"),
               ("}", r"\}"), ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}"))

# Author names and titles carry accented letters and curly quotes. TeX drops a
# glyph it has no font entry for -- silently -- so every non-ASCII character must
# be mapped or the reference prints with letters missing.
UNICODE_TO_TEX = (
    ("\u2019", "'"), ("\u2018", "`"), ("\u201c", "``"), ("\u201d", "''"),
    ("\u2013", "--"), ("\u2014", "---"), ("\u00a0", "~"),
    ("\u010d", "\\v{c}"), ("\u010c", "\\v{C}"),
    ("\u0161", "\\v{s}"), ("\u0160", "\\v{S}"),
    ("\u017e", "\\v{z}"), ("\u017d", "\\v{Z}"),
    ("\u00e9", "\\'{e}"), ("\u00e8", "\\`{e}"), ("\u00fc", '\\"{u}'),
    ("\u00f6", '\\"{o}'), ("\u00e4", '\\"{a}'), ("\u00ed", "\\'{i}"),
    ("\u00e1", "\\'{a}"), ("\u00f3", "\\'{o}"), ("\u00fa", "\\'{u}"),
    ("\u00f1", "\\~{n}"), ("\u00e7", "\\c{c}"), ("\u0142", "\\l{}"),
    ("\u00b5", "$\\mu$"), ("\u03bc", "$\\mu$"),
)


def escape(t: str) -> str:
    for ch, rep in TEX_ESCAPES:
        t = t.replace(ch, rep)
    for ch, rep in UNICODE_TO_TEX:
        t = t.replace(ch, rep)
    return t

def main() -> int:
    if len(sys.argv) != 2:
        sys.exit("usage: mkbib.py <paper.md>")
    md = Path(sys.argv[1]).read_text(encoding="utf-8")
    start = re.search(r"^## References\s*$", md, re.M)
    if not start:
        sys.exit("no '## References' section found")
    section = md[start.end():]
    end = re.search(r"^### ", section, re.M)
    if end:
        section = section[:end.start()]
    entries = re.findall(r"^\[(\d+)\]\s+(.+?)\s*$", section, re.M)
    if not entries:
        sys.exit("no references parsed")
    nums = [int(n) for n, _ in entries]
    if nums != list(range(1, len(nums) + 1)):
        sys.exit(f"references are not numbered 1..N without gaps: {nums[:6]}...")
    out = [r"\begin{thebibliography}{%d}" % len(entries), r"\setlength{\itemsep}{2pt}"]
    for num, body in entries:
        b = escape(body)
        b = re.sub(r'"(.+?)"', r"``\1''", b)
        b = re.sub(r"(arXiv:\d{4}\.\d{4,5})", r"\\emph{\1}", b)
        out.append(r"\bibitem{ref%s} %s" % (num, b))
    out.append(r"\end{thebibliography}")
    text = "\n".join(out)
    leftover = sorted({c for c in text if ord(c) > 127})
    if leftover:
        sys.exit("mkbib: unmapped non-ASCII would be silently dropped by TeX: "
                 + str([f"U+{ord(c):04X} {c!r}" for c in leftover]))
    print(text)
    print(f"[mkbib] {len(entries)} references, numbering preserved 1..{len(entries)}",
          file=sys.stderr)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
