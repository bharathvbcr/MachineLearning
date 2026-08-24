#!/usr/bin/env python3
"""Convert the Markdown paper to a LaTeX body for paper/main.tex.

The Markdown file is the source of truth -- every number in it is checked
against the run records by derive_figures.py --check -- so this script must not
alter content. It does only the mechanical work:

  * strips the title/author/license block, which main.tex owns;
  * strips the References section, which refs.tex owns;
  * rewrites bracketed numeric citations [12] into \\cite{ref12};
  * maps typographic characters the Computer Modern fonts lack;
  * removes pandoc's caption-less-longtable wrapper.

Usage: python3 md2tex.py ../PAPER_2026-08_Recipe_Dependent_Rankings.md > body.tex
"""
from __future__ import annotations
import re, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Applied to pandoc's LaTeX OUTPUT, never to the Markdown input. Injecting $...$
# into Markdown makes pandoc re-escape it into \$...\( and corrupt the sentence.
# A missing glyph is also silently DROPPED by TeX, which would delete the minus
# sign from a negative result, so every character the fonts lack is mapped.
UNICODE_TO_TEX = (
    ("\u2212", "-"), ("\u00d7", "$\\times$"), ("\u2248", "$\\approx$"),
    ("\u2265", "$\\geq$"), ("\u2264", "$\\leq$"), ("\u00b5", "$\\mu$"),
    ("\u03bc", "$\\mu$"), ("\u00a7", "\\S"), ("\u2190", "$\\leftarrow$"),
    ("\u2192", "$\\rightarrow$"), ("\u00b1", "$\\pm$"),
    ("\u2081", "$_1$"), ("\u2082", "$_2$"), ("\u2083", "$_3$"),
    ("\u2084", "$_4$"), ("\u2085", "$_5$"),
    ("\u0394", "$\\Delta$"), ("\u03b4", "$\\delta$"), ("\u00b7", "$\\cdot$"),
    ("\u2026", "\\ldots"), ("\u221e", "$\\infty$"), ("\u221a", "$\\surd$"), ("\u2194", "$\\leftrightarrow$"),
)

def known_refs() -> set[str]:
    bib = HERE / "refs.bib"
    if not bib.exists():
        sys.exit("refs.bib not found; generate it before converting the body")
    return set(re.findall(r"@article\{ref(\d+),", bib.read_text(encoding="utf-8")))

def strip_front_and_refs(md: str) -> str:
    m = re.search(r"^## Abstract\s*$", md, re.M)
    if not m:
        sys.exit("could not find '## Abstract'; refusing to guess where the body starts")
    md = md[m.end():]
    r = re.search(r"^## References\s*$", md, re.M)
    if r:
        rest = md[r.end():]
        tail = re.search(r"^### Provenance of this draft\s*$", rest, re.M)
        md = md[:r.start()] + (rest[tail.start():] if tail else "")
    return md

def rewrite_citations(md: str, refs: set[str]) -> tuple[str, int]:
    """[12] -> \\cite{ref12}. Only groups whose numbers ALL resolve to refs.bib,
    so table cells that look like [8] are left alone. Code blocks are skipped."""
    count = 0
    parts = re.split(r"(```.*?```)", md, flags=re.S)
    def repl(m: re.Match) -> str:
        nonlocal count
        nums = [n.strip() for n in m.group(1).split(",")]
        if not all(n in refs for n in nums):
            return m.group(0)
        count += 1
        return "\\cite{" + ",".join("ref" + n for n in nums) + "}"
    for i in range(0, len(parts), 2):
        parts[i] = re.sub(r"\[(\d+(?:\s*,\s*\d+)*)\]", repl, parts[i])
    return "".join(parts), count

def main() -> int:
    if len(sys.argv) != 2:
        sys.exit("usage: md2tex.py <paper.md>")
    md = Path(sys.argv[1]).read_text(encoding="utf-8")
    refs = known_refs()
    md = strip_front_and_refs(md)
    md, n_cites = rewrite_citations(md, refs)
    # Repo-relative links are not resolvable from a PDF; render as literal paths.
    md = re.sub(r"\[([^\]]+)\]\((?!https?://)[^)]+\)", r"\\texttt{\1}", md)

    proc = subprocess.run(
        # subscript/superscript are DISABLED: the paper writes "~238k" for
        # "approximately 238k", and pandoc's subscript extension reads ~x~ as
        # markup, which silently turned "~238k/~240k" into a subscript.
        ["pandoc", "--from",
         "markdown+pipe_tables+backtick_code_blocks+raw_tex-subscript-superscript",
         "--to", "latex", "--top-level-division=section",
         "--shift-heading-level-by=-1", "--wrap=preserve",
         # Code blocks here are measured output, not source. Highlighting emits a
         # Shaded environment only pandoc's own template defines.
         "--no-highlight"],
        input=md, capture_output=True, text=True,
    )
    if proc.returncode != 0:
        sys.exit(f"pandoc failed:\n{proc.stderr}")

    body = proc.stdout
    for uni, tex in UNICODE_TO_TEX:
        body = body.replace(uni, tex)
    # No table here carries a caption, so pandoc's counter-suppression wrapper is
    # unnecessary and fails without the exact ltcaption setup from its template.
    body = body.replace("{\\def\\LTcaptype{none} % do not increment counter", "{%")
    # Long repo paths inside \texttt run off the page in the artifact tables.
    # \path breaks at / and _ ; plain \texttt does not.
    body = re.sub(r"\\texttt\{([^{}]*/[^{}]*)\}",
                  lambda m: "\\path{" + m.group(1).replace("\\_", "_") + "}", body)
    # longtable is KEPT: pandoc sizes columns as fractions of \linewidth so they
    # already fit, and rewriting to tabular orphans \endhead / \endlastfoot.
    body = body.replace("\\begin{longtable}", "{\\small\\begin{longtable}")
    body = body.replace("\\end{longtable}", "\\end{longtable}}")

    leftover = sorted({c for c in body if ord(c) > 127})
    if leftover:
        print("[md2tex] WARNING unmapped non-ASCII: "
              + str([f"U+{ord(c):04X}" for c in leftover]), file=sys.stderr)
    print(body)
    print(f"[md2tex] {n_cites} citation groups rewritten", file=sys.stderr)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
