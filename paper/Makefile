# arXiv build for ../PAPER_2026-08_Recipe_Dependent_Rankings.md
#
# body.tex and refs.tex are GENERATED. The Markdown one directory up is the
# source of truth: every number in it is checked against the run records by
# derive_figures.py --check, so edits belong there, not here.

MD       := ../PAPER_2026-08_Recipe_Dependent_Rankings.md
FIGDIR   := ../experiment-notes/nanolab/figures
TECTONIC := $(shell command -v tectonic 2>/dev/null)

.PHONY: all clean check arxiv verify

# A generator that fails must not leave a truncated target behind. Without this,
# `python3 mkbib.py ... > refs.tex` creates an empty refs.tex, mkbib exits 1,
# make carries on, and the PDF ships with no bibliography and no warnings -- a
# failed check reading exactly like a passed one.
.DELETE_ON_ERROR:

all: main.pdf

main.pdf: main.tex body.tex refs.tex
ifeq ($(TECTONIC),)
	$(MAKE) latexmk
else
	TECTONIC_DEP_BACKEND=$${TECTONIC_DEP_BACKEND:-vcpkg} tectonic -X compile main.tex
endif

latexmk: main.tex body.tex refs.tex
	latexmk -pdf -interaction=nonstopmode main.tex

body.tex: $(MD) md2tex.py refs.bib
	python3 md2tex.py $(MD) > $@

refs.tex: $(MD) mkbib.py
	python3 mkbib.py $(MD) > $@

# Refuse to ship a PDF whose citations did not resolve.
check: main.pdf
	@if grep -q "Citation .* undefined" main.log 2>/dev/null; then \
	  echo "FAIL: undefined citations:"; grep "Citation .* undefined" main.log; exit 1; fi
	@if grep -qE "Reference .* undefined|There were undefined references" main.log 2>/dev/null; then \
	  echo "FAIL: undefined references"; exit 1; fi
	@echo "OK: no undefined citations or references"

# The manuscript's numbers must still match the runs before a PDF ships.
verify:
	python3 derive_figures.py --check

# arXiv wants a self-contained tarball: sources + figures, no build products.
arxiv: main.pdf check verify
	rm -rf arxiv-src && mkdir -p arxiv-src/figures
	cp main.tex body.tex refs.tex refs.bib arxiv-src/
	cp $(FIGDIR)/*.pdf arxiv-src/figures/ 2>/dev/null || true
	cd arxiv-src && tar czf ../arxiv-submission.tar.gz .
	@echo "wrote paper/arxiv-submission.tar.gz"

clean:
	rm -f main.pdf main.aux main.log main.out main.bbl main.blg main.fls \
	      main.fdb_latexmk body.tex refs.tex conv.log
	rm -rf arxiv-src arxiv-submission.tar.gz
