.PHONY: paper values tables lean-status figs clean all help

LATEX     := pdflatex -interaction=nonstopmode
BIBTEX    := bibtex
MAIN      := main
PYTHON    := python3
UVX_PY    := uv run --with matplotlib python3

all: paper

help:
	@echo "Targets:"
	@echo "  make values      — regenerate values.tex from ../results/*.json"
	@echo "  make tables      — regenerate tables.tex from ../results/*.json"
	@echo "  make lean-status — regenerate lean_status.tex from ../lean/*.lean"
	@echo "  make figs        — regenerate all 5 data-driven figure PDFs + schematic"
	@echo "  make paper       — all above + pdflatex + bibtex + pdflatex + pdflatex"
	@echo "  make clean       — remove build artefacts (keeps .pdf)"
	@echo "  make distclean   — remove everything including .pdf"

values: values.tex
values.tex: generate_values.py $(wildcard ../results/*/results.json) $(wildcard ../results/*/summary.json) $(wildcard ../results/*/loss_trajectory.json)
	$(PYTHON) generate_values.py

tables: tables.tex
tables.tex: generate_tables.py $(wildcard ../results/*/results.json) $(wildcard ../results/*/summary.json) $(wildcard ../results/*/loss_trajectory.json)
	$(PYTHON) generate_tables.py

lean-status: lean_status.tex
lean_status.tex: generate_lean_status.py ../lean/SubspaceOverlap.lean
	$(PYTHON) generate_lean_status.py

# ── Figure generation ────────────────────────────────────────────────────────
FIG_PDFS := figures/fig_A.pdf figures/fig_B.pdf figures/fig_C.pdf \
             figures/fig_D.pdf figures/fig_E.pdf

figs: $(FIG_PDFS)
	@echo "[figs] All figure PDFs up to date."

figures/fig_A.pdf: ../scripts/generate_fig_A.py $(wildcard ../results/spectral_overlap_gamma_petri/results.json) $(wildcard ../results/spectral_overlap_gamma/results.json) $(wildcard ../results/spectral_overlap_gamma_1b/results.json)
	@mkdir -p figures
	$(UVX_PY) ../scripts/generate_fig_A.py

figures/fig_B.pdf: ../scripts/generate_fig_B.py $(wildcard ../results/spectral_overlap_gamma_1b/results.json)
	@mkdir -p figures
	$(UVX_PY) ../scripts/generate_fig_B.py

figures/fig_C.pdf: ../scripts/generate_fig_C.py $(wildcard ../results/spectral_overlap_gamma_petri/results.json)
	@mkdir -p figures
	$(UVX_PY) ../scripts/generate_fig_C.py

figures/fig_D.pdf: ../scripts/generate_fig_D.py $(wildcard ../results/bitfit_dpo_strike/loss_trajectory.json) $(wildcard ../results/bitfit_dpo_strike_extended/loss_trajectory.json)
	@mkdir -p figures
	$(UVX_PY) ../scripts/generate_fig_D.py

figures/fig_E.pdf: ../scripts/generate_fig_E.py $(wildcard ../results/spectral_overlap_gamma_modules/results.json)
	@mkdir -p figures
	$(UVX_PY) ../scripts/generate_fig_E.py

paper: values tables lean-status figs $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex values.tex tables.tex lean_status.tex references.bib $(FIG_PDFS)
	$(LATEX) $(MAIN).tex
	-$(BIBTEX) $(MAIN)
	$(LATEX) $(MAIN).tex
	$(LATEX) $(MAIN).tex

clean:
	rm -f $(MAIN).aux $(MAIN).log $(MAIN).bbl $(MAIN).blg $(MAIN).out $(MAIN).toc

distclean: clean
	rm -f $(MAIN).pdf values.tex tables.tex lean_status.tex
