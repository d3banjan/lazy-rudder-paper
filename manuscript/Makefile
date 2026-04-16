.PHONY: paper values tables lean-status clean all help

LATEX     := pdflatex -interaction=nonstopmode
BIBTEX    := bibtex
MAIN      := main
PYTHON    := python3

all: paper

help:
	@echo "Targets:"
	@echo "  make values      — regenerate values.tex from ../results/*.json"
	@echo "  make tables      — regenerate tables.tex from ../results/*.json"
	@echo "  make lean-status — regenerate lean_status.tex from ../lean/*.lean"
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

paper: values tables lean-status $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex values.tex tables.tex lean_status.tex references.bib
	$(LATEX) $(MAIN).tex
	-$(BIBTEX) $(MAIN)
	$(LATEX) $(MAIN).tex
	$(LATEX) $(MAIN).tex

clean:
	rm -f $(MAIN).aux $(MAIN).log $(MAIN).bbl $(MAIN).blg $(MAIN).out $(MAIN).toc

distclean: clean
	rm -f $(MAIN).pdf values.tex tables.tex lean_status.tex
