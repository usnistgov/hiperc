# Makefile for HiPerC documentation

LATEX = pdflatex
SPHINX = sphinx-build
DOXYGEN = doxygen

all: pdf
.PHONY: all

.PHONY: dox
dox:
	$(DOXYGEN)

.PHONY: tex
tex: dox
	$(MAKE) -C latex && cd latex && $(LATEX) refman.tex

.PHONY: pdf
pdf: tex
	cp latex/refman.pdf hiperc_guide.pdf

.PHONY: html
html: dox
	$(SPHINX) -b html . _build

.PHONY: clean
clean:
	rm -rf _build/* _static/* latex xml rst/*
