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
	$(MAKE) -C latex && \
	cd latex && \
	cp refman.tex hiperc.tex && \
	$(LATEX) hiperc.tex

.PHONY: pdf
pdf: tex
	cp latex/hiperc.pdf hiperc_guide.pdf

.PHONY: html
html: dox
	$(SPHINX) -b html . _build

.PHONY: clean
clean:
	rm -rf _build/* _static/* html latex xml rst/*
