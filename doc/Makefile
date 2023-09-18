# Makefile for HiPerC documentation

all: pdf
.PHONY: all

.PHONY: dox
dox:
	doxygen

.PHONY: tex
tex: dox
	$(MAKE) -C latex && \
	cd latex && \
	cp refman.tex hiperc.tex && \
	latexmk hiperc.tex

.PHONY: pdf
pdf: tex
	cp latex/hiperc.pdf hiperc_guide.pdf

.PHONY: html
html: dox
	sphinx-build -b html . _build

.PHONY: clean
clean:
	rm -rf _build/* _static/* html latex xml rst/*
