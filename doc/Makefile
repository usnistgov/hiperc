# Makefile for HiPerC documentation

all: html/index.html
.PHONY: all

latex/hiperc.tex:
	doxygen && mv latex/refman.tex $@

latex/hiperc.pdf: latex/hiperc.tex
	latexmk -cd $<

html/index.html: latex/hiperc.tex
	sphinx-build -b html . _build

.PHONY: clean html pdf
clean:
	rm -rf _build _static html latex rst
html: html/index.html
pdf:  latex/hiperc.pdf
