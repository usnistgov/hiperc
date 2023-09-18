# Makefile for HiPerC documentation

all: _build/html/index.html
.PHONY: all

_build/latex/hiperc.tex:
	doxygen && mv _build/latex/refman.tex $@

_build/latex/hiperc.pdf: _build/latex/hiperc.tex
	latexmk -cd $<

_build/html/index.html: _build/latex/hiperc.tex
	sphinx-build -b html . _build

.PHONY: clean html pdf
clean:
	rm -rf _build _static html latex rst
html: _build/html/index.html
pdf:  _build/latex/hiperc.pdf
