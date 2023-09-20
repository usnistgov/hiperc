# Makefile for HiPerC documentation

all: _build/html/index.html
.PHONY: all

_build/latex/hiperc.tex:
	doxygen && mv _build/latex/refman.tex $@

_build/latex/hiperc.pdf: _build/latex/hiperc.tex
	latexmk -cd $<

_build/html/indexy.html: _build/latex/hiperc.pdf
	sphinx-build -M html . _build

_build/html/index.html:
	sphinx-build -M html . _build

.PHONY: clean html pdf
clean:
	rm -rf _build rst
html: _build/html/index.html
pdf:  _build/latex/hiperc.pdf
