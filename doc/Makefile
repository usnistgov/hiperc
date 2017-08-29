# Makefile for phasefield-accelerator-benchmarks documentation

LATEX = pdflatex
SPHINX = sphinx-build
DOXYGEN = doxygen

all: pdf

.PHONY: all clean dox pdf sphinx tex

dox:
	$(DOXYGEN)

tex: dox
	$(MAKE) -C latex && cd latex && $(LATEX) refman.tex

pdf: tex
	cp latex/refman.pdf phasefield-accelerator-benchmarks_guide.pdf

sphinx: dox
	$(SPHINX) -b html . _build

clean:
	rm -r _build/* latex xml
