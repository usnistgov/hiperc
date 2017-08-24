# Makefile for phasefield-accelerator-benchmarks documentation

all: pdf

.PHONY: all clean cleanall pdf dox tex

dox:
	doxygen

tex: dox
	$(MAKE) -C latex

pdf: tex
	cp latex/refman.pdf phasefield-accelerator-benchmarks_guide.pdf

clean:
	rm phasefield-accelerator-benchmarks_guide.pdf

cleanall: clean
	rm -r doxygen_sqlite3.db html latex
