# Makefile for phasefield-accelerator-benchmarks documentation

all: pdf

.PHONY: all clean pdf dox tex

dox:
	doxygen

tex: dox
	$(MAKE) -C latex && $(MAKE) -C latex

pdf: tex
	cp latex/refman.pdf phasefield-accelerator-benchmarks_guide.pdf

clean:
	rm -r doxygen_sqlite3.db html latex
