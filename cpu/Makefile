# Makefile for CPU diffusion code
# implementation-independent headers

all: serial openmp tbb

.PHONY: all serial openmp tbb run clean cleanall

serial:
	$(MAKE) -C serial

openmp:
	$(MAKE) -C openmp

tbb:
	$(MAKE) -C tbb

run:
	$(MAKE) -C serial run; $(MAKE) -C openmp run; $(MAKE) -C tbb run

clean:
	$(MAKE) -C serial clean; $(MAKE) -C openmp clean; $(MAKE) -C tbb clean

cleanall:
	$(MAKE) -C serial cleanall; $(MAKE) -C openmp cleanall; $(MAKE) -C tbb cleanall

