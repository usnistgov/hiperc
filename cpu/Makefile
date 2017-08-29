# Makefile for CPU diffusion code
# implementation-independent headers

all: analytic serial openmp tbb

.PHONY: all analytic serial openmp tbb run clean cleanall

analytic:
	$(MAKE) -C analytic

serial:
	$(MAKE) -C serial

openmp:
	$(MAKE) -C openmp

tbb:
	$(MAKE) -C tbb

run:
	$(MAKE) -C analytic run
	$(MAKE) -C serial   run
	$(MAKE) -C openmp   run
	$(MAKE) -C tbb      run

clean:
	$(MAKE) -C analytic clean
	$(MAKE) -C serial   clean
	$(MAKE) -C openmp   clean
	$(MAKE) -C tbb      clean

cleanall:
	$(MAKE) -C analytic cleanall
	$(MAKE) -C serial   cleanall
	$(MAKE) -C openmp   cleanall
	$(MAKE) -C tbb      cleanall

