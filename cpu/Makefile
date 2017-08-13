# Makefile for CPU diffusion code
# implementation-independent headers

all: serial/diffusion openmp/diffusion tbb/diffusion

serial/diffusion:
	$(MAKE) -C serial

openmp/diffusion:
	$(MAKE) -C openmp

tbb/diffusion:
	$(MAKE) -C tbb

runall:
	$(MAKE) -C serial run; $(MAKE) -C openmp run; $(MAKE) -C tbb run

clean:
	$(MAKE) -C serial clean; $(MAKE) -C openmp clean; $(MAKE) -C tbb clean

squeaky:
	$(MAKE) -C serial squeaky; $(MAKE) -C openmp squeaky; $(MAKE) -C tbb squeaky
