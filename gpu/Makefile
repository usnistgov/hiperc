# Makefile for GPU diffusion code
# implementation-independent headers

all: openacc/diffusion cuda/diffusion

openacc/diffusion:
	$(MAKE) -C openacc

cuda/diffusion:
	$(MAKE) -C cuda

run:
	$(MAKE) -C openacc run; $(MAKE) -C cuda run

clean:
	$(MAKE) -C openacc clean; $(MAKE) -C cuda clean

squeaky:
	$(MAKE) -C openacc squeaky; $(MAKE) -C cuda squeaky
