# Makefile for GPU diffusion code
# implementation-independent headers

all: openacc cuda

.PHONY: all cuda openacc run clean cleanall

cuda:
	$(MAKE) -C cuda

openacc:
	$(MAKE) -C openacc

run:
	$(MAKE) -C cuda run; $(MAKE) -C openacc run

clean:
	$(MAKE) -C cuda clean; $(MAKE) -C openacc clean

cleanall:
	$(MAKE) -C cuda cleanall; $(MAKE) -C openacc cleanall
