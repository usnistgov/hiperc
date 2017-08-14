# Makefile for GPU diffusion code
# implementation-independent headers

all: cuda/diffusion openacc/diffusion opencl/diffusion

cuda/diffusion:
	$(MAKE) -C cuda

#openacc/diffusion:
#	$(MAKE) -C openacc

#opencl/diffusion:
#	$(MAKE) -C opencl

run:
	$(MAKE) -C cuda run
#	$(MAKE) -C openacc run; $(MAKE) -C opencl run

clean:
	$(MAKE) -C cuda clean
#	$(MAKE) -C openacc clean; $(MAKE) -C opencl clean

squeaky:
	$(MAKE) -C cuda squeaky
#	$(MAKE) -C openacc squeaky; $(MAKE) -C opencl squeaky
