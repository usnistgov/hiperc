# Makefile for GPU diffusion code
# implementation-independent headers

all: acc/diffusion cuda/diffusion

acc/diffusion:
	$(MAKE) -C acc

cuda/diffusion:
	$(MAKE) -C cuda

clean:
	$(MAKE) -C acc clean; $(MAKE) -C cuda clean

squeaky:
	$(MAKE) -C acc squeaky; $(MAKE) -C cuda squeaky
