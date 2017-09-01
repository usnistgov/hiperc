# Makefile for GPU diffusion code
# CUDA implementation

NVCXX = nvcc
NVCXXFLAGS = -gencode arch=compute_35,code=sm_35 -D_FORCE_INLINES \
             -Wno-deprecated-gpu-targets -std=c++11 \
             --compiler-options="-O3 -Wall -I. -I../common-diffusion -fopenmp"
LINKS = -lm -lpng -lcuda

OBJS = boundaries.o discretization.o mesh.o numerics.o output.o timer.o

all: diffusion
.PHONY: all

diffusion: ../common-diffusion/main.c $(OBJS)
	$(NVCXX) $(NVCXXFLAGS) $(OBJS) $< -o $@ $(LINKS)

boundaries.o: cuda_boundaries.c
	$(NVCXX) $(NVCXXFLAGS) -dc $< -o $@

discretization.o: cuda_discretization.cu
	$(NVCXX) $(NVCXXFLAGS) -dc $< -o $@

mesh.o: ../common-diffusion/mesh.c
	$(NVCXX) $(NVCXXFLAGS) -dc $< -o $@

numerics.o: ../common-diffusion/numerics.c
	$(NVCXX) $(NVCXXFLAGS) -c $<

output.o: ../common-diffusion/output.c
	$(NVCXX) $(NVCXXFLAGS) -dc $< -o $@

timer.o: ../common-diffusion/timer.c
	$(NVCXX) $(NVCXXFLAGS) -dc $< -o $@

.PHONY: run
run: diffusion
	/usr/bin/time -f' Time (%E wall, %U user, %S sys)' ./diffusion ../common-diffusion/params.txt

.PHONY: cleanobjects
cleanobjects:
	rm -f diffusion *.o

.PHONY: cleanoutputs
cleanoutputs:
	rm -f diffusion.*.csv diffusion.*.png runlog.csv

.PHONY: clean
clean: cleanobjects

.PHONY: cleanall
cleanall: cleanobjects cleanoutputs

