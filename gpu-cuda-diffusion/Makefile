# Makefile for GPU diffusion code
# CUDA implementation

NVCXX = nvcc
NVCXXFLAGS = -gencode arch=compute_35,code=sm_35 -D_FORCE_INLINES \
             -Wno-deprecated-gpu-targets -std=c++11 \
             --compiler-options="-O3 -Wall -I. -I../common-diffusion -fopenmp"
LINKS = -lm -lpng -lcuda

OBJS = boundaries.o discretization.o mesh.o numerics.o output.o timer.o

all: diffusion

.PHONY: all run clean cleanobjects cleanoutputs cleanall

diffusion: ../common-diffusion/main.c $(OBJS)
	$(NVCXX) $(NVCXXFLAGS) $(OBJS) $< -o $@ $(LINKS)

boundaries.o: boundaries.c
	$(NVCXX) $(NVCXXFLAGS) -dc $<

discretization.o: discretization.cu
	$(NVCXX) $(NVCXXFLAGS) -dc $<

mesh.o: ../common-diffusion/mesh.c
	$(NVCXX) $(NVCXXFLAGS) -dc $<

numerics.o: ../common-diffusion/numerics.c
	$(NVCXX) $(NVCXXFLAGS) -c $<

output.o: ../common-diffusion/output.c
	$(NVCXX) $(NVCXXFLAGS) -dc $<

timer.o: ../common-diffusion/timer.c
	$(NVCXX) $(NVCXXFLAGS) -dc $<

run: diffusion
	/usr/bin/time -f' Time (%E wall, %U user, %S sys)' ./diffusion ../common-diffusion/params.txt

cleanobjects:
	rm -f diffusion *.o

cleanoutputs:
	rm -f diffusion.*.csv diffusion.*.png runlog.csv

clean: cleanobjects

cleanall: cleanobjects cleanoutputs

