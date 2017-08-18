# Makefile for GPU diffusion code
# CUDA implementation

NVCXX = nvcc
NVCXXFLAGS = -Wno-deprecated-gpu-targets -std=c++11 -gencode arch=compute_35,code=sm_35 \
             --compiler-options="-O2 -Wall -I.. -fopenmp"
LINKS = -lm -lpng -lcuda

OBJS = boundaries.o discretization.o mesh.o output.o timer.o

all: diffusion

diffusion: ../main.c $(OBJS)
	$(NVCXX) $(NVCXXFLAGS) $(OBJS) $< -o $@ $(LINKS)

boundaries.o: boundaries.c
	$(NVCXX) $(NVCXXFLAGS) -dc $<

discretization.o: discretization.cu
	$(NVCXX) $(NVCXXFLAGS) -dc $<

mesh.o: ../mesh.c
	$(NVCXX) $(NVCXXFLAGS) -dc $<

output.o: ../output.c
	$(NVCXX) $(NVCXXFLAGS) -dc $<

timer.o: ../timer.c
	$(NVCXX) $(NVCXXFLAGS) -dc $<

run: diffusion
	/usr/bin/time -f' Time (%E wall, %U user, %S sys)' ./diffusion ../params.txt

clean:
	rm -f diffusion *.o

squeaky:
	rm -f diffusion *.o data.*.csv data.*.png runlog.csv
