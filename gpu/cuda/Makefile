# Makefile for GPU diffusion code
# CUDA implementation

NVCXX = nvcc
NVCXXFLAGS = -Wno-deprecated-gpu-targets --compiler-options="-O2 -Wall -pedantic -I../"
LINKS = -lm -lpng

OBJS = boundaries.o discretization.o mesh.o output.o timer.o

all: diffusion

diffusion: ../main.c $(OBJS)
	$(NVCXX) $(NVCXXFLAGS) $(OBJS) $< -o $@ $(LINKS)

boundaries.o: boundaries.c
	$(NVCXX) $(NVCXXFLAGS) -c $<

discretization.o: discretization.c
	$(NVCXX) $(NVCXXFLAGS) -c $<

mesh.o: ../mesh.c
	$(NVCXX) $(NVCXXFLAGS) -c $<

output.o: ../output.c
	$(NVCXX) $(NVCXXFLAGS) -c $<

timer.o: ../timer.c
	$(NVCXX) $(NVCXXFLAGS) -c $<

run: diffusion
	/usr/bin/time -f' Time (%E wall, %U user, %S sys)' ./diffusion ../params.txt

clean:
	rm -f diffusion *.o

squeaky:
	rm -f diffusion *.o data.*.csv data.*.png runlog.csv
