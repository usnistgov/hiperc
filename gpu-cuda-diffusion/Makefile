# Makefile for HiPerC diffusion code
# CUDA implementation

NVCXX = nvcc
NVCXXFLAGS = -D_FORCE_INLINES -Wno-deprecated-gpu-targets -std=c++11 \
             --compiler-options="-O3 -Wall -I../common-diffusion -fopenmp"
LINKS = -lm -lpng -lcuda

OBJS = boundaries.o data.o discretization.o mesh.o numerics.o output.o timer.o

# Executable
diffusion: cuda_main.cu $(OBJS)
	$(NVCXX) $(NVCXXFLAGS) $(OBJS) $< -o $@ $(LINKS)

# CUDA objects

boundaries.o: cuda_boundaries.cu
	$(NVCXX) $(NVCXXFLAGS) -c $< -o $@

data.o: cuda_data.cu
	$(NVCXX) $(NVCXXFLAGS) -c $< -o $@

discretization.o: cuda_discretization.cu
	$(NVCXX) $(NVCXXFLAGS) -c $< -o $@

# Common objects

mesh.o: ../common-diffusion/mesh.c
	$(NVCXX) $(NVCXXFLAGS) -dc $< -o $@

numerics.o: ../common-diffusion/numerics.c
	$(NVCXX) $(NVCXXFLAGS) -dc $<

output.o: ../common-diffusion/output.c
	$(NVCXX) $(NVCXXFLAGS) -dc $< -o $@

timer.o: ../common-diffusion/timer.c
	$(NVCXX) $(NVCXXFLAGS) -dc $< -o $@

# Helper scripts

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

