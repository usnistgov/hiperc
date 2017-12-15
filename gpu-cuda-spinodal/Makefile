# Makefile for HiPerC spinodal code
# CUDA implementation

NVCXX = nvcc
NVCXXFLAGS = -D_FORCE_INLINES -Wno-deprecated-gpu-targets -std=c++11 \
             --compiler-options="-O3 -Wall -I../common-spinodal -fopenmp"
LINKS = -lm -lpng -lcuda

OBJS = boundaries.o data.o discretization.o mesh.o numerics.o output.o timer.o

# Executable
spinodal: cuda_main.c $(OBJS)
	$(NVCXX) $(NVCXXFLAGS) $(OBJS) $< -o $@ $(LINKS)

# CUDA objects

boundaries.o: cuda_boundaries.cu
	$(NVCXX) $(NVCXXFLAGS) -dc $< -o $@

data.o: cuda_data.cu
	$(NVCXX) $(NVCXXFLAGS) -dc $< -o $@

discretization.o: cuda_discretization.cu
	$(NVCXX) $(NVCXXFLAGS) -dc $< -o $@

# Common objects

mesh.o: ../common-spinodal/mesh.c
	$(NVCXX) $(NVCXXFLAGS) -c $< -o $@

numerics.o: ../common-spinodal/numerics.c
	$(NVCXX) $(NVCXXFLAGS) -c $<

output.o: ../common-spinodal/output.c
	$(NVCXX) $(NVCXXFLAGS) -c $< -o $@

timer.o: ../common-spinodal/timer.c
	$(NVCXX) $(NVCXXFLAGS) -c $< -o $@

# Helper scripts

.PHONY: run
run: spinodal
	/usr/bin/time -f' Time (%E wall, %U user, %S sys)' ./spinodal ../common-spinodal/params.txt

.PHONY: cleanobjects
cleanobjects:
	rm -f spinodal *.o

.PHONY: cleanoutputs
cleanoutputs:
	rm -f spinodal.*.csv spinodal.*.png runlog.csv

.PHONY: clean
clean: cleanobjects

.PHONY: cleanall
cleanall: cleanobjects cleanoutputs

