# Makefile for GPU diffusion code
# OpenCL implementation

CC = gcc
CFLAGS = -O3 -Wall -pedantic -I../common-diffusion -fopenmp
LINKS = -lm -lpng -lOpenCL

OBJS = boundaries.o data.o discretization.o mesh.o numerics.o output.o timer.o

# Executable
diffusion: opencl_main.c $(OBJS)
	$(CC) $(CFLAGS) $(OBJS) $< -o $@ $(LINKS)

# OpenCL objects
boundaries.o: opencl_boundaries.c
	$(CC) $(CFLAGS) -c $<

data.o: opencl_data.c
	$(CC) $(CFLAGS) -c $<

discretization.o: opencl_discretization.c
	$(CC) $(CFLAGS) -c $<

# Common objects
mesh.o: ../common-diffusion/mesh.c
	$(CC) $(CFLAGS) -c $<

numerics.o: ../common-diffusion/numerics.c
	$(CC) $(CFLAGS) -c $<

output.o: ../common-diffusion/output.c
	$(CC) $(CFLAGS) -c $<

timer.o: ../common-diffusion/timer.c
	$(CC) $(CFLAGS) -c $<

# Helper scripts
.PHONY: run
run: diffusion
	/usr/bin/time -f' Time (%E wall, %U user, %S sys)' ./diffusion ../params.txt

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

