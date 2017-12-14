# Makefile for HiPerC spinodal decomposition code
# OpenMP implementation

CC = gcc
CFLAGS = -O3 -Wall -pedantic -I../common-spinodal -fopenmp
LINKS = -lm -lpng

OBJS = boundaries.o discretization.o mesh.o numerics.o output.o timer.o

# Executable
spinodal: openmp_main.c $(OBJS)
	$(CC) $(CFLAGS) $(OBJS) -include omp.h $< -o $@ $(LINKS)

# OpenMP objects
boundaries.o: openmp_boundaries.c
	$(CC) $(CFLAGS) -c $< -o $@

discretization.o: openmp_discretization.c
	$(CC) $(CFLAGS) -c $< -o $@

# Common objects
mesh.o: ../common-spinodal/mesh.c
	$(CC) $(CFLAGS) -c $< -o $@

numerics.o: ../common-spinodal/numerics.c
	$(CC) $(CFLAGS) -c $< -o $@

output.o: ../common-spinodal/output.c
	$(CC) $(CFLAGS) -c $< -o $@

timer.o: ../common-spinodal/timer.c
	$(CC) $(CFLAGS) -c $< -o $@

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
