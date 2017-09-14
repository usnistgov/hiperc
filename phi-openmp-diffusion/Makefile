# Makefile for HiPerC diffusion code
# Xeon Phi implementation

CC = icc
CFLAGS = -O3 -openmp -xmic-avx512 -I../common-diffusion
QOPT = -qopt-prefetch=3 -qopt-report=4 -qopt-report-file="qopt.log"
LINKS = -lpng
# QOFF = -qoffload-arch=mic-avx512 # Only activate if offloading to KNL coprocessor

OBJS = boundaries.o discretization.o mesh.o numerics.o output.o timer.o

# Executable
diffusion: phi_main.c $(OBJS)
	$(CC) $(CFLAGS) $(OBJS) -include omp.h $< -o $@ $(QOPT) $(LINKS)

# Xeon Phi objects
boundaries.o: phi_boundaries.c
	$(CC) $(CFLAGS) -c $< -o $@

discretization.o: phi_discretization.c
	$(CC) $(CFLAGS) -c $< -o $@

# Common objects
mesh.o: ../common-diffusion/mesh.c
	$(CC) $(CFLAGS) -c $< -o $@

numerics.o: ../common-diffusion/numerics.c
	$(CC) $(CFLAGS) -c $< -o $@

output.o: ../common-diffusion/output.c
	$(CC) $(CFLAGS) -c $< -o $@

timer.o: ../common-diffusion/timer.c
	$(CC) $(CFLAGS) -c $< -o $@

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
