# Makefile for CPU diffusion code
# serial implementation (no parallel model)

CC = gcc
CFLAGS = -O3 -Wall -pedantic -I../common-diffusion
LINKS = -lm -lpng

OBJS = boundaries.o discretization.o mesh.o numerics.o output.o timer.o

# Executable
diffusion: serial_main.c $(OBJS)
	$(CC) $(CFLAGS) $(OBJS) $< -o $@ $(LINKS)

# Serial objects
boundaries.o: serial_boundaries.c
	$(CC) $(CFLAGS) -c $< -o $@

discretization.o: serial_discretization.c
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
