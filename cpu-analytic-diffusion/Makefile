# Makefile for HiPerC diffusion code
# Analytical solution

CC = gcc
CFLAGS = -O3 -Wall -pedantic -I../common-diffusion
LINKS = -lm -lpng

OBJS = mesh.o numerics.o output.o timer.o

# Executable
diffusion: analytic_main.c $(OBJS)
	$(CC) $(CFLAGS) $(OBJS) $< -o $@ $(LINKS)

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

.PHONY: compare
compare: run
	composite diffusion.0100000.png ../common-diffusion/diffusion.10000.png -compose difference delta.png

.PHONY: cleanobjects
cleanobjects:
	rm -f diffusion *.o

.PHONY: cleanoutputs
cleanoutputs:
	rm -f delta.png diffusion.*.csv diffusion.*.png runlog.csv

.PHONY: clean
clean: cleanobjects

.PHONY: cleanall
cleanall: cleanobjects cleanoutputs
