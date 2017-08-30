# Makefile for CPU diffusion code
# serial implementation (no parallel model)

CC = gcc
CFLAGS = -O3 -Wall -pedantic -I. -I../common-diffusion
LINKS = -lm -lpng

OBJS = boundaries.o discretization.o mesh.o numerics.o output.o timer.o

all: diffusion

.PHONY: all run clean cleanobjects cleanoutputs cleanall

diffusion: ../common-diffusion/main.c $(OBJS)
	$(CC) $(CFLAGS) $(OBJS) $< -o $@ $(LINKS)

boundaries.o: boundaries.c
	$(CC) $(CFLAGS) -c $<

discretization.o: discretization.c
	$(CC) $(CFLAGS) -c $<

mesh.o: ../common-diffusion/mesh.c
	$(CC) $(CFLAGS) -c $<

numerics.o: ../common-diffusion/numerics.c
	$(CC) $(CFLAGS) -c $<

output.o: ../common-diffusion/output.c
	$(CC) $(CFLAGS) -c $<

timer.o: ../common-diffusion/timer.c
	$(CC) $(CFLAGS) -c $<

run: diffusion
	/usr/bin/time -f' Time (%E wall, %U user, %S sys)' ./diffusion ../common-diffusion/params.txt

cleanobjects:
	rm -f diffusion *.o

cleanoutputs:
	rm -f diffusion.*.csv diffusion.*.png runlog.csv

clean: cleanobjects

cleanall: cleanobjects cleanoutputs
