# Makefile for CPU diffusion code
# OpenMP implementation

CC = gcc
CFLAGS = -O3 -Wall -pedantic -I../ -fopenmp
LINKS = -lm -lpng

OBJS = boundaries.o discretization.o mesh.o output.o timer.o

all: diffusion

.PHONY: all run cleanobjects cleanoutputs clean cleanall

diffusion: ../main.c $(OBJS)
	$(CC) $(CFLAGS) $(OBJS) -include omp.h $< -o $@ $(LINKS)

boundaries.o: boundaries.c
	$(CC) $(CFLAGS) -c $<

discretization.o: discretization.c
	$(CC) $(CFLAGS) -c $<

mesh.o: ../mesh.c
	$(CC) $(CFLAGS) -c $<

output.o: ../output.c
	$(CC) $(CFLAGS) -c $<

timer.o: ../timer.c
	$(CC) $(CFLAGS) -c $<

run: diffusion
	/usr/bin/time -f' Time (%E wall, %U user, %S sys)' ./diffusion ../params.txt

cleanobjects:
	rm -f diffusion *.o

cleanoutputs: 
	rm -f diffusion.*.csv diffusion.*.png runlog.csv

clean: cleanobjects

cleanall: cleanobjects cleanoutputs 
