# Makefile for CPU diffusion code
# OpenMP implementation

CC = gcc
CFLAGS = -O2 -Wall -pedantic -I../ -fopenmp
LINKS = -lm -lpng

OBJS = boundaries.o discretization.o mesh.o output.o timer.o

all: diffusion

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

clean:
	rm -f diffusion *.o

squeaky:
	rm -f diffusion *.o data.*.csv data.*.png runlog.csv
