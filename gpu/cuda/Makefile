# Makefile for GPU diffusion code
# CUDA implementation

CC = nvcc
CFLAGS = -O2 -Wall -pedantic -I../
LINKS = -lm -lpng

OBJS = boundaries.o discretization.o mesh.o output.o timer.o

all: diffusion

diffusion: ../main.c $(OBJS)
	$(CC) $(CFLAGS) $(OBJS) $< -o $@ $(LINKS)

boundaries.o: boundaries.c
	$(CC) $(CFLAGS) -c $<

discretization.o: discretization.c
	$(CC) $(CFLAGS) -c $<

mesh.o: ../mesh.c
	$(CC) -c $<

output.o: ../output.c
	$(CC) -c $<

timer.o: ../timer.c
	$(CC) -c $<

run: diffusion
	/usr/bin/time -f' Time (%E wall, %U user, %S sys)' ./diffusion ../params.txt

clean:
	rm -f diffusion *.o

squeaky:
	rm -f diffusion *.o data.*.csv data.*.png runlog.csv
