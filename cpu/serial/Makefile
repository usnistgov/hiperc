# Makefile for CPU diffusion code
# serial implementation (no parallel model)

CC = gcc
CFLAGS = -O2 -Wall -pedantic -I../
LINKS = -lm -lpng

OBJS = ../mesh.o ../output.o ../timer.o \
       boundaries.o discretization.o

all: diffusion

diffusion: ../main.c $(OBJS)
	$(CC) $(CFLAGS) $(OBJS) $< -o $@ $(LINKS)

boundaries.o: boundaries.c
	$(CC) $(CFLAGS) -c $<

discretization.o: discretization.c
	$(CC) $(CFLAGS) -c $<

run: diffusion
	./diffusion ../params.txt

clean:
	rm -f diffusion *.o data.*.csv data.*.png runlog.csv
