# Makefile for CPU diffusion code

CC = gcc
CFLAGS = -O0 -g -Wall -Werror -Wmissing-prototypes -pedantic
LINKS = -lm -lpng

OBJS = mesh.o boundaries.o discretization.o output.o

all: diffusion

diffusion: diffusion.c $(OBJS)
	$(CC) $(CFLAGS) $(OBJS) $< -o $@ $(LINKS)

mesh.o: mesh.c
	$(CC) $(CFLAGS) -c $<

boundaries.o: boundaries.c
	$(CC) $(CFLAGS) -c $<

discretization.o: discretization.c
	$(CC) $(CFLAGS) -c $<

output.o: output.c
	$(CC) $(CFLAGS) -c $<

run: diffusion
	./diffusion params.txt

clean:
	rm -f diffusion *.o data.*.csv data.*.png error.csv
