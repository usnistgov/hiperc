# Makefile for CPU diffusion code

CC = gcc
CFLAGS = -O2 -Wall -Werror -Wmissing-prototypes -pedantic

OBJS = mesh.o boundaries.o discretization.o output.o

all: diffusion

diffusion: diffusion.c $(OBJS)
	$(CC) $(CFLAGS) $(OBJS) $< -o $@

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
	rm -f diffusion *.o
