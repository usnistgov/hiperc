# Makefile for CPU diffusion code
# implementation-independent headers

CC = g++ # TBB requires C++, so g++ is required over gcc
CFLAGS = -O2 -Wall -pedantic

all: mesh.o output.o timer.o

mesh.o: mesh.c
	$(CC) $(CFLAGS) -c $<

output.o: output.c
	$(CC) $(CFLAGS) -c $<

timer.o: timer.c
	$(CC) $(CFLAGS) -c $<

clean:
	rm -f *.o
