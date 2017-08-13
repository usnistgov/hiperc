# Makefile for CPU diffusion code
# serial implementation (no parallel model)

CXX = g++
CPPFLAGS = -O2 -Wall -pedantic -I../
LINKS = -lm -lpng -ltbb

OBJS = ../mesh.o ../output.o ../timer.o \
       boundaries.o discretization.o

all: diffusion

diffusion: ../main.c $(OBJS)
	$(CXX) $(CPPFLAGS) $(OBJS) $< -o $@ $(LINKS)

../mesh.o:
	$(MAKE) -C ..

../output.o:
	$(MAKE) -C ..

../timer.o:
	$(MAKE) -C ..

boundaries.o: boundaries.c
	$(CXX) $(CPPFLAGS) -c $<

discretization.o: discretization.c
	$(CXX) $(CPPFLAGS) -c $<

run: diffusion
	./diffusion ../params.txt

clean:
	rm -f diffusion *.o data.*.csv data.*.png runlog.csv
