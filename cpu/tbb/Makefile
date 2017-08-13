# Makefile for CPU diffusion code
# serial implementation (no parallel model)

CXX = g++
CPPFLAGS = -O2 -Wall -pedantic -I../
LINKS = -lm -lpng -ltbb

OBJS = boundaries.o discretization.o mesh.o output.o timer.o

all: diffusion

diffusion: ../main.c $(OBJS)
	$(CXX) $(CPPFLAGS) $(OBJS) $< -o $@ $(LINKS)

boundaries.o: boundaries.c
	$(CXX) $(CPPFLAGS) -c $<

discretization.o: discretization.c
	$(CXX) $(CPPFLAGS) -c $<

mesh.o: ../mesh.c
	$(CXX) $(CPPFLAGS) -c $<

output.o: ../output.c
	$(CXX) $(CPPFLAGS) -c $<

timer.o: ../timer.c
	$(CXX) $(CPPFLAGS) -c $<

run: diffusion
	/usr/bin/time -f' Time (%E wall, %U user, %S sys)' ./diffusion ../params.txt

clean:
	rm -f diffusion *.o

squeaky:
	rm -f diffusion *.o data.*.csv data.*.png runlog.csv
