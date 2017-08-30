# Makefile for GPU diffusion code
# OpenCL implementation

CXX = g++
CXXFLAGS = -O3 -Wall -std=c++11 -I. -I../common-diffusion -fopenmp
LINKS = -lm -lpng -lOpenCL

OBJS = boundaries.o discretization.o mesh.o numerics.o output.o timer.o

all: diffusion

.PHONY: all run clean cleanobjects cleanoutputs cleanall

diffusion: ../main.c $(OBJS)
	$(CXX) $(CXXFLAGS) $(OBJS) $< -o $@ $(LINKS)

boundaries.o: boundaries.c
	$(CXX) $(CXXFLAGS) -c $<

discretization.o: discretization.cl
	$(CXX) $(CXXFLAGS) -c $<

mesh.o: ../mesh.c
	$(CXX) $(CXXFLAGS) -c $<

numerics.o: ../common-diffusion/numerics.c
	$(CXX) $(CXXFLAGS) -c $<

output.o: ../output.c
	$(CXX) $(CXXFLAGS) -c $<

timer.o: ../timer.c
	$(CXX) $(CXXFLAGS) -c $<

run: diffusion
	/usr/bin/time -f' Time (%E wall, %U user, %S sys)' ./diffusion ../params.txt

cleanobjects:
	rm -f diffusion *.o

cleanoutputs:
	rm -f diffusion.*.csv diffusion.*.png runlog.csv

clean: cleanobjects

cleanall: cleanobjects cleanoutputs

