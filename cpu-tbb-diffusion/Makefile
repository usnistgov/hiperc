# Makefile for CPU diffusion code
# Threading Building Blocks implementation

CXX = g++
CXXFLAGS = -O3 -Wall -pedantic -std=c++11 -I. -I../common-diffusion
LINKS = -lm -lpng -ltbb

OBJS = boundaries.o discretization.o mesh.o numerics.o output.o timer.o

all: diffusion

.PHONY: all run clean cleanobjects cleanoutputs cleanall

diffusion: ../common-diffusion/main.c $(OBJS)
	$(CXX) $(CXXFLAGS) $(OBJS) $< -o $@ $(LINKS)

boundaries.o: boundaries.cpp
	$(CXX) $(CXXFLAGS) -c $<

discretization.o: discretization.cpp
	$(CXX) $(CXXFLAGS) -c $<

mesh.o: ../common-diffusion/mesh.c
	$(CXX) $(CXXFLAGS) -c $<

numerics.o: ../common-diffusion/numerics.c
	$(CXX) $(CXXFLAGS) -c $<

output.o: ../common-diffusion/output.c
	$(CXX) $(CXXFLAGS) -c $<

timer.o: ../common-diffusion/timer.c
	$(CXX) $(CXXFLAGS) -c $<

run: diffusion
	/usr/bin/time -f' Time (%E wall, %U user, %S sys)' ./diffusion ../common-diffusion/params.txt

cleanobjects:
	rm -f diffusion *.o

cleanoutputs:
	rm -f diffusion.*.csv diffusion.*.png runlog.csv

clean: cleanobjects

cleanall: cleanobjects cleanoutputs

