# Makefile for HiPerC diffusion code
# Threading Building Blocks implementation

CXX = g++
CXXFLAGS = -O3 -Wall -pedantic -std=c++11 -I../common-diffusion
LINKS = -lm -lpng -ltbb

OBJS = boundaries.o discretization.o mesh.o numerics.o output.o timer.o

# Executable
diffusion: tbb_main.c $(OBJS)
	$(CXX) $(CXXFLAGS) $(OBJS) $< -o $@ $(LINKS)

# TBB objects
boundaries.o: tbb_boundaries.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

discretization.o: tbb_discretization.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Common objects
mesh.o: ../common-diffusion/mesh.c
	$(CXX) $(CXXFLAGS) -c $< -o $@

numerics.o: ../common-diffusion/numerics.c
	$(CXX) $(CXXFLAGS) -c $< -o $@

output.o: ../common-diffusion/output.c
	$(CXX) $(CXXFLAGS) -c $< -o $@

timer.o: ../common-diffusion/timer.c
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Helper scripts
.PHONY: run
run: diffusion
	/usr/bin/time -f' Time (%E wall, %U user, %S sys)' ./diffusion ../common-diffusion/params.txt

.PHONY: cleanobjects
cleanobjects:
	rm -f diffusion *.o

.PHONY: cleanoutputs
cleanoutputs:
	rm -f diffusion.*.csv diffusion.*.png runlog.csv

.PHONY: clean
clean: cleanobjects

.PHONY: cleanall
cleanall: cleanobjects cleanoutputs

