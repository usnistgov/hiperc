# Makefile for GPU diffusion code
# OpenACC implementation

CXX = pgcc
CXXFLAGS = -O3 -I. -I../common-diffusion -acc -ta=tesla -Minfo=accel -mp
LINKS = -lm -lpng

OBJS = boundaries.o discretization.o mesh.o numerics.o output.o timer.o

all: diffusion
.PHONY: all

diffusion: ../common-diffusion/main.c $(OBJS)
	$(CXX) $(CXXFLAGS) $(OBJS) $< -o $@ $(LINKS)

boundaries.o: openacc_boundaries.c
	$(CXX) $(CXXFLAGS) -c $< -o $@

discretization.o: openacc_discretization.c
	$(CXX) $(CXXFLAGS) -c $< -o $@

mesh.o: ../common-diffusion/mesh.c
	$(CXX) $(CXXFLAGS) -c $< -o $@

numerics.o: ../common-diffusion/numerics.c
	$(CXX) $(CXXFLAGS) -c $< -o $@

output.o: ../common-diffusion/output.c
	$(CXX) $(CXXFLAGS) -c $< -o $@

timer.o: ../common-diffusion/timer.c
	$(CXX) $(CXXFLAGS) -c $< -o $@

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

