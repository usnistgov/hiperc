# Makefile for GPU diffusion code
# OpenACC implementation

CXX = pgcc
CXXFLAGS = -O3 -I. -I../common-diffusion -acc -ta=tesla -Minfo=accel -mp
LINKS = -lm -lpng

OBJS = boundaries.o discretization.o mesh.o numerics.o output.o timer.o

all: diffusion

.PHONY: all run clean cleanobjects cleanoutputs cleanall

diffusion: ../common-diffusion/main.c $(OBJS)
	$(CXX) $(CXXFLAGS) $(OBJS) $< -o $@ $(LINKS)

boundaries.o: boundaries.c
	$(CXX) $(CXXFLAGS) -c $<

discretization.o: discretization.c
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

