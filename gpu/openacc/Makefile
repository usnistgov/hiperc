# Makefile for GPU diffusion code
# OpenACC implementation

CXX = pgcc
CXXFLAGS = -O3 -I. -I.. -acc -ta=tesla -Minfo=accel -mp
LINKS = -lm -lpng

OBJS = boundaries.o discretization.o mesh.o output.o timer.o

all: diffusion

.PHONY: all run clean cleanobjects cleanoutputs cleanall

diffusion: ../main.c $(OBJS)
	$(CXX) $(CXXFLAGS) $(OBJS) $< -o $@ $(LINKS)

boundaries.o: boundaries.c
	$(CXX) $(CXXFLAGS) -c $<

discretization.o: discretization.c
	$(CXX) $(CXXFLAGS) -c $<

mesh.o: ../mesh.c
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

