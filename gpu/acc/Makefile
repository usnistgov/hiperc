# Makefile for GPU diffusion code
# OpenACC implementation

CXX = pgcc
CXXFLAGS = -O2 -I../ -acc -ta=radeon -Minfo=accel -mp
LINKS = -lm -lpng

OBJS = boundaries.o discretization.o mesh.o output.o timer.o

all: diffusion

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

clean:
	rm -f diffusion *.o

squeaky:
	rm -f diffusion *.o data.*.csv data.*.png runlog.csv
