# CPU code

This directory contains implementations of the diffusion equation for CPUs.

## Working Code

 - [x] serial
 - [x] OpenMP
 - [x] Threading Building Blocks

## Usage

This directory, and each sub-directory, contains a makefile with three important invocations:
 1. ```make``` will build the executable, named ```diffusion```, from its dependencies.
 2. ```make run``` will execute ```diffusion``` using the defaults listed in ```params.txt```,
    writing PNG and CSV output for inspection. ```runlog.csv``` contains the time-evolution of
    the weighted sum-of-squares residual from the analytical solution, as well as runtime data.
 3. ```make clean``` will remove the executable and object files ```.o```, but not the data.

To test the code, ```make run``` from this directory (```cpu```).

## Dependencies

To build this code, you must have installed
 * [GNU make](https://www.gnu.org/software/make/);
 * the [GNU compiler collection](https://gcc.gnu.org) (```gcc``` and ```g++```);
 * the [PNG](http://www.libpng.org/pub/png/libpng.html) library;
 * the [Threading Building Blocks (TBB)](https://www.threadingbuildingblocks.org) library.

These are usually available through the package manager. For example,
```apt-get install make libpng12-dev libtbb-dev``` or ```yum install make libpng-devel tbb-devel```.

## Source Layout

```
 cpu
 ├── serial
 │   ├── boundaries.c
 │   ├── discretization.c
 │   └── Makefile
 ├── openmp
 │   ├── boundaries.c
 │   ├── discretization.c
 │   └── Makefile
 ├── tbb
 │   ├── boundaries.cpp
 │   ├── discretization.cpp
 │   └── Makefile
 ├── diffusion.h
 ├── main.c
 ├── Makefile
 ├── mesh.c
 ├── output.c
 ├── params.txt
 ├── README.md
 └── timer.c
```

The interface (prototypes for all functions) is defined in the top-level ```diffusion.h```.
The mesh, output, and timer functions contain no specialized code, and therefore
```mesh.c```, ```output.c```, and ```timer.c``` reside alongside ```diffusion.h``` and ```main.c```.
The implementation of boundary conditions and discretized mathematics depend strongly on
the parallelization scheme, so each sub-directory contains specialized versions of ```boundaries.c```
and ```discretization.c```. When ```make``` is called, each ```.c``` file gets compiled into an object ```.o``` in the
sub-directory, allowing for different compilers in each case. 

The default input file ```params.txt``` defines nine values via key-value pairs,
with one pair per line. The two-character keys are predefined, and must be one of
```{nt, nx, ny, dx, dy, ns, nc, dc, co}```. Descriptive comments follow the value on each line.
If you wish to change parameters (D, runtime, etc.), either modify ```params.txt``` in place and
```make run```, or create your own copy of ```params.txt``` and execute ```./diffusion newparams```.
The file name and extension make no difference, so long as it contains plain text.
