# Hedgehog CPU diffusion code

Implementation of the diffusion equation for the CPU with Hedgehog

## Usage

This directory contains a CMake file  with three important invocations:

1. Create the build directory: `mkdir build` then `cd build`
2. Create Makefile with CMake: 
   `cmake -DHedgehog_INCLUDE_DIR=/path/to/Hedgehog/src -DCMAKE_BUILD_TYPE=Release ../` 
3. `make` will build the executable, named `diffusion_Hedgehog`, from its
   dependencies.
4. `./diffusion_Hedgehog ../../common_diffusion/params.txt` will execute
   `diffusion_Hedgehog` using the defaults listed in
   `../common_diffusion/params.txt`, writing PNG and CSV output for inspection.
   `runlog.csv` contains the time-evolution of the weighted sum-of-squares
   residual from the analytical solution, as well as runtime data.
5. `make clean` will remove the executable and object files `.o`,
   but not the data.

## Dependencies

To build this code, you must have installed
* [GNU make][_make]
* [CMake][_cmake]
* [GNU compiler collection][_gcc]
* [PNG library][_png]
* [Hedgehog][_hedgehog]

These are usually available through the package manager. For example, `apt-get
install cmake make libpng12-dev` or `yum install cmake make libpng-devel`.

## Customization

The default input file `../common-diffusion/params.txt` defines
key-value pairs, with one pair per line. The two-character keys are
predefined, and must all be present. Descriptive comments follow the
value on each line. If you wish to change parameters (*D*, runtime,
etc.), either modify `params.txt` in place and `make run`, or create
your own copy of `params.txt` and execute `./diffusion
<your_params.txt>`. The file name and extension make no difference, so
long as it contains plain text.

<!-- References -->

[_make]: https://www.gnu.org/software/make/
[_cmake]: https://cmake.org/download/
[_gcc]:  https://gcc.gnu.org
[_png]:  http://www.libpng.org/pub/png/libpng.html
[_hedgehog]: https://github.com/usnistgov/hedgehog
