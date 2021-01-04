# HTGS CPU diffusion code

Implementation of the diffusion equation for the CPU with HTGS

## Usage

This directory contains a CMake file  with three important invocations:
 1. ```mkdir build``` create a build directory and ```cd build```
 2. ```cmake -DHTGS_INCLUDE_DIR=/path/to/HTGS/src -DCMAKE_BUILD_TYPE=Release ../``` create Makefile with cmake command
 3. ```make``` will build the executable, named ```diffusion_HTGS```,
    from its dependencies.
 4. ```/diffusion_HTGS ../../common_diffusion/params.txt``` will execute ```diffusion_HTGS``` using the defaults listed in
    ```../common_diffusion/params.txt```, writing PNG and CSV output for
    inspection. ```runlog.csv``` contains the time-evolution of the weighted
    sum-of-squares residual from the analytical solution, as well as runtime
    data.
 5. ```make clean``` will remove the executable and object files ```.o```,
    but not the data.

## Dependencies

To build this code, you must have installed
 * [GNU make][_make]
 * [CMake][_cmake]
 * [GNU compiler collection][_gcc]
 * [PNG library][_png]
 * [HTGS][_htgs]

These are usually available through the package manager. For example,
```apt-get install make libpng12-dev``` or
```yum install make libpng-devel```.

## Customization

The default input file ```../common-diffusion/params.txt``` defines key-value
pairs, with one pair per line. The two-character keys are predefined, and must
all be present. Descriptive comments follow the value on each line. If you wish
to change parameters (D, runtime, etc.), either modify ```params.txt``` in
place and ```make run```, or create your own copy of ```params.txt``` and
execute ```./diffusion <your_params.txt>```. The file name and extension make
no difference, so long as it contains plain text.

[_make]: https://www.gnu.org/software/make/
[_cmake]: https://cmake.org/download/
[_gcc]:  https://gcc.gnu.org
[_png]:  http://www.libpng.org/pub/png/libpng.html
[_htgs]: https://github.com/usnistgov/htgs
