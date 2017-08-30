# TBB CPU diffusion code

implementation of the diffusion equation for the
CPU with Threading Building Blocks

## Usage

This directory contains a makefile with three important invocations:
 1. ```make``` will build the executable, named ```diffusion```,
    from its dependencies.
 2. ```make run``` will execute ```diffusion``` using the defaults listed in
    ```../common_diffusion/params.txt```, writing PNG and CSV output for
    inspection. ```runlog.csv``` contains the time-evolution of the weighted
    sum-of-squares residual from the analytical solution, as well as runtime
    data.
 3. ```make clean``` will remove the executable and object files ```.o```,
    but not the data.

## Dependencies

To build this code, you must have installed
 * [GNU make][_make]
 * [GNU compiler collection][_gcc]
 * [PNG library][_png]
 * [Threading Building Blocks (TBB) library][_tbb]

These are usually available through the package manager. For example,
```apt-get install make libpng12-dev libtbb-dev``` or
```yum install make libpng-devel tbb-devel```.

## Customization

The default input file ```../common-diffusion/params.txt``` defines key-value
pairs, with one pair per line. The two-character keys are predefined, and must
all be present. Descriptive comments follow the value on each line. If you wish
to change parameters (D, runtime, etc.), either modify ```params.txt``` in
place and ```make run```, or create your own copy of ```params.txt``` and
execute ```./diffusion <your_params.txt>```. The file name and extension make
no difference, so long as it contains plain text.

[_make]: https://www.gnu.org/software/make/
[_gcc]:  https://gcc.gnu.org
[_png]:  http://www.libpng.org/pub/png/libpng.html
[_tbb]:  https://www.threadingbuildingblocks.org
