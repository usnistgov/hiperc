# Xeon Phi code

This directory contains implementations of the diffusion equation for KNL, with help from the following text:
> Jeffers, Reinders, and Sodani. *Intel Xeon Phi Processor High Performance Programming: Knights Landing Edition,* 2 Ed. Morgan Kaufmann. New York: 2016.

## Work in Progress

 - [ ] Knights Landing

## Usage

This directory, and each sub-directory, contains a makefile with three important invocations:
 1. ```make``` will build the executable, named ```diffusion```, from its dependencies.
 2. ```make run``` will execute ```diffusion``` using the defaults listed in ```params.txt```,
    writing PNG and CSV output for inspection. ```runlog.csv``` contains the time-evolution of
    the weighted sum-of-squares residual from the analytical solution, as well as runtime data.
 3. ```make clean``` will remove the executable and object files ```.o```, but not the data.

To test the code, ```make run``` from this directory (```phi```).

## Dependencies

To build this code, you must have installed
 * [GNU make](https://www.gnu.org/software/make/);
 * the [PNG library](http://www.libpng.org/pub/png/libpng.html);
 * the [Intel compiler](https://software.intel.com/en-us/intel-compilers).

These are usually available through the package manager. For example,
```apt-get install make libpng12-dev```. The Intel compiler is proprietary,
but available to open-source developers at no cost. Hardware capable of AVX-512
instructions, whether a primary Xeon CPU or a Xeon Phi, is also required.

Proof-of-concept KNL code can be run on [Rescale's ScaleX](http://www.rescale.com/products/)
and equivalent HPC cloud computing platforms.

### Disclaimer

Certain commercial entities, equipment, or materials may be identified in this
document in order to describe an experimental procedure or concept adequately.
Such identification is not intended to imply recommendation or endorsement by
the [National Institute of Standards and Technology (NIST)](http://www.nist.gov),
nor is it intended to imply that the entities, materials, or equipment are
necessarily the best available for the purpose.

## Source Layout

```
 phi
 ├── knl
 │   ├── boundaries.c
 │   ├── discretization.c
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
