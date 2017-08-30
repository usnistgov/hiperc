# GPU code

This directory contains implementations of the diffusion equation for GPUs, with help from the following text:
> Kirk and Wu. *Programming Massively Parallel Processors: A Hands-On Approach,* 3 Ed. Morgan Kaufmann. New York: 2017.

## Working Code

 - [x] OpenACC
 - [x] CUDA

## Usage

This directory, and each sub-directory, contains a makefile with three important invocations:
 1. ```make``` will build the executable, named ```diffusion```, from its dependencies.
 2. ```make run``` will execute ```diffusion``` using the defaults listed in ```params.txt```,
    writing PNG and CSV output for inspection. ```runlog.csv``` contains the time-evolution of
    the weighted sum-of-squares residual from the analytical solution, as well as runtime data.
 3. ```make clean``` will remove the executable and object files ```.o```, but not the data.

To test the code, ```make run``` from this directory (```gpu```).

## Dependencies

To build this code, you must have installed
 * [GNU make](https://www.gnu.org/software/make/);
 * the [PNG library](http://www.libpng.org/pub/png/libpng.html);
 * the [CUDA toolkit](https://developer.nvidia.com/cuda-toolkit);
 * the [PGI compiler](http://www.pgroup.com/products/community.htm); 
 * the [OpenCL library](https://www.khronos.org/opencl/);
 * the OpenCL runtime for [AMD](http://developer.amd.com/tools-and-sdks/opencl-zone/),
   [NVIDIA](https://developer.nvidia.com/opencl), or
   [Intel](https://software.intel.com/en-us/articles/opencl-drivers) hardware.

```make``` and ```libpng``` can be installed through your operating system's
package manager, *e.g.* ```apt-get install make libpng12-dev```. The other
software (CUDA, OpenCL, and PGI compiler) should be installed using up-to-date
distributions from their websites since the packaged versions are often several
versions behind, and GPU hardware support evolves quickly. Note that CUDA is
not compatible with all GPU architectures. CUDA hardware can be emulated on the
CPU using the [MCUDA framework](http://impact.crhc.illinois.edu/mcuda.aspx).
Proof-of-concept trials on GPU hardware can be run on [Amazon's EC2](
https://aws.amazon.com/ec2/Elastic-GPUs/) and equivalent HPC cloud computing platforms.

### Disclaimer

Certain commercial entities, equipment, or materials may be identified in this
document in order to describe an experimental procedure or concept adequately.
Such identification is not intended to imply recommendation or endorsement by
the [National Institute of Standards and Technology (NIST)](http://www.nist.gov),
nor is it intended to imply that the entities, materials, or equipment are
necessarily the best available for the purpose.

## Source Layout

```
 gpu
 ├── cuda
 │   ├── boundaries.c
 │   ├── discretization.cu
 │   └── Makefile
 ├── openacc
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

The interface (prototypes for all functions) is defined in the top-level
```diffusion.h```. The mesh, output, and timer functions contain no specialized
code, and therefore ```mesh.c```, ```output.c```, and ```timer.c``` reside
alongside ```diffusion.h``` and ```main.c```. The implementation of boundary
conditions and discretized mathematics depend strongly on the parallelization
scheme, so each sub-directory contains specialized versions of ```boundaries.c```
and ```discretization.c```. When ```make``` is called, each ```.c``` file gets
compiled into an object ```.o``` in the sub-directory, allowing for different
compilers in each case. 

The default input file ```params.txt``` defines nine values via key-value pairs,
with one pair per line. The two-character keys are predefined, and must be one
of ```{nt, nx, ny, dx, dy, ns, nc, dc, co}```. Descriptive comments follow the
value on each line. If you wish to change parameters (D, runtime, etc.), either
modify ```params.txt``` in place and ```make run```, or create your own copy of
```params.txt``` and execute ```./diffusion newparams```. The file name and
extension make no difference, so long as it contains plain text.

If you read the ```Makefile```s, you will see that these GPU codes also invoke
OpenMP (via compiler flags ```-fopenmp``` or ```-mp```). This is because some
operations &mdash; namely array allocation and application of boundary
conditions &mdash; cannot be performed efficiently on the GPU, due to the high
expense of transferring data in and out compared to the small amount of work
to be done. These programs therefore implement an "OpenMP + X" programming
model, where CPU threading is used to modify a few values and GPU processing
is used to perform the real work.
