# OpenCL GPU code

implementation of the diffusion equation with
OpenMP threading and OpenCL acceleration

This directory contains implementations of the diffusion equation for GPUs.
The configuration of the OpenCL machinery in [opencl_data.c](opencl_data.c)
closely follows the example set by
> [OpenCLDiffusion][_ocld] from the repository
[A 2d diffusion model in OpenCL C][_scrblnrd3].

## Usage

This directory contains a makefile with three important invocations:
 1. ```make``` will build the executable, named ```diffusion```,
    from its dependencies.
 2. ```make run``` will execute ```diffusion``` using the defaults listed in
    ```../common-diffusion/params.txt```, writing PNG and CSV output for
    inspection. ```runlog.csv``` contains the time-evolution of the weighted
    sum-of-squares residual from the analytical solution, as well as runtime
    data.
 3. ```make clean``` will remove the executable and object files ```.o```,
    but not the data.

## Dependencies

To build this code, you must have installed
 * [GNU make][_make]
 * [PNG library][_png]
 * [OpenCL headers][_opencl]

This software can be installed through your operating system's package manager,
*e.g.* ```apt-get install make libpng12-dev opencl-headers```.
Proof-of-concept trials on GPU hardware can be run on [Amazon's EC2][_aws] and
equivalent HPC cloud computing platforms.

If you read the ```Makefile```s, you will see that this GPU code also depends on
OpenMP (provided by the compiler and invoked with the ```-fopenmp``` flag). This
is because some operations &mdash; namely array allocation and application of
boundary conditions &mdash; cannot be performed efficiently on the GPU, due to
the high expense of transferring data in and out compared to the small amount of
work to be done. This program therefore implements an "OpenMP + OpenCL"
programming model, where CPU threading is used to modify a few values and GPU
processing is used to perform the real work.

## Customization

The default input file ```../common-diffusion/params.txt``` defines key-value
pairs, with one pair per line. The two-character keys are predefined, and must
all be present. Descriptive comments follow the value on each line. If you wish
to change parameters (D, runtime, etc.), either modify ```params.txt``` in
place and ```make run```, or create your own copy of ```params.txt``` and
execute ```./diffusion <your_params.txt>```. The file name and extension make
no difference, so long as it contains plain text.

## Data Structures

OpenCL interacts with your machine's hardware through the following data structures:

 * Platform: OpenCL environment supporting a specific vendor's hardware, *e.g.*
             AMD OpenCL SDK, Intel OpenCL SDK, or Nvidia OpenCL SDK.
             Represented by ```cl_platform_id```.
 * Device: the CPU, GPU, FPGA, or other hardware managed by the Platform.
           Represented by ```cl_device_id```.
 * Context: a shared environment allowing threads and devices within the same
            Context to exchange data and perform collective operations.
            Represented by ```cl_context```.
 * Kernel: the fundamental tasks to be executed in the Context.
           Represented by ```cl_kernel```.
 * Program: the set of Kernels that can be run in the Context. Built from
            source files with ```.cl``` suffix, or from C strings written in
            the init_opencl() program.
            Represented by ```cl_program```.
 * Queue: the sequence of Programs to run on each Platform.
          Represented by ```cl_command_queue```.

### Disclaimer

Certain commercial entities, equipment, or materials may be identified in this
document in order to describe an experimental procedure or concept adequately.
Such identification is not intended to imply recommendation or endorsement by
the [National Institute of Standards and Technology (NIST)](http://www.nist.gov),
nor is it intended to imply that the entities, materials, or equipment are
necessarily the best available for the purpose.

[_amdcl]:     http://developer.amd.com/tools-and-sdks/opencl-zone/
[_aws]:       https://aws.amazon.com/ec2/Elastic-GPUs/
[_cuda]:      https://developer.nvidia.com/cuda-toolkit
[_icl]:       https://software.intel.com/en-us/articles/opencl-drivers
[_make]:      https://www.gnu.org/software/make/
[_mcuda]:     http://impact.crhc.illinois.edu/mcuda.aspx
[_nvcl]:      https://developer.nvidia.com/opencl
[_ocld]:      https://github.com/scrblnrd3/GPGPU-Diffusion/tree/master/OpenCLDiffusion/main.c
[_opencl]:    https://www.khronos.org/opencl/
[_pgi]:       http://www.pgroup.com/products/community.htm
[_png]:       http://www.libpng.org/pub/png/libpng.html
[_scrblnrd3]: https://github.com/scrblnrd3/GPGPU-Diffusion
