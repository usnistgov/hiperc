# phasefield-accelerator-benchmarks

Ever wonder if a GPU or Xeon Phi accelerator card would make your code faster?
Fast enough to justify the expense to your manager, adviser, or funding agency?
This project can help answer your questions!

## Work in Progress

 - [ ] diffusion
   - [x] cpu
     - [x] serial
     - [x] OpenMP
     - [x] Threading Building Blocks
   - [x] gpu
     - [x] OpenACC
     - [x] CUDA
   - [ ] phi
     - [ ] Knights Landing
 - [ ] spinodal
   - [ ] &middot;&middot;&middot;
 - [ ] ripening
   - [ ] &middot;&middot;&middot;

## Accelerator Languages

There are six mainstream approaches to shared-memory parallel programming,
with varying coding complexity and hardware dependencies:

 * **POSIX threads**: MIMD-capable threading for multi-core CPU architectures.
   Challenging to properly implement, but with ample opportunity to tune performance.
   Provided by all compilers and compatible with any hardware configuration.
 * **OpenMP**: loop-level parallelism for multi-core CPU architectures.
   Simple to implement for SIMD programs, but with little opportunity for performance tuning.
   Implementation simply requires prefixing target loops with ```#pragma``` statements.
   Provided by all compilers and compatible with any hardware configuration.
 * **Threading Building Blocks**: loop-level parallelism for multi-core CPU architectures
   using C++. Similar to OpenMP, but requires a wrapper around parallel regions that is
   more complicated than a one-line ```#pragma```. This provides more direct opportunities
   for performance tuning. Available as an open-source library.
 * **OpenACC**: loop-level massive parallelism for GPU architectures.
   Like OpenMP, implementation requires prefixing target code with ```#pragma``` statements,
   with little opportunity for performance tuning. Provided in compilers from Cray, PGI, and GNU;
   depends on a compatible graphics card, drivers, and CUDA library installation.
 * **CUDA**: general-purpose massive parallelism for GPGPU architectures.
   Like POSIX threading but for GPUs, provides low-level capabilities and ample opportunity
   for performance tuning. Requires a purpose-built compiler (nvcc, gpucc), libraries,
   and a compatible graphics card or accelerator.
 * **Xeon Phi**: low-level and loop-level massive parallelism for RISC CPU-based accelerators
   supporting AVX-512 vectorized instructions. Programmed like threaded CPU code,
   but with more opportunities for tuning and much greater performance.
   Only available through the Intel compiler, and requires Xeon Phi hardware.

Generically speaking, OpenMP and OpenACC provide low barriers for entry into acceleration;
CUDA and Xeon Phi require high investments for hardware and compilers, but offer the greatest
capabilities for performance and optimization of a specific application. CUDA hardware can be
emulated on the CPU using the [MCUDA framework](http://impact.crhc.illinois.edu/mcuda.aspx).
Proof-of-concept trials on GPU and KNL hardware can be run on [Amazon's EC2](
https://aws.amazon.com/ec2/Elastic-GPUs/), [Rescale's ScaleX](http://www.rescale.com/products/),
and equivalent HPC cloud computing platforms.

## Basic Algorithm

Diffusion and phase-field problems depend extensively on the divergence of gradients, *e.g.*
> &part;*c*/&part;*t* = &nabla;&middot;*D*&nabla;*c*

When *D* is constant, this simplifies to
> &part;*c*/&part;*t* = *D*&nabla;&sup2;*c*

This equation can be discretized, *e.g.* in 1D:
> &Delta;*c*/&Delta;*t* &asymp; *D*(*c*&#8314; - 2*c*&#8304; + *c*&#8315;)/(&Delta;*x*)&sup2;

This discretization is a special case of [convolution](https://en.wikipedia.org/wiki/Discrete_Laplace_operator#Image_Processing),
wherein a constant kernel of weighting coefficients is applied to an input dataset to produce a transformed output.

<table>
<tr>
  <td>
    <table>
      <caption>
        1D Laplacian
      </caption>
      <tr>
        <td> 1</td>
        <td>-2</td>
        <td> 1</td>
      </tr>
    </table>
  </td>
  <td>
    <table>
      <caption>
        2D Laplacian *
      </caption>
      <tr>
        <td>
          <table>
            <caption>
              5-point stencil
            </caption>
            <tr>
              <td> 0</td>
              <td> 1</td>
              <td> 0</td>
            </tr>
            <tr>
              <td> 1</td>
              <td>-4</td>
              <td> 1</td>
            </tr>
            <tr>
              <td> 0</td>
              <td> 1</td>
              <td> 0</td>
            </tr>
          </table>
        </td>
        <td>
          <table>
            <caption>
              9-point stencil
            </caption>
            <tr>
              <td>  1</td>
              <td>  4</td>
              <td>  1</td>
            </tr>
            <tr>
              <td>  4</td>
              <td>-20</td>
              <td>  4</td>
            </tr>
            <tr>
              <td>  1</td>
              <td>  4</td>
              <td>  1</td>
            </tr>
          </table>
        </td>
      </tr>
    </table>
  </td>
</table>

> \* The 9-point stencil uses first- and second-nearest neighbors. There is
another form that uses first- and third-nearest neighbors, which would be
straightforward to implement as a variation on this code.

In addition, computing values for the next timestep given values from the
previous timestep and the Laplacian values is a vector-add operation.
Accelerators and coprocessors are well-suited to this type of computation.
Therefore, to demonstrate the use of this hardware in materials science
applications, these examples flow according to the following pseudocode:
```
Start
  Allocate arrays using CPU
  Apply initial conditions to grid marked "old" using CPU
  While elapsed time is less than final time
  Do
    Apply boundary conditions using CPU
    Compute Laplacian using "old" values using accelerator
    Solve for "new" values using "old" and Laplacian values using accelerator
    Increment elapsed time by one timestep
    If elapsed time is an even increment of a specified interval
    Then
      Write an image file to disk
    Endif
  Done
  Write final values to disk in comma-separated value format
  Free arrays
Finish
```

## Using the Demonstration Codes

The example codes in this repository implement the same basic algorithm using
whichever of the mainstream accelerator programming methods apply. Running the
code on different parallel hardware configurations &mdash; CPU threading, GPU
offloading, and CPU coprocessing &mdash; provides a benchmark of these tools
using common computational materials science workloads. Comparing performance
against the serial baseline will help you make informed decisions about which
development pathways are appropriate for your scientific computing projects.
Note that the examples do not depend on a particular simulation framework:
dependencies are kept minimal, and the C functions are kept as simple as
possible to enhance readability for study and reusability in other codes.
The goal here is to learn how to use accelerators for materials science
simulations, not to enhance or promote any particular software package.

### Running the Codes

This repository is hierarchical, with the real codes of interest residing in
the lowest-ranked folders: ```cpu/serial```, ```cpu/openmp``` and
```gpu/cuda``` are expected to be the most useful. To compile, you may ```cd```
into each of these and run ```make```. If you wish to compile all the examples
for a particular hardware type, (```cpu```, ```gpu```, ```phi```), simply run
```make``` in that mid-level folder: ```make``` will be called recursively in
each of the child directories corresponding to different programming models
available for the hardware. If the executables build, *i.e.* ```make``` returns
without errors in either the middle or lowest levels, you can ```make run```
to execute the programs and gather data.

### What to Expect

As the solver marches along, an indicator will display the start time, progress,
and runtime in your terminal:

> Fri Aug 18 21:05:47 2017 [• • • • • • • • • • • • • • • • • • • •]  0h: 7m:15s

If the progress bar is not moving, or to check that the machine is working hard,
use a hardware monitoring tool. Here is a brief, definitely not comprehensive
list of options:
- **CPU**: any system monitor provided by your operating system will work. Look
  for CPU utilization greater than 100%, but moderate memory consumption. On
  GNU/Linux systems, [htop](http://hisham.hm/htop/) provides a rich interface
  to running processes in the terminal, which is helpful if you're running remotely.
- **GPU**: use a GPU monitor designed for your hardware. Some options include
  [nvidia-smi](https://developer.nvidia.com/nvidia-system-management-interface),
  [radeontop](https://github.com/clbr/radeontop), and
  [intel_gpu_top](https://github.com/ChrisCummins/intel-gpu-tools).
- **KNL**: the same monitor used for the CPU should also report load on the
  Knights Landing processor.

As it runs, the code will write a series of PNG image files (```diffusion.00?0000.png```)
in the same directory as the running executable resides; at the end, it will
write the final values to ```diffusion.0100000.csv```. It will also write a
summary file, ```runlog.csv```, containing the following columns:
 - **iter**: number of completed iterations
 - **sim_time**: elapsed simulation time (with &Delta;t=1, the first two columns are equal)
 - **wrss**: weighted sum-of-squares residual between the numerical values and analytical solution
 - **conv_time**: cumulative real time spent computing the Laplacian (convolution)
 - **step_time**: cumulative real time spent updating the composition (time-stepping)
 - **IO_time**: cumulative real time spent writing PNG files
 - **soln_time**: cumulative real time spent computing the analytical solution
 - **run_time**: elapsed real time

At timestep 10,000 the expected ```wrss=0.0029```; the rendered initial and final
images should look like these (grayscale, ```0``` is black and ```1``` is white):

| *t* = 0&middot;&Delta;*t*            | *t*=10000&middot;&Delta;*t*          |
| :----------------------------------: | :----------------------------------: |
| ![initial conc](diffusion.00000.png) | ![final conc](diffusion.10000.png)   |

The boundary conditions are fixed values of ```1``` along the lower-left half
and upper-right half walls, no flux everywhere else, with an initial value of
```0``` everywhere. These conditions represent a carburizing process, with
partial exposure (rather than the entire left and right walls) to produce an
inhomogeneous workload and highlight numerical errors at the boundaries.

If your simulation results do not look like this, or if your final ```wrss```
deviates from the expected value, something may be wrong with the installation,
hardware, or implementation. Please [file an issue](
https://github.com/usnistgov/phasefield-accelerator-benchmarks/issues) and share
what happened. You might have found a bug!

## Contributions and Contact

Forks of this git repository are encouraged, and pull requests providing patches
or implementations are more than welcome.
Questions, concerns, and feedback regarding the source code provided in this git
repository should be addressed to trevor.keller@nist.gov (Trevor Keller).

## Hardware Cheatsheet

| System | CPU                    | Threads | GPU                | Cores | Arch  | Phi     | Cores   |
| :----: | :--------------------: | ------: | :----------------: | ----: | :---: | :-----: | ------: |
| Huginn | Intel Xeon E5-1650 v3  | 12      | NVIDIA Quadro K620 | 384   | sm_50 | &empty; | &empty; |
| rgpu   | Intel Xeon E5620       | 16      | NVIDIA Tesla C2075 | 2880  | sm_20 | &empty; | &empty; |
| rgpu2  | Intel Xeon E5-2697A v4 | 32      | NVIDIA Tesla K80   | 2880  | sm_35 | &empty; | &empty; |
| rgpu3  | Intel Xeon E5-2697A v4 | 32      | NVIDIA Tesla K80   | 2880  | sm_35 | &empty; | &empty; |

## Disclaimer

Certain commercial entities, equipment, or materials may be identified in this
document in order to describe an experimental procedure or concept adequately.
Such identification is not intended to imply recommendation or endorsement by
the [National Institute of Standards and Technology (NIST)](http://www.nist.gov),
nor is it intended to imply that the entities, materials, or equipment are
necessarily the best available for the purpose.
