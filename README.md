# phasefield-accelerator-benchmarks

Ever wonder if a GPU or Xeon Phi accelerator card would make your code faster?
Fast enough to justify the expense to your manager, adviser, or funding agency?
This project can help answer your questions!

[![Documentation on readthedocs][doc_img]][doc_lnk] [![Conversation on Gitter][chat_img]][chat_lnk]

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

## Accelerator Languages

There are six mainstream approaches to shared-memory parallel programming,
with varying coding complexity and hardware dependencies:

 * **POSIX threads**: [MIMD][_mimd]-capable threading for multi-core CPU
   architectures. Challenging to properly implement, but with ample opportunity
   to tune performance. Provided by all compilers and compatible with any
   hardware configuration.
 * [**OpenMP**][_omp]: loop-level parallelism for multi-core CPU architectures.
   Simple to implement for [SIMD][_simd] programs, but with little opportunity
   for performance tuning. Implementation simply requires prefixing target
   loops with ```#pragma``` directives. Provided by all compilers and
   compatible with any hardware configuration.
 * [**Threading Building Blocks**][_tbb]: loop-level parallelism for multi-core
   CPU architectures using C++. Similar to [OpenMP][_omp], but requires a
   wrapper around parallel regions that is more complicated than a one-line
   ```#pragma```. This provides more direct opportunities for performance
   tuning. Available as an open-source library.
 * [**OpenACC**][_acc]: loop-level massive parallelism for GPU architectures.
   Like [OpenMP][_omp], implementation requires prefixing target code with
   ```#pragma``` directives, with little opportunity for performance tuning.
   Provided in compilers from [Cray][_cray], [PGI][_pgi], and [GNU][_gnu];
   depends on a compatible graphics card, drivers, and [CUDA][_cuda] or
   [OpenCL][_ocl] library installation.
 * [**CUDA**][_cuda]: general-purpose massive parallelism for GPU architectures.
   Like POSIX threading but for GPUs, provides low-level capabilities and ample
   opportunity for performance tuning. Requires a purpose-built compiler (nvcc,
   gpucc), libraries, and a compatible graphics card or accelerator.
 * [**Xeon Phi**][_phi]: low-level and loop-level massive parallelism for
   [ccNUMA][_ccn] multicore CPU-based accelerators supporting AVX-512
   vectorized instructions. Programmed like threaded CPU code, but with more
   opportunities for tuning and much greater performance. Only available
   through the Intel compiler, and requires Xeon Phi hardware.

Generically speaking, [OpenMP][_omp] and [OpenACC][_acc] provide low barriers
for entry into acceleration; [CUDA][_cuda] and [Xeon Phi][_phi] require high
investments for hardware and compilers, but offer the greatest capabilities for
performance and optimization of a specific application. CUDA hardware can be
emulated on the CPU using the [MCUDA framework][_mcuda]. Proof-of-concept
trials on GPU and KNL hardware can be run on [Amazon's EC2][_ec2],
[Rescale's ScaleX][_scalex], and equivalent HPC cloud computing platforms.

## Basic Algorithm

Diffusion and phase-field problems depend extensively on the divergence of
gradients, *e.g.*
> &part;*c*/&part;*t* = &nabla;&middot;*D*&nabla;*c*

When *D* is constant, this simplifies to
> &part;*c*/&part;*t* = *D*&nabla;&sup2;*c*

This equation can be discretized, *e.g.* in 1D:
> &Delta;*c*/&Delta;*t* &asymp; *D*(*c*&#8314; - 2*c*&#8304; + *c*&#8315;)/(&Delta;*x*)&sup2;

This discretization is a special case of [convolution][_conv], wherein a
constant kernel of weighting coefficients is applied to an input dataset to
produce a transformed output.

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

> \* This canonical 9-point (3&times;3) stencil uses first- and
second-nearest neighbors. There is a 9-point (4&times;4) form that uses first-
and third-nearest neighbors, which is also implemented in the source code;
it is less efficient than the canonical form.

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

### Source Code Documentation

You are encouraged to browse the source for this project to see how it works.
This project is documented using [Doxygen][_doxy], which can help guide you
through the source code layout and intent. This guide is included as
[```phasefield-accelerator-benchmarks_guide.pdf```][_doc]. To build the
documentation yourself, with [Doxygen][_doxy], [LaTeX][_latex], and
[Make][_make] installed, ```cd``` into ```doc``` and run ```make```. Then
browse the source code to your heart's content.

## Running the Demonstration Programs

This repository has a flat structure. Code common to each problem type are
lumped together, *e.g.* in ```common-diffusion```. The remaining implementation
folders have three-part names: ```architecture-threading-model```. To compile
code for your setup of interest, ```cd``` into its directory and run ```make```
(note that this will not work in the ```common``` folders). If the executable
builds, *i.e.* ```make``` returns without errors, you can ```make run```
to execute the program and gather timing data. If you wish to attempt building
or running all the example codes, execute ```make``` or ```make run``` from
this top-level directory: it will recursively call the corresponding ```make```
command in every sub-directory.

### What to Expect

As the solver marches along, an indicator will display the start time, progress,
and runtime in your terminal:

> Fri Aug 18 21:05:47 2017 [• • • • • • • • • • • • • • • • • • • •]  0h: 7m:15s

If the progress bar is not moving, or to check that the machine is working hard,
use a hardware monitoring tool. Here is a brief, definitely not comprehensive
list of options:
- **CPU**: any system monitor provided by your operating system will work. Look
  for CPU utilization greater than 100%, but moderate memory consumption. On
  GNU/Linux systems, [htop][_htop] provides a rich interface
  to running processes in the terminal, which is helpful if you're running remotely.
- **GPU**: use a GPU monitor designed for your hardware. Some options include
  [nvidia-smi][_nvsmi], [radeontop][_amdtop], and [intel_gpu_top][_inteltop].
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

At timestep 10,000 the expected ```wrss=0.002895``` using the 5-point stencil;
the rendered initial and final images should look like these (grayscale,
```0``` is black and ```1``` is white):

| *t* = 0&middot;&Delta;*t*           | *t*=10000&middot;&Delta;*t*     |
| :---------------------------------: | :-----------------------------: |
| ![initial conc][_initial_diffusion] | ![final conc][_final_diffusion] |

The boundary conditions are fixed values of ```1``` along the lower-left half
and upper-right half walls, no flux everywhere else, with an initial value of
```0``` everywhere. These conditions represent a carburizing process, with
partial exposure (rather than the entire left and right walls) to produce an
inhomogeneous workload and highlight numerical errors at the boundaries.

If your compiler returns warnings or errors, your simulation results do not look
like this, or if your final ```wrss``` deviates from the expected value,
something may be wrong with the installation, hardware, or implementation.
Please [file an issue][_issues] and share what happened.
You probably found a bug!

## Reusing the Demonstration Code

The flat file structure is intended to make it easy for you to extract code
for modification and reuse in your research code. To do so, copy the three-part
folder corresponding to your setup of interest, *e.g.* ```gpu-cuda-diffusion```,
to another location (outside this repository). Then copy the contents of the
common folder it depends upon, *e.g.* ```common-diffusion```, into the new
folder location. Finally, edit the ```Makefile``` within the new folder to
remove references to the old common folder. This should centralize everything
you need to remix and get started in the new folder.

## Work in Progress

 - [ ] cpu
   - [x] analytical
     - [x] diffusion
   - [ ] serial
     - [x] diffusion
     - [ ] spinodal
     - [ ] ripening
   - [ ] OpenMP
     - [x] diffusion
     - [ ] spinodal
     - [ ] ripening
   - [ ] Threading Building Blocks
     - [x] diffusion
     - [ ] spinodal
     - [ ] ripening
 - [ ] gpu
   - [ ] CUDA
     - [x] diffusion
     - [ ] spinodal
     - [ ] ripening
   - [ ] OpenACC
     - [x] diffusion
     - [ ] spinodal
     - [ ] ripening
   - [ ] OpenCL
     - [ ] diffusion
     - [ ] spinodal
     - [ ] ripening
 - [ ] phi
   - [ ] OpenMP
     - [ ] diffusion
     - [ ] spinodal
     - [ ] ripening

## Contributions and Contact

Forks of this git repository are encouraged, and pull requests providing patches
or implementations are more than welcome.
Questions, concerns, and feedback regarding this source code should be addressed
to trevor.keller@nist.gov (Trevor Keller), or [filed as an issue][_issues].

## Disclaimer

Certain commercial entities, equipment, or materials may be identified in this
document in order to describe an experimental procedure or concept adequately.
Such identification is not intended to imply recommendation or endorsement by
the [National Institute of Standards and Technology (NIST)][_NIST], nor is it
intended to imply that the entities, materials, or equipment are necessarily
the best available for the purpose.

[_NIST]:     http://www.nist.gov
[_doc]:      doc/phasefield-accelerator-benchmarks_guide.pdf
[_initial_diffusion]: common-diffusion/diffusion.00000.png
[_final_diffusion]:   common-diffusion/diffusion.10000.png
[doc_img]:   https://readthedocs.org/projects/phasefield-accelerator-benchmarks/badge/?version=latest
[doc_lnk]:   http://phasefield-accelerator-benchmarks.readthedocs.io/en/latest/?badge=latest
[chat_img]:  https://badges.gitter.im/phasefield-accelerator-benchmarks/Lobby.svg
[chat_lnk]:  https://gitter.im/phasefield-accelerator-benchmarks/Lobby?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge
[_issues]:   https://github.com/usnistgov/phasefield-accelerator-benchmarks/issues
[_acc]:      https://www.openacc.org/
[_amdtop]:   https://github.com/clbr/radeontop
[_ccn]:      https://en.wikipedia.org/wiki/Non-uniform_memory_access#Cache_coherent_NUMA
[_conv]:     https://en.wikipedia.org/wiki/Discrete_Laplace_operator#Image_Processing
[_cray]:     http://www.cray.com/
[_cuda]:     https://developer.nvidia.com/cuda-zone
[_doxy]:     http://www.stack.nl/~dimitri/doxygen/
[_ec2]:      https://aws.amazon.com/ec2/Elastic-GPUs/
[_gnu]:      https://gcc.gnu.org/
[_htop]:     http://hisham.hm/htop/
[_inteltop]: https://github.com/ChrisCummins/intel-gpu-tools
[_latex]:    https://www.latex-project.org/
[_make]:     https://www.gnu.org/software/make/
[_mcuda]:    http://impact.crhc.illinois.edu/mcuda.aspx
[_mimd]:     https://en.wikipedia.org/wiki/MIMD
[_nvsmi]:    https://developer.nvidia.com/nvidia-system-management-interface
[_ocl]:      https://www.khronos.org/opencl/
[_omp]:      http://www.openmp.org/
[_pgi]:      http://www.pgroup.com/
[_phi]:      https://www.intel.com/content/www/us/en/products/processors/xeon-phi/xeon-phi-processors.html
[_scalex]:   http://www.rescale.com/products/
[_simd]:     https://en.wikipedia.org/wiki/SIMD
[_tbb]:      https://www.threadingbuildingblocks.org/
