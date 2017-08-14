# accelerator-testing

Diffusion and phase-field models for accelerator architectures

## Work in Progress

 - [ ] diffusion
   - [x] cpu
     - [x] serial
     - [x] OpenMP
     - [x] Threading Building Blocks
   - [ ] gpu
     - [ ] OpenACC
     - [ ] CUDA
   - [ ] phi
     - [ ] Knights Landing
 - [ ] spinodal
   - [ ] &middot;&middot;&middot;
 - [ ] ripening
   - [ ] &middot;&middot;&middot;

## Basic Algorithm

Diffusion and phase-field problems depend extensively on the divergence of gradients, *e.g.*
> &part;*c*/&part;*t* = &nabla;&middot;*D*&nabla;*c*

When *D* is constant, this simplifies to
> &part;*c*/&part;*t* = *D*&nabla;&sup2;*c*

In 1D, the Laplacian can be discretized:
> &part;*c*/&part;*t* &asymp; *D*(*c*&#8314; - 2*c*&#8304; + *c*&#8315;)/(*h*&sup2;)

This discretization is a special case of [convolution](https://en.wikipedia.org/wiki/Discrete_Laplace_operator#Image_Processing),
wherein a constant kernel of weighting coefficients is applied to an input dataset to produce a transformed output.

1D Laplacian convolution kernel:
<table>
  <tr>
    <td>1</td>
    <td>-2</td>
    <td>1</td>
  </tr>
</table>

2D Laplacian convolution kernel:
<table>
  <tr>
    <td>0</td>
    <td>1</td>
    <td>0</td>
  </tr>
  <tr>
    <td>1</td>
    <td>-4</td>
    <td>1</td>
  </tr>
  <tr>
    <td>0</td>
    <td>1</td>
    <td>0</td>
  </tr>
</table>

Accelerators and coprocessors are well-suited to this type of computation.

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
capabilities for performance and optimization of a specific application.
Proof-of-concept code for GPU and KNL can be run on [Amazon's EC2](https://aws.amazon.com/ec2/Elastic-GPUs/),
[Rescale's ScaleX](http://www.rescale.com/products/), and equivalent HPC cloud computing platforms.

### Disclaimer

Certain commercial entities, equipment, or materials may be identified in this
document in order to describe an experimental procedure or concept adequately.
Such identification is not intended to imply recommendation or endorsement by
the [National Institute of Standards and Technology (NIST)](http://www.nist.gov),
nor is it intended to imply that the entities, materials, or equipment are
necessarily the best available for the purpose.

## Contributions and Contact

Forks of this git repository are encouraged, and pull requests providing patches
or implementations are more than welcome.
Questions, concerns, and feedback regarding the source code provided in this git
repository should be addressed to trevor.keller@nist.gov (Trevor Keller).

### Hardware Cheatsheet

| System | CPU                    | Threads | RAM   | GPU                         | Cores | Phi      | Cores    |
| :----: | :--------------------: | ------: | ----: | :-------------------------: | ----: | :------: | -------: |
| Huginn | Intel Xeon E5-1650 v3  | 12      | 64 GB | 1&times; NVIDIA Quadro K620 | 384   | &empty;  | &empty;  |
| rgpu2  | Intel Xeon E5-2697A v4 | 32      | 64 GB | 2&times; NVIDIA Tesla K40m  | 2880  | &empty;  | &empty;  |
| rgpu3  | Intel Xeon E5-2697A v4 | 32      | 64 GB | 2&times; NVIDIA Tesla K40m  | 2880  | &empty;  | &empty;  |
