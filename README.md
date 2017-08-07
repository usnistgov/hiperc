# accelerator-testing
Diffusion and phase-field models for accelerator architectures

## Basic Algorithm
Diffusion and phase-field problems depend extensively on the divergence of gradients, or Laplacian operators:

&part;c/&part;t = D&nabla;&sup2;c &asymp; D(c&#8314; - 2c&#8304; + c&#8315;)/(h&sup2;)

This discretization is a special case of [convolution](https://en.wikipedia.org/wiki/Discrete_Laplace_operator#Image_Processing).

1D convolution kernel:

<table>
  <tr>
    <td>1</td>
    <td>-2</td>
    <td>1</td>
  </tr>
</table>

2D convolution kernel:

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

Accelerators are well-suited to the convolution of these kernels (or stencils) with input data matrices.

## Accelerator Languages

There are five mainstream approaches to shared-memory parallel programming,
with varying coding complexity and hardware dependencies:

 1. **OpenMP**: loop-level parallelism for multi-core CPU architectures.
    Simple to implement for SIMD programs, but with little opportunity for performance tuning.
    Implementation simply requires prefixing target loops with ```#pragma``` statements.
    Provided by all compilers and compatible with any hardware configuration.
 2. **POSIX threads**: MIMD-capable threading for multi-core CPU architectures.
    Challenging to properly implement, but with ample opportunity to tune performance.
    OpenMP is a specialized wrapper for POSIX threading for SIMD applications.
 3. **OpenACC**: loop-level massive parallelism for GPU architectures.
    Like OpenMP, implementation requires prefixing target code with ```#pragma``` statements,
    with little opportunity for performance tuning.
    Provided in compilers from Cray, PGI, and GNU;
    depends on a compatible graphics card and CUDA library installation.
 4. **CUDA**/**OpenCL**: general-purpose massive parallelism for GPGPU architectures.
    Like POSIX threading but for GPUs, provides low-level capabilities and ample opportunity for performance tuning.
    Requires a purpose-built compiler (nvcc, gpucc), libraries, and a compatible graphics card or accelerator.
 5. **Xeon Phi**: loop-level massive parallelism for RISC CPU-based accelerators,
    specifically the Intel Xeon Phi product line. Supports AVX-512 vectorized instructions.
    Only available through the Intel compiler, and requires Xeon Phi accelerator hardware.

Generically speaking, OpenMP and OpenAcc provide low barriers for entry into acceleration;
CUDA and Xeon Phi require high investments for hardware and compilers, but offer the greatest
capabilities for performance and optimization of a specific application.
Proof-of-concept GPU code can be run on Amazon's [HPC EC2](https://aws.amazon.com/ec2/Elastic-GPUs/), and
with a similar offering through [Rescale ScaleX](http://www.rescale.com/products/) for Xeon Phi.

## Hardware Cheatsheet
| System     | CPU                    | Threads | RAM   | GPU                         | Cores | Phi      | Cores    |
| :--------: | :--------------------: | ------: | ----: | :-------------------------: | ----: | :------: | -------: |
| Huginn     | Intel Xeon E5-1650 v3  | 12      | 64 GB | 1&times; NVIDIA Quadro K620 | 384   | &empty;  | &empty;  |
| rgpu2      | Intel Xeon E5-2697A v4 | 32      | 64 GB | 2&times; NVIDIA Tesla K40m  | 2880  | &empty;  | &empty;  |
| rgpu3      | Intel Xeon E5-2697A v4 | 32      | 64 GB | 2&times; NVIDIA Tesla K40m  | 2880  | &empty;  | &empty;  |
