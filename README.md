# accelerator-testing
Diffusion and phase-field models for accelerator architectures

## Hardware Cheatsheet
| System     | CPU                    | Threads | RAM   | GPU                         | Cores | Phi      | Cores    |
| :--------: | :--------------------: | ------: | ----: | :-------------------------: | ----: | :------: | -------: |
| Huginn     | Intel Xeon E5-1650 v3  | 12      | 64 GB | 1&times; NVIDIA Quadro K620 | 384   | &empty;  | &empty;  |
| rgpu2      | Intel Xeon E5-2697A v4 | 32      | 64 GB | 2&times; NVIDIA Tesla K40m  | 2880  | &empty;  | &empty;  |
| rgpu3      | Intel Xeon E5-2697A v4 | 32      | 64 GB | 2&times; NVIDIA Tesla K40m  | 2880  | &empty;  | &empty;  |

## Basic Algorithm
Diffusion and phase-field problems depend extensively on the divergence of gradients, or Laplacian operators:

&part;c/&part;t = D&nabla;&sup2;c &asymp; D(c&#8314; -2c&#8304; + c&#8315;)/(h&sup2;)

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
    <td>-2</td>
    <td>1</td>
  </tr>
  <tr>
    <td>0</td>
    <td>1</td>
    <td>0</td>
  </tr>
</table>

Accelerators are well-suited to the convolution of these kernels (or stencils) with input data matrices.
