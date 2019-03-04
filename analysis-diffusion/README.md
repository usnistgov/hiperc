# Analysis of Diffusion Benchmark Problem

The analytical solution can be approximated using an error function:

$$
  c(x,y,t) = \mathrm{erfc}\left(\frac{x}{\sqrt{4Dt}}\right).
$$

For all implementations below, the residual was computed as the $L_2$ norm
of the difference between the numerical and analytical solutions. All
residuals exactly matched, indicating no effect of the implementation on
the computed result.

## Convolution Implementation Details

The core implementation difference lies in the convolution computation. For
ease of comparison, these functions are summarized below. The prototype for
all of these functions is

```c
void compute_convolution (fp_t ** conc_old,
                          fp_t ** conc_lap,
                          fp_t ** mask_lap,
                          const int nx,
                          const int ny,
                          const int nm );
```

### Serial Convolution on the CPU

```c
for ( int j = nm/2 ; j < ny-nm/2 ; j++) {
  for ( int i = nm/2 ; i < nx-nm/2 ; i++) {
    fp_t value = 0.0;
    for ( int mj = -nm/2 ; mj < nm/2 +1; mj++)
      for ( int mi = -nm/2 ; mi < nm/2 +1; mi++)
        value += mask_lap[mj + nm/2][mi + nm/2]
               * conc_old[j + mj][i + mi];
    conc_lap[j][i] = value ;
  }
}
```

### OpenMP Convolution on the CPU

```c
#pragma omp parallel
{
  # pragma omp for collapse (2)
  for ( int j = nm/2 ; j < ny-nm/2 ; j++) {
    for ( int i = nm/2 ; i < nx-nm/2 ; i++) {
      fp_t value = 0.0;
      for ( int mj = -nm/2 ; mj < nm/2 +1; mj++)
        for ( int mi = -nm/2 ; mi < nm/2 +1; mi++)
          value += mask_lap[mj + nm/2][mi + nm/2]
                * conc_old[j + mj][i + mi];
      conc_lap[j][i] = value ;
    }
  }
}
```

### Threading Building Blocks Convolution on the CPU

```c
/* Lambda function executed on each thread, solving convolution */
tbb::parallel_for(tbb::blocked_range2d<int>(nm/2, nx-nm/2, nm/2, ny-nm/2),
[=]( const tbb :: blocked_range2d < int >& r ) {
  for (int j = r.cols().begin(); j != r.cols().end(); j++) {
    for (int i = r.rows().begin(); i != r.rows().end(); i++) {
      fp_t value = 0.0;
      for (int mj = -nm/2; mj < nm/2 + 1; mj++)
        for (int mi = -nm/2; mi < nm/2 + 1; mi++)
          value += mask_lap[mj + nm/2] [mi + nm/2]
                 * conc_old[j + mj ] [i + mi ];
          conc_lap[j][i] = value ;
    }
  }
}
);
```

### OpenAcc Convolution on the GPU

```c
#pragma acc declare present (conc_old[0:ny][0:nx],
                             conc_lap[0:ny][0:nx],
                             mask_lap[0:nm][0:nm])

#pragma acc parallel
{
  # pragma acc loop collapse (2)
  for (int j = nm/2; j < ny-nm/2; j++) {
    for (int i = nm/2; i < nx-nm/2; i++) {
      fp_t value = 0.;
      # pragma acc loop seq collapse (2)
      for (int mj = -nm/2; mj < 1 + nm/2; mj++)
        for (int mi = -nm/2; mi < 1 + nm/2; mi++)
          value += mask_lap[mj + nm/2][mi + nm/2] 
                 * conc_old[j + mj][i + mi];
      conc_lap[j][i] = value;
    }
  }
}
```

## Tiled Convolution Details

The CUDA and OpenCL codes implement a tiled convolution algorithm:

![tiled-convolution][tiled-convolution]
**Fig:** *Schematic of Tiled GPU Convolution Algorithm*

* Copy the convolution stencil/kernel into constant cache, 64 KB max.
* Each block of threads allocates a shared tile, including a halo of
  points belonging to adjacent tiles.
* Each thread copies its grid entry into its shared tile entry.
* The convolution is computed, with drastically reduced cache misses.

### CUDA Convolution on the GPU

```c
__global__ void diffusion_kernel(fp_t * d_conc_old, fp_t * d_conc_new,
                                 fp_t * d_conc_lap,
                                 const int nx, const int ny, const int nm,
                                 const fp_t D, const fp_t dt )
{
  int dst_x, dst_y, dst_nx, dst_ny;
  int src_x, src_y, src_nx, src_ny;
  int til_x, til_y, til_nx;
  fp_t value = 0.;

  /* source and tile width include the halo cells */
  src_nx = blockDim.x;
  src_ny = blockDim.y;
  til_nx = src_nx;

  /* destination width excludes the halo cells */
  dst_nx = src_nx - nm + 1;
  dst_ny = src_ny - nm + 1;

  /* determine tile indices on which to operate */
  til_x = threadIdx.x;
  til_y = threadIdx.y;
  dst_x = blockIdx.x * dst_nx + til_x;
  dst_y = blockIdx.y * dst_ny + til_y;
  src_x = dst_x - nm/2;
  src_y = dst_y - nm/2;

  /* copy tile: __shared__ gives access to all threads working on this tile */
  extern __shared__ fp_t d_conc_tile[];
  if (src_x >= 0 && src_x < nx &&
      src_y >= 0 && src_y < ny ) {
    /* if src_y == 0, then dst_y == nm / 2: this is a halo row */
    d_conc_tile[til_nx * til_y + til_x] = d_conc_old[nx * src_y + src_x];
  }
  __syncthreads ();

  if (til_x < dst_nx && til_y < dst_ny) {
    for (int j = 0; j < nm ; j++)
      for (int i = 0; i < nm ; i++)
        value += d_mask[j * nm + i] 
               * d_conc_tile[til_nx * (til_y + j) + til_x + i];
    if ( dst_y < ny && dst_x < nx )
      d_conc_lap[nx * dst_y + dst_x ] = value ;
  }
  __syncthreads ();
}
```

### OpenCL Convolution on the GPU

```c
__kernel void convolution_kernel(__global fp_t * d_conc_old,
                                 __global fp_t * d_conc_lap,
                                 __constant fp_t * d_mask,
                                 __local fp_t * d_conc_tile,
                                 const int nx, const int ny, const int nm)
{
  const int src_ny = get_local_size(0);
  const int src_nx = get_local_size(1);
  const int til_nx = src_nx;
  const int dst_ny = src_ny - nm + 1;
  const int dst_nx = src_nx - nm + 1;
  const int til_x = get_local_id(0);
  const int til_y = get_local_id(1);
  const int dst_x = get_group_id(0) * dst_ny + til_x;
  const int dst_y = get_group_id(1) * dst_nx + til_y;
  const int src_x = dst_x - nm/2;
  const int src_y = dst_y - nm/2;

  if (src_x >= 0 && src_x < nx && src_y >= 0 && src_y < ny )
    d_conc_tile[til_nx * til_y + til_x] = d_conc_old[nx * src_y + src_x];
  barrier (CLK_LOCAL_MEM_FENCE);

  if (til_x < dst_ny && til_y < dst_nx) {
    fp_t value = 0.;
    for (int j = 0; j < nm ; j++)
      for (int i = 0; i < nm ; i++)
        value += d_mask[nm * j + i]
               * d_conc_tile[til_nx * (til_y + j) + til_x + i];
    if (dst_y < ny && dst_x < nx)
      d_conc_lap[nx * dst_y + dst_x] = value ;
  }
  barrier (CLK_LOCAL_MEM_FENCE);
}
```

## CPU Diffusion Results

![cpu-runtime-256][cpu256]
**Fig:** *Runtimes for 256×256 grid, CPU implementations*

![cpu-runtime-1024][cpu1024]
**Fig:** *Runtimes for 1024×1024 grid, CPU implementations*

## GPU Diffusion Results

![gpu-runtime-256][gpu256]
**Fig:** *Runtimes for 256×256 grid, GPU implementations*

![gpu-runtime-1024][gpu1024]
**Fig:** *Runtimes for 1024×1024 grid, GPU implementations*

## Summary of Results

![all-runtime-256][all256]
**Fig:** *Runtimes for 256×256 grid, all implementations*

![all-runtime-1024][all1024]
**Fig:** *Runtimes for 1024×1024 grid, all implementations*

### Runtime for Each Grid Size, seconds

| Implementation | $256^2$ Grid | $512^2$ Grid | $768^2$ Grid | $1024^2$ Grid |
| -------------- | ------------ | ------------ | ------------ | ------------- |
| CPU: Serial    | 18           | 244          | 1243         | 4600          |
| GPU: OpenAcc   | 12           | 118          | 537          | 1573          |
| CPU: TBB       | 7            | 39           | 153          | 436           |
| CPU: OpenMP    | 4            | 23           | 106          | 297           |
| GPU: OpenCL    | 3            | 7            | 30           | 74            |
| GPU: CUDA      | 2            | 7            | 25           | 67            |

### Speedup for Each Grid Size, Relative to OpenMP

| Implementation | $256^2$ Grid | $512^2$ Grid | $768^2$ Grid | $1024^2$ Grid |
| -------------- | ------------ | ------------ | ------------ | ------------- |
| CPU: Serial    | 0.2          | 0.1          | 0.1          | 0.1           |
| GPU: OpenAcc   | 0.3          | 0.2          | 0.2          | 0.2           |
| CPU: TBB       | 0.6          | 0.6          | 0.7          | 0.7           |
| CPU: OpenMP    | 1.0          | 1.0          | 1.0          | 1.0           |
| GPU: OpenCL    | 1.3          | 3.3          | 3.5          | 4.0           |
| GPU: CUDA      | 2.0          | 3.3          | 4.2          | 4.4           |


## Conclusions

* The diffusion equation is ripe for acceleration,
  particularly when the discretization is based on
  finite differences.
* OpenMP was the fastest CPU-based implementation.
  OpenMP is also easier to implement and debug than
  Threading Building Blocks, due to the use of lambda
  functions in the latter framework.
* OpenAcc was the slowest implementation, probably
  due to conservative scheduling of data transfers
  from system to GPU memory.
* CUDA edged out OpenCL just barely as the fastest
  GPU-based implementation. CUDA is also easier to
  implement and debug, since GPU kernels in OpenCL
  are provided as text files for just-in-time
  compiling.

<!-- Image Links -->

[tiled-convolution]: img/tiled-convolution.png "Tiled convolution sketch"

[cpu256]:  img/cpu-runtime-256.png  "Runtimes for 256x256 grid"
[cpu1024]: img/cpu-runtime-1024.png "Runtimes for 1024x1024 grid"

[gpu256]:  img/gpu-runtime-256.png  "Runtimes for 256x256 grid"
[gpu1024]: img/gpu-runtime-1024.png "Runtimes for 1024x1024 grid"

[all256]:  img/all-runtime-256.png  "Runtimes for 256x256 grid"
[all1024]: img/all-runtime-1024.png "Runtimes for 1024x1024 grid"
