/**********************************************************************************
 HiPerC: High Performance Computing Strategies for Boundary Value Problems
 written by Trevor Keller and available from https://github.com/usnistgov/hiperc

 This software was developed at the National Institute of Standards and Technology
 by employees of the Federal Government in the course of their official duties.
 Pursuant to title 17 section 105 of the United States Code this software is not
 subject to copyright protection and is in the public domain. NIST assumes no
 responsibility whatsoever for the use of this software by other parties, and makes
 no guarantees, expressed or implied, about its quality, reliability, or any other
 characteristic. We would appreciate acknowledgement if the software is used.

 This software can be redistributed and/or modified freely provided that any
 derivative works bear some notice that they are derived from it, and any modified
 versions bear some notice that they have been modified.

 Questions/comments to Trevor Keller (trevor.keller@nist.gov)
 **********************************************************************************/

/**
 \file  cuda_discretization.cu
 \brief Implementation of boundary condition functions with CUDA acceleration
*/

#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <cuda.h>

extern "C" {
#include "cuda_data.h"
#include "numerics.h"
}

#include "cuda_kernels.cuh"

__constant__ fp_t d_mask[MAX_MASK_W * MAX_MASK_H];

__global__ void convolution_kernel(fp_t* d_conc_old,
                                   fp_t* d_conc_lap,
                                   const int nx,
                                   const int ny,
                                   const int nm)
{
	int dst_x, dst_y, dst_nx, dst_ny;
	int src_x, src_y, src_nx, src_ny;
	int til_x, til_y, til_nx;
	fp_t value=0.;

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
	    src_y >= 0 && src_y < ny) {
		/* if src_y==0, then dst_y==nm/2: this is a halo row */
		d_conc_tile[til_nx * til_y + til_x] = d_conc_old[nx * src_y + src_x];
	}

	/* tile data is shared: wait for all threads to finish copying */
	__syncthreads();

	/* compute the convolution */
	if (til_x < dst_nx && til_y < dst_ny) {
		for (int j = 0; j < nm; j++) {
			for (int i = 0; i < nm; i++) {
				value += d_mask[j * nm + i] * d_conc_tile[til_nx * (til_y+j) + til_x+i];
			}
		}
		/* record value */
		if (dst_y < ny && dst_x < nx) {
			d_conc_lap[nx * dst_y + dst_x] = value;
		}
	}

	/* wait for all threads to finish writing */
	__syncthreads();
}

__global__ void diffusion_kernel(fp_t* d_conc_old,
                                 fp_t* d_conc_new,
                                 fp_t* d_conc_lap,
                                 const int nx,
                                 const int ny,
                                 const int nm,
                                 const fp_t D,
                                 const fp_t dt)
{
	int thr_x, thr_y, x, y;

	/* determine indices on which to operate */
	thr_x = threadIdx.x;
	thr_y = threadIdx.y;

	x = blockDim.x * blockIdx.x + thr_x;
	y = blockDim.y * blockIdx.y + thr_y;

	/* explicit Euler solution to the equation of motion */
	if (x < nx && y < ny) {
		d_conc_new[nx * y + x] = d_conc_old[nx * y + x]
		              + dt * D * d_conc_lap[nx * y + x];
	}

	/* wait for all threads to finish writing */
	__syncthreads();
}

void compute_convolution_tiled(fp_t* d_conc_old, fp_t* d_conc_lap,
                         const int nx, const int ny, const int nm,
                         const int bx, const int by)
{
  dim3 tile_size(bx, by, 1);
  dim3 num_tiles(ceil(float(nx) / (tile_size.x - nm + 1)),
                 ceil(float(ny) / (tile_size.y - nm + 1)),
                 1);
  size_t buf_size = (tile_size.x + nm) * (tile_size.y + nm) * sizeof(fp_t);

  convolution_kernel<<<num_tiles,tile_size,buf_size>>> (d_conc_old,
                                                        d_conc_lap,
                                                        nx, ny, nm);

}

void update_composition_tiled(fp_t* d_conc_old, fp_t* d_conc_lap, fp_t* d_conc_new,
                        const int nx, const int ny, const int nm,
                        const int bx, const int by, const fp_t D, const fp_t dt)
{
  dim3 tile_size(bx, by, 1);
  dim3 num_tiles(ceil(float(nx) / (tile_size.x - nm + 1)),
                 ceil(float(ny) / (tile_size.y - nm + 1)),
                 1);

  diffusion_kernel<<<num_tiles,tile_size>>> (d_conc_old,
                                             d_conc_new,
                                             d_conc_lap,
                                             nx, ny, nm, D, dt);	
}

void read_out_result(fp_t** conc, fp_t* d_conc, const int nx, const int ny)
{
  cudaMemcpy(conc[0], d_conc, nx * ny * sizeof(fp_t), cudaMemcpyDeviceToHost);
}

/**
 \brief Reference showing how to invoke the convolution kernel.

 A stand-alone function like this incurs the cost of host-to-device data
 transfer each time it is called: it is a teaching tool, not reusable code.
 It is the basis for cuda_diffusion_solver(), which achieves much better
 performance by bundling CUDA kernels together and intelligently managing
 data transfers between the host (CPU) and device (GPU).
*/
void standalone_convolution(fp_t** conc_old, fp_t** conc_lap, fp_t** mask_lap,
                         const int bx, const int by,
                         const int nm,
                         const int nx, const int ny)
{
	fp_t* d_conc_old, *d_conc_lap;

	/* allocate memory on device */
	cudaMalloc((void **) &d_conc_old, nx * ny * sizeof(fp_t));
	cudaMalloc((void **) &d_conc_lap, nx * ny * sizeof(fp_t));

	/* divide matrices into blocks of TILE_W * TILE_H threads */
	dim3 tile_size(bx,
	               by,
	               1);
	dim3 num_tiles(ceil(float(nx) / (tile_size.x - nm + 1)),
	               ceil(float(ny) / (tile_size.y - nm + 1)),
	               1);
	size_t buf_size = (tile_size.x + nm) * (tile_size.x + nm) * sizeof(fp_t);

	/* transfer mask in to constant device memory */
	cudaMemcpyToSymbol(d_mask, mask_lap[0], nm * nm * sizeof(fp_t));

	/* transfer data from host in to device */
	cudaMemcpy(d_conc_old, conc_old[0], nx * ny * sizeof(fp_t),
	           cudaMemcpyHostToDevice);

	/* compute Laplacian */
	convolution_kernel<<<num_tiles,tile_size,buf_size>>> (
		d_conc_old, d_conc_lap, nx, ny, nm
	);

	/* transfer from device out to host */
	cudaMemcpy(conc_lap[0], d_conc_lap, nx * ny * sizeof(fp_t),
	           cudaMemcpyDeviceToHost);

	/* free memory on device */
	cudaFree(d_conc_old);
	cudaFree(d_conc_lap);
}
