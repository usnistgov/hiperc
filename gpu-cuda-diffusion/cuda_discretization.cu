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
#include "boundaries.h"
#include "discretization.h"
#include "numerics.h"
#include "mesh.h"
#include "timer.h"
}

#include "cuda_kernels.cuh"

__constant__ fp_t d_mask[MAX_MASK_W * MAX_MASK_H];

__global__ void convolution_kernel(fp_t* d_conc_old,
                                   fp_t* d_conc_lap,
                                   const int nx,
                                   const int ny,
                                   const int nm)
{
	int i, j, tx, ty;
	int dst_row, dst_col, dst_cols, dst_rows;
	int src_row, src_col, src_cols, src_rows;
	fp_t value=0.;

	/* source tile width includes the halo cells */
	src_cols = blockDim.x;
	src_rows = blockDim.y;

	/* destination tile width excludes the halo cells */
	dst_cols = src_cols - nm + 1;
	dst_rows = src_rows - nm + 1;

	/* determine indices on which to operate */
	tx = threadIdx.x;
	ty = threadIdx.y;

	dst_col = blockIdx.x * dst_cols + tx;
	dst_row = blockIdx.y * dst_rows + ty;

	src_col = dst_col - nm/2;
	src_row = dst_row - nm/2;

	/* copy tile: __shared__ gives access to all threads working on this tile */
	__shared__ fp_t d_conc_tile[TILE_H + MAX_MASK_H - 1][TILE_W + MAX_MASK_W - 1];

	if (src_row >= 0 && src_row < ny &&
	    src_col >= 0 && src_col < nx) {
		/* if src_row==0, then dst_row==nm/2: this is a halo row */
		d_conc_tile[ty][tx] = d_conc_old[src_row * nx + src_col];
	} else {
		d_conc_tile[ty][tx] = 0.;
	}

	/* tile data is shared: wait for all threads to finish copying */
	__syncthreads();

	/* compute the convolution */
	if (tx < dst_cols && ty < dst_rows) {
		for (j = 0; j < nm; j++) {
			for (i = 0; i < nm; i++) {
				value += d_mask[j * nm + i] * d_conc_tile[j+ty][i+tx];
			}
		}
		/* record value */
		if (dst_row < ny && dst_col < nx) {
			d_conc_lap[dst_row * nx + dst_col] = value;
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
	int tx, ty, row, col;

	/* determine indices on which to operate */
	tx = threadIdx.x;
	ty = threadIdx.y;

	col = blockDim.x * blockIdx.x + tx;
	row = blockDim.y * blockIdx.y + ty;

	/* explicit Euler solution to the equation of motion */
	if (row < ny && col < nx) {
		d_conc_new[row * nx + col] = d_conc_old[row * nx + col]
		                  + dt * D * d_conc_lap[row * nx + col];
	}

	/* wait for all threads to finish writing */
	__syncthreads();
}

void cuda_diffusion_solver(struct CudaData* dev, fp_t** conc_new,
                           int nx, int ny, int nm, fp_t bc[2][2],
                           fp_t D, fp_t dt, int checks,
                           fp_t* elapsed, struct Stopwatch* sw)
{
	double start_time;
	int check=0;

	/* divide matrices into blocks of TILE_W *TILE_H threads */
	dim3 tile_size(TILE_W,
	               TILE_H,
	               1);
	dim3 num_tiles(ceil(float(nx) / (tile_size.x - nm + 1)),
	               ceil(float(ny) / (tile_size.y - nm + 1)),
	               1);

	for (check = 0; check < checks; check++) {
		/* apply boundary conditions */
		boundary_kernel<<<num_tiles,tile_size>>> (
			dev->conc_old, nx, ny, nm
		);

		/* compute Laplacian */
		start_time = GetTimer();
		convolution_kernel<<<num_tiles,tile_size>>> (
			dev->conc_old, dev->conc_lap, nx, ny, nm
		);
		sw->conv += GetTimer() - start_time;

		/* compute result */
		start_time = GetTimer();
		diffusion_kernel<<<num_tiles,tile_size>>> (
			dev->conc_old, dev->conc_new, dev->conc_lap, nx, ny, nm, D, dt
		);
		sw->step += GetTimer() - start_time;

		swap_pointers_1D(&(dev->conc_old), &(dev->conc_new));

		*elapsed += dt;
	}
	/* after swap, new data is in dev->conc_old */


	/* transfer from device out to host (conc_new) */
	start_time = GetTimer();
	cudaMemcpy(conc_new[0], dev->conc_old, nx * ny * sizeof(fp_t),
	           cudaMemcpyDeviceToHost);
	sw->file += GetTimer() - start_time;
}

void check_solution(fp_t** conc_new, fp_t** conc_lap, int nx, int ny,
                    fp_t dx, fp_t dy, int nm, fp_t elapsed, fp_t D,
                    fp_t bc[2][2], fp_t* rss)
{
	fp_t sum=0.;

	#pragma omp parallel reduction(+:sum)
	{
		int i, j;
		fp_t r, cal, car, ca, cn;

		#pragma omp for collapse(2) private(ca,cal,car,cn,i,j,r)
		for (j = nm/2; j < ny-nm/2; j++) {
			for (i = nm/2; i < nx-nm/2; i++) {
				/* numerical solution */
				cn = conc_new[j][i];

				/* shortest distance to left-wall source */
				r = distance_point_to_segment(dx * (nm/2), dy * (nm/2),
				                              dx * (nm/2), dy * (ny/2),
				                              dx * i, dy * j);
				analytical_value(r, elapsed, D, bc, &cal);

				/* shortest distance to right-wall source */
				r = distance_point_to_segment(dx * (nx-1-nm/2), dy * (ny/2),
				                              dx * (nx-1-nm/2), dy * (ny-1-nm/2),
				                              dx * i, dy * j);
				analytical_value(r, elapsed, D, bc, &car);

				/* superposition of analytical solutions */
				ca = cal + car;

				/* residual sum of squares (RSS) */
				conc_lap[j][i] = (ca - cn) * (ca - cn) / (fp_t)((nx-1-nm/2) * (ny-1-nm/2));
			}
		}

		#pragma omp for collapse(2) private(i,j)
		for (j = nm/2; j < ny-nm/2; j++) {
			for (i = nm/2; i < nx-nm/2; i++) {
				sum += conc_lap[j][i];
			}
		}
	}

	*rss = sum;
}

void compute_convolution(fp_t** conc_old, fp_t** conc_lap, fp_t** mask_lap,
                         int nx, int ny, int nm)
{
	/* If you must compute the convolution separately, do so here.
	 * It is strongly recommended that you roll CUDA tasks into one function:
	 * This legacy function is included to show basic usage of the kernel.
	 */

	fp_t* d_conc_old, *d_conc_lap;
	const int margin = 0; /* nm - 1; */

	/* allocate memory on device */
	cudaMalloc((void **) &d_conc_old, nx * ny * sizeof(fp_t));
	cudaMalloc((void **) &d_conc_lap, nx * ny * sizeof(fp_t));

	/* divide matrices into blocks of TILE_W * TILE_H threads */
	dim3 tile_size(TILE_W,
	               TILE_H,
	               1);
	dim3 num_tiles(ceil(float(nx) / (tile_size.x - nm + 1)),
	               ceil(float(ny) / (tile_size.y - nm + 1)),
	               1);

	/* transfer mask in to constant device memory */
	cudaMemcpyToSymbol(d_mask, mask_lap[0], nm * nm * sizeof(fp_t));

	/* transfer data from host in to device */
	cudaMemcpy(d_conc_old, conc_old[0], nx * ny * sizeof(fp_t),
	           cudaMemcpyHostToDevice);

	/* compute Laplacian */
	convolution_kernel<<<num_tiles,tile_size>>> (
		d_conc_old, d_conc_lap, nx, ny, nm
	);

	/* transfer from device out to host */
	cudaMemcpy(conc_lap[0], d_conc_lap, nx * ny * sizeof(fp_t),
	           cudaMemcpyDeviceToHost);

	/* free memory on device */
	cudaFree(d_conc_old);
	cudaFree(d_conc_lap);
}
