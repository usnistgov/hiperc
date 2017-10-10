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
	int i, j, y;
	int dst_til_y, dst_til_x, dst_til_nx, dst_til_ny;
	int src_til_y, src_til_x, src_til_nx, src_til_ny;
	int sha_til_x, sha_til_y, sha_til_nx;
	fp_t value=0.;

	/* source and shared tile width include the halo cells */
	src_til_nx = blockDim.x;
	src_til_ny = blockDim.y;
	sha_til_nx = src_til_nx;

	/* destination tile width excludes the halo cells */
	dst_til_nx = src_til_nx - nm + 1;
	dst_til_ny = src_til_ny - nm + 1;

	/* determine indices on which to operate */
	sha_til_x = threadIdx.x;
	sha_til_y = threadIdx.y;

	dst_til_x = blockIdx.x * dst_til_nx + sha_til_x;
	dst_til_y = blockIdx.y * dst_til_ny + sha_til_y;

	src_til_x = dst_til_x - nm/2;
	src_til_y = dst_til_y - nm/2;

	/* copy tile: __shared__ gives access to all threads working on this tile */
	extern __shared__ fp_t d_conc_tile[];

	if (src_til_y >= 0 && src_til_y < ny &&
	    src_til_x >= 0 && src_til_x < nx) {
		/* if src_til_y==0, then dst_til_y==nm/2: this is a halo row */
		d_conc_tile[sha_til_y * sha_til_nx + sha_til_x] = d_conc_old[src_til_y * nx + src_til_x];
	} else {
		d_conc_tile[sha_til_y * sha_til_nx + sha_til_x] = 0.;
	}

	/* tile data is shared: wait for all threads to finish copying */
	__syncthreads();

	/* compute the convolution */
	if (sha_til_x < dst_til_nx && sha_til_y < dst_til_ny) {
		for (j = 0; j < nm; j++) {
			y = (j+sha_til_y) * sha_til_nx;
			for (i = 0; i < nm; i++) {
				value += d_mask[j * nm + i] * d_conc_tile[y + i + sha_til_x];
			}
		}
		/* record value */
		if (dst_til_y < ny && dst_til_x < nx) {
			d_conc_lap[dst_til_y * nx + dst_til_x] = value;
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
		d_conc_new[y * nx + x] = d_conc_old[y * nx + x]
		                  + dt * D * d_conc_lap[y * nx + x];
	}

	/* wait for all threads to finish writing */
	__syncthreads();
}

void cuda_diffusion_solver(struct CudaData* dev, fp_t** conc_new,
                           fp_t bc[2][2], int bx, int by,
                           int nm, int nx, int ny,
                           fp_t D, fp_t dt, int checks,
                           fp_t* elapsed, struct Stopwatch* sw)
{
	double start_time;
	int check=0;

	/* divide matrices into blocks of (TILE_W x TILE_H) threads */

	dim3 tile_dim(bx - nm + 1, by - nm + 1, 1);
	dim3 bloc_dim(nm - 1 + floor(float(nx)/tile_dim.x), nm - 1 + floor(float(ny)/tile_dim.y), 1);
	size_t tile_size = bx * by;

	for (check = 0; check < checks; check++) {
		/* apply boundary conditions */
		boundary_kernel<<<bloc_dim,tile_dim>>> (
			dev->conc_old, nx, ny, nm
		);

		/* compute Laplacian */
		start_time = GetTimer();
		convolution_kernel<<<bloc_dim,tile_dim,tile_size>>> (
			dev->conc_old, dev->conc_lap, nx, ny, nm
		);
		sw->conv += GetTimer() - start_time;

		/* compute result */
		start_time = GetTimer();
		diffusion_kernel<<<bloc_dim,tile_dim>>> (
			dev->conc_old, dev->conc_new, dev->conc_lap, nx, ny, nm, D, dt
		);
		sw->step += GetTimer() - start_time;

		swap_pointers_1D(&(dev->conc_old), &(dev->conc_new));
	}
	/* after swap, new data is in dev->conc_old */

	*elapsed += dt * checks;

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
                         int bx, int by,
                         int nm,
                         int nx, int ny)
{
	/* If you must compute the convolution separately, do so here.
	 * It is strongly recommended that you roll CUDA tasks into one function:
	 * This legacy function is included to show basic usage of the kernel.
	 */

	fp_t* d_conc_old, *d_conc_lap;

	/* allocate memory on device */
	cudaMalloc((void **) &d_conc_old, nx * ny * sizeof(fp_t));
	cudaMalloc((void **) &d_conc_lap, nx * ny * sizeof(fp_t));

	/* divide matrices into blocks of (TILE_W x TILE_W) threads */
	/* dim3 tile_dim(TILE_W - nm/2, TILE_H - nm/2, 1); */
	/* dim3 bloc_dim(ceil(fp_t(nx)/tile_dim.x)+1, ceil(fp_t(ny)/tile_dim.y)+1, 1); */
	dim3 tile_dim(bx - nm +1, by - nm + 1, 1);
	dim3 bloc_dim(nm - 1 + floor(float(nx)/tile_dim.x), nm - 1 + floor(float(ny)/tile_dim.y), 1);
	size_t tile_size = bx * by;

	/* transfer mask in to constant device memory */
	cudaMemcpyToSymbol(d_mask, mask_lap[0], nm * nm * sizeof(fp_t));

	/* transfer data from host in to device */
	cudaMemcpy(d_conc_old, conc_old[0], nx * ny * sizeof(fp_t),
	           cudaMemcpyHostToDevice);

	/* compute Laplacian */
	convolution_kernel<<<bloc_dim,tile_dim,tile_size>>> (
		d_conc_old, d_conc_lap, nx, ny, nm
	);

	/* transfer from device out to host */
	cudaMemcpy(conc_lap[0], d_conc_lap, nx * ny * sizeof(fp_t),
	           cudaMemcpyDeviceToHost);

	/* free memory on device */
	cudaFree(d_conc_old);
	cudaFree(d_conc_lap);
}
