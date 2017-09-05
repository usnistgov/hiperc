/**********************************************************************************
 This file is part of Phase-field Accelerator Benchmarks, written by Trevor Keller
 and available from https://github.com/usnistgov/phasefield-accelerator-benchmarks.

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
#include "boundaries.h"
#include "discretization.h"
#include "timer.h"
}

#include "cuda_kernels.cuh"

__constant__ fp_t d_mask[MAX_MASK_W * MAX_MASK_H];

__global__ void convolution_kernel(fp_t* conc_old, fp_t* conc_lap,
                                   int nx, int ny, int nm)
{
	int i, j, tx, ty,
	    dst_row, dst_col, dst_tile_w, dst_tile_h,
	    src_row, src_col, src_tile_w, src_tile_h;
	fp_t value=0.;

	/* source tile width includes the halo cells */
	src_tile_w = blockDim.x;
	src_tile_h = blockDim.y;

	/* destination tile width excludes the halo cells */
	dst_tile_w = src_tile_w - nm + 1;
	dst_tile_h = src_tile_h - nm + 1;

	/* determine indices on which to operate */
	tx = threadIdx.x;
	ty = threadIdx.y;

	dst_row = blockIdx.y * dst_tile_h + ty;
	dst_col = blockIdx.x * dst_tile_w + tx;

	src_row = dst_row - nm/2;
	src_col = dst_col - nm/2;

	/* copy tile: __shared__ gives access to all threads working on this tile	*/
	__shared__ fp_t conc_tile[MAX_TILE_H + MAX_MASK_H - 1][MAX_TILE_W + MAX_MASK_W - 1];

	if ((src_row >= 0) && (src_row < ny) &&
	    (src_col >= 0) && (src_col < nx)) {
		/* if src_row==0, then dst_row==nm/2: this is a halo row */
		conc_tile[ty][tx] = conc_old[src_row * nx + src_col];
	} else {
		/* points outside the halo should be switched off */
		conc_tile[ty][tx] = 0.;
	}

	/* tile data is shared: wait for all threads to finish copying */
	__syncthreads();

	/* compute the convolution */
	if (tx < dst_tile_w && ty < dst_tile_h) {
		for (j = 0; j < nm; j++) {
			for (i = 0; i < nm; i++) {
				value += d_mask[j * nm + i] * conc_tile[j+ty][i+tx];
			}
		}
		/* record value */
		if (dst_row < ny && dst_col < nx) {
			conc_lap[dst_row * nx + dst_col] = value;
		}
	}

	/* wait for all threads to finish writing */
	__syncthreads();
}

void compute_convolution(fp_t** conc_old, fp_t** conc_lap, fp_t** mask_lap,
                         int nx, int ny, int nm)
{
	/* If you must compute the convolution separately, do so here. */
	/* It is strongly recommended that you roll CUDA tasks into one function. */

	fp_t* d_conc_old, *d_conc_lap;

	/* allocate memory on device */
	cudaMalloc((void **) &d_conc_old, nx * ny * sizeof(fp_t));
	cudaMalloc((void **) &d_conc_lap, nx * ny * sizeof(fp_t));

	/* divide matrices into blocks of (MAX_TILE_W x MAX_TILE_W) threads */
	dim3 threads(MAX_TILE_W - nm/2, MAX_TILE_H - nm/2, 1);
	dim3 blocks(ceil(fp_t(nx)/threads.x)+1, ceil(fp_t(ny)/threads.y)+1, 1);

	/* transfer mask in to constant device memory */
	cudaMemcpyToSymbol(d_mask, mask_lap[0], nm * nm * sizeof(fp_t));

	/* transfer data from host in to device */
	cudaMemcpy(d_conc_old, conc_old[0], nx * ny * sizeof(fp_t),
	           cudaMemcpyHostToDevice);

	/* compute Laplacian */
	convolution_kernel<<<blocks,threads>>>(d_conc_old, d_conc_lap, nx, ny, nm);

	/* transfer from device out to host */
	cudaMemcpy(conc_lap[0], d_conc_lap, nx * ny * sizeof(fp_t),
	           cudaMemcpyDeviceToHost);

	/* free memory on device */
	cudaFree(d_conc_old);
	cudaFree(d_conc_lap);
}

__global__ void diffusion_kernel(fp_t* conc_old, fp_t* conc_new, fp_t* conc_lap,
                                 int nx, int ny, int nm, fp_t D, fp_t dt)
{
	int tx, ty, row, col;

	/* determine indices on which to operate */
	tx = threadIdx.x;
	ty = threadIdx.y;

	row = blockDim.y * blockIdx.y + ty;
	col = blockDim.x * blockIdx.x + tx;

	/* explicit Euler solution to the equation of motion */
	if (row < ny && col < nx) {
		conc_new[row * nx + col] = conc_old[row * nx + col]
		                         + dt * D * conc_lap[row * nx + col];
	}

	/* wait for all threads to finish writing */
	__syncthreads();
}

void solve_diffusion_equation(fp_t** conc_old, fp_t** conc_new, fp_t** conc_lap,
                              fp_t** mask_lap, int nx, int ny, int nm,
                              fp_t bc[2][2], fp_t D, fp_t dt, fp_t* elapsed,
                              struct Stopwatch* sw)
{
	fp_t* d_conc_old, *d_conc_new, *d_conc_lap;
	fp_t sum=0.;
	double start_time;

	/* divide matrices into blocks of (MAX_TILE_W x MAX_TILE_W) threads */
	dim3 threads(MAX_TILE_W - nm/2, MAX_TILE_H - nm/2, 1);
	dim3 blocks(ceil(fp_t(nx)/threads.x)+1, ceil(fp_t(ny)/threads.y)+1, 1);

	/* allocate memory on device */
	cudaMalloc((void **) &d_conc_old, nx * ny * sizeof(fp_t));
	cudaMalloc((void **) &d_conc_lap, nx * ny * sizeof(fp_t));
	cudaMalloc((void **) &d_conc_new, nx * ny * sizeof(fp_t));

	/* transfer data from host in to device */
	start_time = GetTimer();
	cudaMemcpy(d_conc_old, conc_old[0], nx * ny * sizeof(fp_t),
	           cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(d_bc, bc[0], 2 * 2 * sizeof(fp_t));
	cudaMemcpyToSymbol(d_mask, mask_lap[0], nm * nm * sizeof(fp_t));
	sw->file += GetTimer() - start_time;

	/* apply boundary conditions */
	boundary_kernel<<<blocks,threads>>>(d_conc_old, nx, ny, nm);

	/* compute Laplacian */
	start_time = GetTimer();
	convolution_kernel<<<blocks,threads>>>(d_conc_old, d_conc_lap, nx, ny, nm);
	sw->conv += GetTimer() - start_time;

	/* compute result */
	start_time = GetTimer();
	diffusion_kernel<<<blocks,threads>>>(d_conc_old, d_conc_new, d_conc_lap,
	                                     nx, ny, nm, D, dt);
	sw->step += GetTimer() - start_time;

	/* transfer from device out to host */
	start_time = GetTimer();
	cudaMemcpy(conc_new[0], d_conc_new, nx * ny * sizeof(fp_t),
	           cudaMemcpyDeviceToHost);
	sw->file += GetTimer() - start_time;

	/* free memory on device */
	cudaFree(d_conc_old);
	cudaFree(d_conc_lap);
	cudaFree(d_conc_new);

	*elapsed += dt;
}

__device__ fp_t d_euclidean_distance(fp_t ax, fp_t ay, fp_t bx, fp_t by)
{
	return sqrt((ax - bx) * (ax - bx) + (ay - by) * (ay - by));
}

__device__ fp_t d_distance_point_to_segment(fp_t ax, fp_t ay,
                                            fp_t bx, fp_t by,
                                            fp_t px, fp_t py)
{
	fp_t L2, t, zx, zy;

	L2 = (ax - bx) * (ax - bx) + (ay - by) * (ay - by);
	if (L2 == 0.) /* line segment is just a point */
		return d_euclidean_distance(ax, ay, px, py);
	t = fmax(0., fmin(1., ((px - ax) * (bx - ax)
	                     + (py - ay) * (by - ay)) / L2));
	zx = ax + t * (bx - ax);
	zy = ay + t * (by - ay);
	return d_euclidean_distance(px, py, zx, zy);
}

__device__ void d_analytical_value(fp_t x, fp_t t,
                                   fp_t D, fp_t bc[2][2], fp_t* c)
{
	*c = bc[1][0] * (1.0 - erf(x / sqrt(4.0 * D * t)));
}

__global__ void solution_kernel(fp_t* conc_new, fp_t* conc_lap, int nx, int ny,
                                fp_t dx, fp_t dy, int nm, fp_t elapsed, fp_t D,
                                fp_t bc[2][2])
{
	int tx, ty, row, col;
	fp_t r, cal, car, ca, cn;

	/* determine indices on which to operate */
	tx = threadIdx.x;
	ty = threadIdx.y;

	row = blockDim.y * blockIdx.y + ty;
	col = blockDim.x * blockIdx.x + tx;

	/* explicit Euler solution to the equation of motion */
	if (row > nm/2 && row < ny-1-nm/2 && col > nm/2 && col < nx-1-nm/2) {
		/* numerical solution */
		cn = conc_new[row * nx + col];

		/* shortest distance to left-wall source */
		r = d_distance_point_to_segment(dx * (nm/2), dy * (nm/2),
		                                dx * (nm/2), dy * (ny/2),
		                                dx * col, dy * row);
		d_analytical_value(r, elapsed, D, bc, &cal);

		/* shortest distance to right-wall source */
		r = d_distance_point_to_segment(dx * (nx-1-nm/2), dy * (ny/2),
		                                dx * (nx-1-nm/2), dy * (ny-1-nm/2),
		                                dx * col, dy * row);
		d_analytical_value(r, elapsed, D, bc, &car);

		/* superposition of analytical solutions */
		ca = cal + car;

		/* residual sum of squares (RSS) */
		conc_lap[row * nx + col] = (ca - cn) * (ca - cn)
		                         / (fp_t)((nx-1-nm/2) * (ny-1-nm/2));
	}

	/* wait for all threads to finish writing */
	__syncthreads();
}

void check_solution(fp_t** conc_new, fp_t** conc_lap, int nx, int ny,
                    fp_t dx, fp_t dy, int nm, fp_t elapsed, fp_t D,
                    fp_t bc[2][2], fp_t* rss)
{
	fp_t sum=0.;
	fp_t* d_conc_new, *d_conc_lap;

	/* divide matrices into blocks of (MAX_TILE_W x MAX_TILE_W) threads */
	dim3 threads(MAX_TILE_W - nm/2, MAX_TILE_H - nm/2, 1);
	dim3 blocks(ceil(fp_t(nx)/threads.x)+1, ceil(fp_t(ny)/threads.y)+1, 1);

	/* allocate memory on device */
	cudaMalloc((void **) &d_conc_new, nx * ny * sizeof(fp_t));
	cudaMalloc((void **) &d_conc_lap, nx * ny * sizeof(fp_t));

	/* transfer data from host in to device */
	cudaMemcpy(d_conc_new, conc_new[0], nx * ny * sizeof(fp_t),
	           cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(d_bc, bc[0], 2 * 2 * sizeof(fp_t));

	/* compute analytical solution */
	solution_kernel<<<blocks,threads>>>(d_conc_new, d_conc_lap,
	                                    nx, ny, dx, dy, nm, elapsed, D, bc);

	/* transfer from device out to host */
	cudaMemcpy(conc_lap[0], d_conc_lap, nx * ny * sizeof(fp_t),
	           cudaMemcpyDeviceToHost);

	/* perform parallel reduction of result */
	#pragma omp parallel for collapse(2) reduction(+:sum)
	for (int j = nm/2; j < ny-nm/2; j++) {
		for (int i = nm/2; i < nx-nm/2; i++) {
			sum += conc_lap[j][i];
		}
	}

	*rss = sum;

	/* free memory on device */
	cudaFree(d_conc_new);
	cudaFree(d_conc_lap);
}
