/*
	File: discretization.c
	Role: implementation of discretized mathematical operations with OpenMP threading and CUDA acceleration

	Questions/comments to trevor.keller@nist.gov
	Bugs/requests to https://github.com/tkphd/accelerator-testing
*/

#include <math.h>
#include <omp.h>
#ifndef MCUDA
#include <cuda.h>
#include <cutil.h>
#else
#include <mcuda.h>
#endif

#include "diffusion.h"

/* CUDA allocates memory tiles on the GPU statically, so their sizes must be hard coded */
#define TILE_W 32
#define TILE_H 32

void set_threads(int n)
{
	omp_set_num_threads(n);
}

/* round a/b up to the next integer */
inline int iDivUp(int a, int b){
	return (a % b != 0) ? (a / b + 1) : (a / b);
}

void set_mask(double dx, double dy, int* nm, double** M)
{
	/* M is initialized to zero, so corners can be ignored */
	*nm = 1;

	M[0][1] =  1. / (dy * dy); /* up */
	M[1][0] =  1. / (dx * dx); /* left */
	M[1][1] = -2. * (dx*dx + dy*dy) / (dx*dx * dy*dy); /* middle */
	M[1][2] =  1. / (dx * dx); /* right */
	M[2][1] =  1. / (dy * dy); /* down */
}

__global__ void convolution_kernel(double* A, double* B, double* C, const double __restrict__ *M, int nx, int ny, int nm)
{
	/* DANGER: the source (book) example was written for image data, which is row-major. Check loop indices! */

	/* one CUDA core operates on one array index */
	int i, j, tx, ty, src_row, dst_row, src_col, dst_col;
	double value = 0.;

	/* determine indices on which to operate */
	tx = threadIdx.x;
	ty = threadIdx.y;
	dst_row = blockIdx.x * DST_TILE_W + ty;
	dst_col = blockIdx.y * DST_TILE_H + tx;
	src_row = dst_row - nm/2;
	src_col = dst_col - nm/2;

	/* copy A data into this thread's tile buffer */
	__shared__ double N_ds[TILE_H + MAX_MASK_H - 1][TILE_W + MAX_MASK_W - 1];

	if ((src_row >= 0) && (src_row < ny)) && (src_col >= 0) && (src_col < nx)) {
		N_ds[ty][tx] = A[src_row*nx + src_col];
	} else {
		N_ds[ty][tx] = 0.;
	}

	/* compute the convolution */
	if (ty < DST_TILE_W && tx < DST_TILE_WIDTH) {
		for (j = 0; j < nm; j++) {
			for (i = 0; i < nm; i++) {
				value += M[j][i] * N_ds[j+tx][i+ty];
			}
		}
		/* record value */
		if (dst_row < ny && dst_col < nx) {
			C[dst_row*nx + dst_col] = value;
		}
	}
}

void compute_convolution(double** A, double** C, double** M, int nx, int ny, int nm, int bs)
{
	/* Rejoice: easily CUDA-able! */
	double** d_A, **d_C, **d_M;

	dim3 threads(bs, bs);
	dim3 blocks(iDivUp(nx, threads.x), iDivUp(ny, threads.y));

	/* allocate memory on device */
	cudaMalloc((void **)&d_A, nx * ny * sizeof(double));
	cudaMalloc((void **)&d_C, nx * ny * sizeof(double));
	cudaMalloc((void **)&d_M, nm * nm * sizeof(double));

	/* transfer data from host in to device */
	cudaMemcpy(d_A, A, nx * ny * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_C, C, nx * ny * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_M, M, nm * nm * sizeof(double), cudaMemcpyHostToDevice);

	/* compute result */
	convolution_kernel<<<blocks, threads>>>(d_A, d_C, d_M, nx, ny, nm);

	/* transfer from device out from host */
	cudaMemcpy(C, d_C, nx * ny * sizeof(double), cudaMemcpyDeviceToHost);

	/* free memory on device */
	cudaFree(d_A);
	cudaFree(d_C);
	cudaFree(d_M);
}

void step_in_time(double** A, double** B, double** C, int nx, int ny, double D, double dt, double* elapsed)
{
	/* Rejoice: easily CUDA-able! */

	#pragma omp parallel
	{
		int i, j;

		#pragma omp for collapse(2)
		for (j = 1; j < ny-1; j++)
			for (i = 1; i < nx-1; i++)
				B[j][i] = A[j][i] + dt * D * C[j][i];
	}

	*elapsed += dt;
}

void check_solution(double** A, int nx, int ny, double dx, double dy, int bs, double elapsed, double D, double bc[2][2], double* rss)
{
	/* Not easily CUDA-able without a prefix-sum formulation */

	/* OpenCL does not have a GPU-based erf() definition, using Maclaurin series approximation */
	double sum=0.;
	#pragma omp parallel reduction(+:sum)
	{
		int i, j;
		double ca, cal, car, cn, poly_erf, r, trss, z, z2;

		#pragma omp for collapse(2)
		for (j = 1; j < ny-1; j++) {
			for (i = 1; i < nx-1; i++) {
				/* numerical solution */
				cn = A[j][i];

				/* shortest distance to left-wall source */
				r = (j < ny/2) ? dx * (i - 1) : sqrt(dx*dx * (i - 1) * (i - 1) + dy*dy * (j - ny/2) * (j - ny/2));
				z = r / sqrt(4. * D * elapsed);
				z2 = z * z;
				poly_erf = (z < 1.5) ? 2. * z * (1. + z2 * (-1./3 + z2 * (1./10 + z2 * (-1./42 + z2 / 216)))) / sqrt(M_PI) : 1.;
				cal = bc[1][0] * (1. - poly_erf);

				/* shortest distance to right-wall source */
				r = (j >= ny/2) ? dx * (nx-2 - i) : sqrt(dx*dx * (nx-2 - i)*(nx-2 - i) + dy*dy * (ny/2 - j)*(ny/2 - j));
				z = r / sqrt(4. * D * elapsed);
				z2 = z * z;
				poly_erf = (z < 1.5) ? 2. * z * (1. + z2 * (-1./3 + z2 * (1./10 + z2 * (-1./42 + z2 / 216)))) / sqrt(M_PI) : 1.;
				car = bc[1][0] * (1. - poly_erf);

				/* superposition of analytical solutions */
				ca = cal + car;

				/* residual sum of squares (RSS) */
				trss = (ca - cn) * (ca - cn) / (double)((nx-2) * (ny-2));
				sum += trss;
			}
		}
	}

	*rss = sum;
}
