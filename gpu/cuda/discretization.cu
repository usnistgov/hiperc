/*
	File: discretization.c
	Role: implementation of discretized mathematical operations with OpenMP threading and CUDA acceleration

	Questions/comments to trevor.keller@nist.gov
	Bugs/requests to https://github.com/usnistgov/phasefield-accelerator-benchmarks
*/

#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <cuda.h>

extern "C" {
#include "diffusion.h"
}

/* CUDA allocates memory tiles on the GPU statically, so their sizes must be hard coded */
#define MAX_TILE_W 32
#define MAX_TILE_H 32
#define MAX_MASK_W 3
#define MAX_MASK_SIZE (MAX_MASK_W * MAX_MASK_W)

__constant__ fp_t Mc[MAX_MASK_SIZE];

void set_threads(int n)
{
	omp_set_num_threads(n);
}

void five_point_Laplacian_stencil(fp_t dx, fp_t dy, fp_t** M)
{
	M[0][1] =  1. / (dy * dy); /* up */
	M[1][0] =  1. / (dx * dx); /* left */
	M[1][1] = -2. * (dx*dx + dy*dy) / (dx*dx * dy*dy); /* middle */
	M[1][2] =  1. / (dx * dx); /* right */
	M[2][1] =  1. / (dy * dy); /* down */
}

void nine_point_Laplacian_stencil(fp_t dx, fp_t dy, fp_t** M)
{
	M[0][0] =   1. / (6. * dx * dy);
	M[0][1] =   4. / (6. * dy * dy);
	M[0][2] =   1. / (6. * dx * dy);

	M[1][0] =   4. / (6. * dx * dx);
	M[1][1] = -10. * (dx*dx + dy*dy) / (6. * dx*dx * dy*dy);
	M[1][2] =   4. / (6. * dx * dx);

	M[2][0] =   1. / (6. * dx * dy);
	M[2][1] =   4. / (6. * dy * dy);
	M[2][2] =   1. / (6. * dx * dy);
}

void set_mask(fp_t dx, fp_t dy, int nm, fp_t** M)
{
	five_point_Laplacian_stencil(dx, dy, M);
}

__global__ void convolution_kernel(fp_t* A, fp_t* C, int nx, int ny, int nm)
{
	/* Notes:
		* The source matrix (A) and destination matrix (C) must be identical in size
		* One CUDA core operates on one array index: there is no nested loop over matrix elements
		* The halo (nm/2 perimeter cells) in C are unallocated garbage
		* The same cells in A are boundary values, and contribute to the convolution
		* N_ds is the shared tile data array... dunno where the name comes from yet
	*/

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

	/* copy tile from A: __shared__ gives access to all threads working on this tile */
	__shared__ fp_t N_ds[MAX_TILE_H + MAX_MASK_W - 1][MAX_TILE_W + MAX_MASK_W - 1];

	if ((src_row >= 0) && (src_row < ny) &&
	    (src_col >= 0) && (src_col < nx)) {
		/* if src_row==0, then dst_row==nm/2: this is a halo row, still contributing to the output */
		N_ds[ty][tx] = A[src_row * nx + src_col];
	} else {
		/* points outside the halo should be switched off */
		N_ds[ty][tx] = 0.;
	}

	/* tile data is shared: wait for all threads to finish copying */
	__syncthreads();

	/* compute the convolution */
	if (tx < dst_tile_w && ty < dst_tile_h) {
		for (j = 0; j < nm; j++) {
			for (i = 0; i < nm; i++) {
				value += Mc[j * nm + i] * N_ds[j+ty][i+tx];
			}
		}
		/* record value */
		if (dst_row < ny && dst_col < nx) {
			C[dst_row * nx + dst_col] = value;
		}
	}

	/* wait for all threads to finish writing */
	__syncthreads();
}

void compute_convolution(fp_t** A, fp_t** C, fp_t** M, int nx, int ny, int nm, int bs)
{
	fp_t* d_A, *d_C;

	if (bs > MAX_TILE_W) {
		printf("Error: requested block size %i exceeds the statically allocated array size.\n", bs);
		exit(-1);
	}

	/* allocate memory on device */
	cudaMalloc((void **) &d_A, nx * ny * sizeof(fp_t));
	cudaMalloc((void **) &d_C, nx * ny * sizeof(fp_t));

	/* transfer data from host in to device */
	cudaMemcpy(d_A, A[0], nx * ny * sizeof(fp_t), cudaMemcpyHostToDevice);

	/* transfer mask in to constant device memory */
	cudaMemcpyToSymbol(Mc, M[0], nm * nm * sizeof(fp_t));

	/* divide matrices into blocks of (bs x bs) threads */
	dim3 threads(bs - nm/2, bs - nm/2, 1);
	dim3 blocks(ceil(fp_t(nx)/threads.x)+1, ceil(fp_t(ny)/threads.y)+1, 1);

	/* compute result */
	convolution_kernel<<<blocks, threads>>>(d_A, d_C, nx, ny, nm);

	/* transfer from device out from host */
	cudaMemcpy(C[0], d_C, nx * ny * sizeof(fp_t), cudaMemcpyDeviceToHost);

	/* free memory on device */
	cudaFree(d_A);
	cudaFree(d_C);
}

__global__ void diffusion_kernel(fp_t* A, fp_t* B, fp_t* C,
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
		B[row * nx + col] = A[row * nx + col] + dt * D * C[row * nx + col];
	}

	/* wait for all threads to finish writing */
	__syncthreads();
}

void solve_diffusion_equation(fp_t** A, fp_t** B, fp_t** C,
                              int nx, int ny, int nm, int bs, fp_t D, fp_t dt, fp_t* elapsed)
{
	fp_t* d_A, *d_B, *d_C;

	/* allocate memory on device */
	cudaMalloc((void **) &d_A, nx * ny * sizeof(fp_t));
	cudaMalloc((void **) &d_B, nx * ny * sizeof(fp_t));
	cudaMalloc((void **) &d_C, nx * ny * sizeof(fp_t));

	/* transfer data from host in to device */
	cudaMemcpy(d_A, A[0], nx * ny * sizeof(fp_t), cudaMemcpyHostToDevice);
	cudaMemcpy(d_C, C[0], nx * ny * sizeof(fp_t), cudaMemcpyHostToDevice);

	/* divide matrices into blocks of (bs x bs) threads */
	dim3 threads(bs - nm/2, bs - nm/2, 1);
	dim3 blocks(ceil(fp_t(nx)/threads.x)+1, ceil(fp_t(ny)/threads.y)+1, 1);

	/* compute result */
	diffusion_kernel<<<blocks, threads>>>(d_A, d_B, d_C, nx, ny, nm, D, dt);

	/* transfer from device out from host */
	cudaMemcpy(B[0], d_B, nx * ny * sizeof(fp_t), cudaMemcpyDeviceToHost);

	/* free memory on device */
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

	*elapsed += dt;
}

void analytical_value(fp_t x, fp_t t, fp_t D, fp_t bc[2][2], fp_t* c)
{
	*c = bc[1][0] * (1. - erf(x / sqrt(4. * D * t)));
}

void check_solution(fp_t** A,
                    int nx, int ny, fp_t dx, fp_t dy, int nm, int bs,
                    fp_t elapsed, fp_t D, fp_t bc[2][2], fp_t* rss)
{
	fp_t sum=0.;
	#pragma omp parallel reduction(+:sum)
	{
		int i, j;
		fp_t r, cal, car, ca, cn, trss;

		#pragma omp for collapse(2)
		for (j = nm/2; j < ny-nm/2; j++) {
			for (i = nm/2; i < nx-nm/2; i++) {
				/* numerical solution */
				cn = A[j][i];

				/* shortest distance to left-wall source */
				r = (j < ny/2) ? dx * (i - nm/2) : sqrt(dx*dx * (i - nm/2) * (i - nm/2) + dy*dy * (j - ny/2) * (j - ny/2));
				analytical_value(r, elapsed, D, bc, &cal);

				/* shortest distance to right-wall source */
				r = (j >= ny/2) ? dx * (nx-nm/2-1 - i) : sqrt(dx*dx * (nx-nm/2-1 - i)*(nx-nm/2-1 - i) + dy*dy * (ny/2 - j)*(ny/2 - j));
				analytical_value(r, elapsed, D, bc, &car);

				/* superposition of analytical solutions */
				ca = cal + car;

				/* residual sum of squares (RSS) */
				trss = (ca - cn) * (ca - cn) / (fp_t)((nx-nm/2-1) * (ny-nm/2-1));
				sum += trss;
			}
		}
	}

	*rss = sum;
}
