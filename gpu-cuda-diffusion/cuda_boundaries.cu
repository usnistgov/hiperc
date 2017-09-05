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
 \file  cuda_boundaries.cu
 \brief Implementation of boundary condition functions with OpenMP threading
*/

#include <math.h>
#include <omp.h>

extern "C" {
#include "boundaries.h"
#include "cuda_kernels.cuh"
}

#include "cuda_kernels.cuh"

__constant__ fp_t d_bc[2][2];

void set_boundaries(fp_t bc[2][2])
{
	fp_t clo = 0.0, chi = 1.0;
	bc[0][0] = clo; /* bottom boundary */
	bc[0][1] = clo; /* top boundary */
	bc[1][0] = chi; /* left boundary */
	bc[1][1] = chi; /* right boundary */
}

void apply_initial_conditions(fp_t** conc, int nx, int ny, int nm, fp_t bc[2][2])
{
	#pragma omp parallel
	{
		int i, j;

		#pragma omp for collapse(2)
		for (j = 0; j < ny; j++)
			for (i = 0; i < nx; i++)
				conc[j][i] = bc[0][0];

		#pragma omp for collapse(2)
		for (j = 0; j < ny/2; j++)
			for (i = 0; i < 1+nm/2; i++)
				conc[j][i] = bc[1][0]; /* left half-wall */

		#pragma omp for collapse(2)
		for (j = ny/2; j < ny; j++)
			for (i = nx-1-nm/2; i < nx; i++)
				conc[j][i] = bc[1][1]; /* right half-wall */
	}
}

__global__ void boundary_kernel(fp_t* conc, int nx, int ny, int nm)
{
	int tx, ty, row, col;
	int ihi, ilo, jhi, jlo, offset;

	/* determine indices on which to operate */

	tx = threadIdx.x;
	ty = threadIdx.y;

	row = blockDim.y * blockIdx.y + ty;
	col = blockDim.x * blockIdx.x + tx;

	/* apply fixed boundary values: sequence does not matter */

	if (row >= 0 && row < ny/2 && col >= 0 && col < 1+nm/2) {
		conc[row * nx + col] = d_bc[1][0]; /* left value */
	}

	if (row >= ny/2 && row < ny && col >= nx-1-nm/2 && col < nx) {
		conc[row * nx + col] = d_bc[1][1]; /* right value */
	}

	/* wait for all threads to finish writing */
	__syncthreads();

	/* apply no-flux boundary conditions: inside to out, sequence matters */

	for (offset = 0; offset < nm/2; offset++) {
		ilo = nm/2 - offset;
		ihi = nx - 1 - nm/2 + offset;
		if (col == ilo-1 && row >= 0 && row < ny) {
			conc[row * nx + col] = conc[row * nx + ilo]; /* left condition */
		} else if (col == ihi+1 && row >= 0 && row < ny) {
			conc[row * nx + col] = conc[row * nx + ihi]; /* right condition */
		}
		__syncthreads();
	}

	for (offset = 0; offset < nm/2; offset++) {
		jlo = nm/2 - offset;
		jhi = ny - 1 - nm/2 + offset;
		if (row == jlo-1 && col >= 0 && col < nx) {
			conc[row * nx + col] = conc[jlo * nx + col]; /* bottom condition */
		} else if (row == jhi+1 && col >= 0 && col < nx) {
			conc[row * nx + col] = conc[jhi * nx + col]; /* top condition */
		}
		__syncthreads();
	}
}

void apply_boundary_conditions(fp_t** conc, int nx, int ny, int nm, fp_t bc[2][2])
{
	fp_t* d_conc;

	/* allocate memory on device */
	cudaMalloc((void **) &d_conc, nx * ny * sizeof(fp_t));

	/* divide matrices into blocks of (MAX_TILE_W x MAX_TILE_W) threads */
	dim3 threads(MAX_TILE_W - nm/2, MAX_TILE_W - nm/2, 1);
	dim3 blocks(ceil(fp_t(nx)/threads.x)+1, ceil(fp_t(ny)/threads.y)+1, 1);

	/* transfer mask in to constant device memory */
	cudaMemcpyToSymbol(d_bc, bc[0], 2 * 2 * sizeof(fp_t));

	/* transfer data from host in to device */
	cudaMemcpy(d_conc, conc[0], nx * ny * sizeof(fp_t), cudaMemcpyHostToDevice);

	/* apply boundary conditions */
	boundary_kernel<<<blocks, threads>>>(d_conc, nx, ny, nm);

	/* transfer from device out to host */
	cudaMemcpy(conc[0], d_conc, nx * ny * sizeof(fp_t), cudaMemcpyDeviceToHost);

	/* free memory on device */
	cudaFree(d_conc);
}
