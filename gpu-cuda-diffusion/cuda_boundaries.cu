/**********************************************************************************
 HiPerC: High Performance Computing Strategies for Boundary Value Problems
 Written by Trevor Keller and available from https://github.com/usnistgov/hiperc
 **********************************************************************************/

/**
 \file  cuda_boundaries.cu
 \brief Implementation of boundary condition functions with OpenMP threading
*/

#include <math.h>
#include <omp.h>

extern "C" {
#include "boundaries.h"
}

#include "cuda_kernels.cuh"

void apply_initial_conditions(fp_t** conc, const int nx, const int ny, const int nm)
{
	#pragma omp parallel
	{
		#pragma omp for collapse(2)
		for (int j = 0; j < ny; j++)
			for (int i = 0; i < nx; i++)
				conc[j][i] = 0.;

		#pragma omp for collapse(2)
		for (int j = 0; j < ny/2; j++)
			for (int i = 0; i < 1+nm/2; i++)
				conc[j][i] = 1.; /* left half-wall */

		#pragma omp for collapse(2)
		for (int j = ny/2; j < ny; j++)
			for (int i = nx-1-nm/2; i < nx; i++)
				conc[j][i] = 1.; /* right half-wall */
	}
}

__global__ void boundary_kernel(fp_t* d_conc,
                                const int nx,
                                const int ny,
                                const int nm)
{
	/* determine indices on which to operate */
	const int tx = threadIdx.x;
	const int ty = threadIdx.y;

	const int row = blockDim.y * blockIdx.y + ty;
	const int col = blockDim.x * blockIdx.x + tx;

	/* apply fixed boundary values: sequence does not matter */

	if (row < ny/2 && col < 1+nm/2) {
		d_conc[row * nx + col] = 1.; /* left value */
	}

	if (row >= ny/2 && row < ny && col >= nx-1-nm/2 && col < nx) {
		d_conc[row * nx + col] = 1.; /* right value */
	}

	/* wait for all threads to finish writing */
	__syncthreads();

	/* apply no-flux boundary conditions: inside to out, sequence matters */

	for (int offset = 0; offset < nm/2; offset++) {
		const int ilo = nm/2 - offset;
		const int ihi = nx - 1 - nm/2 + offset;
		const int jlo = nm/2 - offset;
		const int jhi = ny - 1 - nm/2 + offset;

		if (ilo-1 == col && row < ny) {
			d_conc[row * nx + ilo-1] = d_conc[row * nx + ilo]; /* left condition */
		}
		if (ihi+1 == col && row < ny) {
			d_conc[row * nx + ihi+1] = d_conc[row * nx + ihi]; /* right condition */
		}
		if (jlo-1 == row && col < nx) {
			d_conc[(jlo-1) * nx + col] = d_conc[jlo * nx + col]; /* bottom condition */
		}
		if (jhi+1 == row && col < nx) {
			d_conc[(jhi+1) * nx + col] = d_conc[jhi * nx + col]; /* top condition */
		}

		__syncthreads();
	}
}
