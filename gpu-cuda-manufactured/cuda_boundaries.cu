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
 \file  cuda_boundaries.cu
 \brief Implementation of boundary condition functions with OpenMP threading
*/

#include <math.h>
#include <omp.h>

extern "C" {
#include "boundaries.h"
}

#include "cuda_kernels.cuh"

void apply_initial_conditions(fp_t** conc,
                              const fp_t dx, const fp_t dy,
                              const int  nx, const int  ny, const int nm,
                              const fp_t A1, const fp_t A2,
                              const fp_t B1, const fp_t B2,
                              const fp_t C2, const fp_t kappa)
{
	#pragma omp parallel
	{
		#pragma omp for collapse(2)
		for (int j = 0; j < ny; j++) {
			for (int i = 0; i < nx; i++) {
                const fp_t x = dx * (i - nm/2);
                const fp_t y = dy * (j - nm/2);
                const fp_t t = 0.;
				manufactured_solution(x, y, t, A1, A2, B1, B2, C2, kappa, &conc[j][i]);
            }
        }
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

    /* set indices of real data along the boundary */
    const int ilo = nm/2;          /* left col */
    const int ihi = nx - 1 - nm/2; /* right col */
    const int jlo = nm/2;          /* bottom row */
    const int jhi = ny - 1 - nm/2; /* top row */

    /* set values for Dirichlet boundaries */
    const fp_t nlo = 0.;
    const fp_t nhi = 1.;

    for (int offset = 0; offset < nm/2; offset++) {
        /* apply periodic conditions on x-axis boundaries */
		if (ilo-offset-1 == col && row < ny)
			d_conc[row * nx + col] = d_conc[row * nx + ilo - offset]; /* left condition: copy right boundary cell - offset */

		if (ihi+offset+1 == col && row < ny)
			d_conc[row * nx + col] = d_conc[row * nx + ihi + offset]; /* right condition: copy left boundary cell + offset */

        /* apply Dirichlet conditions on y-axis boundaries */
		if (jlo-offset-1 == row && col < nx)
			d_conc[row * nx + col] = nlo;                             /* bottom condition: constant */

		if (jhi+offset+1 == row && col < nx)
			d_conc[row * nx + col] = nhi;                             /* top condition: constant */
	}

    __syncthreads();    
}
