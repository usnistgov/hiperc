/**********************************************************************************
 HiPerC: High Performance Computing Strategies for Boundary Value Problems
 Written by Trevor Keller and available from https://github.com/usnistgov/hiperc
 **********************************************************************************/

/**
 \file  openacc_boundaries.c
 \brief Implementation of boundary condition functions with OpenMP threading
*/

#include <math.h>
#include <omp.h>
#include "boundaries.h"
#include "openacc_kernels.h"

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

void boundary_kernel(fp_t** __restrict__ conc, const int nx, const int ny, const int nm)
{
	/* apply fixed boundary values: sequence does not matter */

	#pragma acc declare present(conc[0:ny][0:nx])
	{
		#pragma acc parallel
		{
			#pragma acc loop independent collapse(2)
			for (int j = 0; j < ny/2; j++)
			{
				for (int i = 0; i < 1+nm/2; i++) {
					conc[j][i] = 1.; /* left value */
				}
			}

			#pragma acc loop independent collapse(2)
			for (int j = ny/2; j < ny; j++)
			{
				for (int i = nx-1-nm/2; i < nx; i++) {
					conc[j][i] = 1.; /* right value */
				}
			}
		}

		/* apply no-flux boundary conditions: inside to out, sequence matters */

		for (int offset = 0; offset < nm/2; offset++) {
			#pragma acc parallel
			{
				int ilo = nm/2 - offset;
				int ihi = nx - 1 - nm/2 + offset;
				int jlo = nm/2 - offset;
				int jhi = ny - 1 - nm/2 + offset;

				#pragma acc loop independent
				for (int j = 0; j < ny; j++)
				{
					conc[j][ilo-1] = conc[j][ilo]; /* left condition */
					conc[j][ihi+1] = conc[j][ihi]; /* right condition */
				}

				#pragma acc loop independent
				for (int i = 0; i < nx; i++)
				{
					conc[jlo-1][i] = conc[jlo][i]; /* bottom condition */
					conc[jhi+1][i] = conc[jhi][i]; /* top condition */
				}
			}
		}
	}
}

void apply_boundary_conditions(fp_t** conc, const int nx, const int ny, const int nm)
{
	#pragma acc data copy(conc[0:ny][0:nx])
	{
		boundary_kernel(conc, nx, ny, nm);
	}
}
