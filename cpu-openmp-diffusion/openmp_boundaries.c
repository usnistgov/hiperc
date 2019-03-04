/**********************************************************************************
 HiPerC: High Performance Computing Strategies for Boundary Value Problems
 Written by Trevor Keller and available from https://github.com/usnistgov/hiperc
 **********************************************************************************/

/**
 \file  openmp_boundaries.c
 \brief Implementation of boundary condition functions with OpenMP threading
*/

#include <math.h>
#include <omp.h>
#include "boundaries.h"

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

void apply_boundary_conditions(fp_t** conc, const int nx, const int ny, const int nm)
{
	#pragma omp parallel
	{
		/* apply fixed boundary values: sequence does not matter */

		#pragma omp for collapse(2)
		for (int j = 0; j < ny/2; j++) {
			for (int i = 0; i < 1+nm/2; i++) {
				conc[j][i] = 1.; /* left value */
			}
		}

		#pragma omp for collapse(2)
		for (int j = ny/2; j < ny; j++) {
			for (int i = nx-1-nm/2; i < nx; i++) {
				conc[j][i] = 1.; /* right value */
			}
		}

		/* apply no-flux boundary conditions: inside to out, sequence matters */

		for (int offset = 0; offset < nm/2; offset++) {
			const int ilo = nm/2 - offset;
			const int ihi = nx - 1 - nm/2 + offset;
			#pragma omp for
			for (int j = 0; j < ny; j++) {
				conc[j][ilo-1] = conc[j][ilo]; /* left condition */
				conc[j][ihi+1] = conc[j][ihi]; /* right condition */
			}
		}

		for (int offset = 0; offset < nm/2; offset++) {
			const int jlo = nm/2 - offset;
			const int jhi = ny - 1 - nm/2 + offset;
			#pragma omp for
			for (int i = 0; i < nx; i++) {
				conc[jlo-1][i] = conc[jlo][i]; /* bottom condition */
				conc[jhi+1][i] = conc[jhi][i]; /* top condition */
			}
		}
	}
}
