/**********************************************************************************
 HiPerC: High Performance Computing Strategies for Boundary Value Problems
 Written by Trevor Keller and available from https://github.com/usnistgov/hiperc
 **********************************************************************************/

/**
 \file  phi_boundaries.c
 \brief Implementation of boundary condition functions with OpenMP threading
*/

#include <math.h>
#include <omp.h>
#include "boundaries.h"

void set_boundaries(fp_t bc[2][2])
{
	/* Change these values to your liking: */
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
		#pragma omp for collapse(2)
		for (int j = 0; j < ny; j++)
			for (int i = 0; i < nx; i++)
				conc[j][i] = bc[0][0];

		#pragma omp for collapse(2)
		for (int j = 0; j < ny/2; j++)
			for (int i = 0; i < 1+nm/2; i++)
				conc[j][i] = bc[1][0]; /* left half-wall */

		#pragma omp for collapse(2)
		for (int j = ny/2; j < ny; j++)
			for (int i = nx-1-nm/2; i < nx; i++)
				conc[j][i] = bc[1][1]; /* right half-wall */
	}
}

void apply_boundary_conditions(fp_t** conc, int nx, int ny, int nm, fp_t bc[2][2])
{
	#pragma omp parallel
	{
		/* apply fixed boundary values: sequence does not matter */

		#pragma omp for collapse(2) private(i,j)
		for (int j = 0; j < ny/2; j++) {
			for (int i = 0; i < 1+nm/2; i++) {
				conc[j][i] = bc[1][0]; /* left value */
			}
		}

		#pragma omp for collapse(2) private(i,j)
		for (int j = ny/2; j < ny; j++) {
			for (int i = nx-1-nm/2; i < nx; i++) {
				conc[j][i] = bc[1][1]; /* right value */
			}
		}

		/* apply no-flux boundary conditions: inside to out, sequence matters */

		for (offset = 0; offset < nm/2; offset++) {
			const int ilo = nm/2 - offset;
			const int ihi = nx - 1 - nm/2 + offset;
			#pragma omp for private(j)
			for (int j = 0; j < ny; j++) {
				conc[j][ilo-1] = conc[j][ilo]; /* left condition */
				conc[j][ihi+1] = conc[j][ihi]; /* right condition */
			}
		}

		for (offset = 0; offset < nm/2; offset++) {
			const int jlo = nm/2 - offset;
			const int jhi = ny - 1 - nm/2 + offset;
			#pragma omp for private(i)
			for (int i = 0; i < nx; i++) {
				conc[jlo-1][i] = conc[jlo][i]; /* bottom condition */
				conc[jhi+1][i] = conc[jhi][i]; /* top condition */
			}
		}
	}
}
