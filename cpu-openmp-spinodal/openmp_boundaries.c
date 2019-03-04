/**********************************************************************************
 HiPerC: High Performance Computing Strategies for Boundary Value Problems
 Written by Trevor Keller and available from https://github.com/usnistgov/hiperc
 **********************************************************************************/

/**
 \file  openmp_boundaries.c
 \brief Implementation of CHiMaD 1b boundary conditions with OpenMP threading
*/

#include <math.h>
#include <omp.h>
#include "boundaries.h"

void apply_initial_conditions(fp_t** conc, const int nx, const int ny, const int nm)
{
	const fp_t C0 = 0.50;
	const fp_t ep = 0.01;

	#pragma omp parallel
	{
		#pragma omp for collapse(2)
		for (int j = 0; j < ny; j++) {
			for (int i = 0; i < nx; i++) {
				const int y = j - nm/2;
				const int x = i - nm/2;
				conc[j][i] = C0 + ep * (  cos(0.105 * x) * cos(0.110 * y)
										+ cos(0.130 * x) * cos(0.087 * y)
										* cos(0.130 * x) * cos(0.087 * y)
										+ cos(0.025 * x - 0.150 * y)
										* cos(0.070 * x - 0.020 * y)
										);
			}
		}
	}
}

void apply_boundary_conditions(fp_t** conc, const int nx, const int ny, const int nm)
{
	#pragma omp parallel
	{
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
