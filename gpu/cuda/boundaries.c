/*
	File: boundaries.c
	Role: implementation of boundary condition functions with OpenMP threading

	Questions/comments to trevor.keller@nist.gov
	Bugs/requests to https://github.com/tkphd/accelerator-testing
*/

#include <math.h>
#include <omp.h>

#include "diffusion.h"

void set_boundaries(double bc[2][2])
{
	/* indexing is A[y][x], so bc = [[ylo,yhi], [xlo,xhi]] */
	double clo = 0.0, chi = 1.0;
	bc[0][0] = clo; /* bottom boundary */
	bc[0][1] = clo; /* top boundary */
	bc[1][0] = chi; /* left boundary */
	bc[1][1] = chi; /* right boundary */
}

void apply_initial_conditions(double** A, int nx, int ny, double bc[2][2])
{
	#pragma omp parallel
	{
		int i, j;

		#pragma omp for collapse(2)
		for (j = 1; j < ny; j++)
			for (i = 1; i < nx; i++)
				A[j][i] = bc[0][0];

		#pragma omp for nowait
		for (j = 1; j < ny/2; j++)
			A[j][1] = bc[1][0]; /* left half-wall */

		#pragma omp for
		for (j = ny/2; j < ny-1; j++)
			A[j][nx-2] = bc[1][1]; /* right half-wall */
	}
}

void apply_boundary_conditions(double** A, int nx, int ny, double bc[2][2])
{
	#pragma omp parallel
	{
		/* Set fixed value (c=1) along left and bottom, zero-flux elsewhere */
		int i, j;

		#pragma omp for
		for (j = 1; j < ny/2; j++)
			A[j][1] = bc[1][0]; /* left value */

		#pragma omp for
		for (j = ny/2; j < ny-1; j++)
			A[j][nx-2] = bc[1][1]; /* right value */

		#pragma omp for nowait
		for (j = 1; j < ny-1; j++) {
			A[j][0] = A[j][1]; /* left condition */
			A[j][nx-1] = A[j][nx-2]; /* right condition */
		}

		/* bottom boundary */
		#pragma omp for nowait
		for (i = 1; i < nx-1; i++) {
			A[0][i] = A[1][i]; /* top condition */
			A[ny-1][i] = A[ny-2][i]; /* bottom condition */
		}
	}
}
