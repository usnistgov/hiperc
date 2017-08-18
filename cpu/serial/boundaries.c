/*
	File: boundaries.c
	Role: implementation of boundary condition functions without threading

	Questions/comments to trevor.keller@nist.gov
	Bugs/requests to https://github.com/usnistgov/phasefield-accelerator-benchmarks
*/

#include <math.h>

#include "diffusion.h"

void set_boundaries(fp_t bc[2][2])
{
	/* indexing is A[y][x], so bc = [[ylo,yhi], [xlo,xhi]] */
	fp_t clo = 0.0, chi = 1.0;
	bc[0][0] = clo; /* bottom boundary */
	bc[0][1] = clo; /* top boundary */
	bc[1][0] = chi; /* left boundary */
	bc[1][1] = chi; /* right boundary */
}

void apply_initial_conditions(fp_t** A, int nx, int ny, int nm, fp_t bc[2][2])
{
	int i, j;

	for (j = nm/2; j < ny; j++)
		for (i = nm/2; i < nx; i++)
			A[j][i] = bc[0][0];

	for (j = nm/2; j < ny/2; j++)
		A[j][nm/2] = bc[1][0]; /* left half-wall */

	for (j = ny/2; j < ny-nm/2; j++)
		A[j][nx-nm/2-1] = bc[1][1]; /* right half-wall */
}

void apply_boundary_conditions(fp_t** A, int nx, int ny, int nm, fp_t bc[2][2])
{
	/* Set fixed value (c=1) along left and bottom, zero-flux elsewhere */
	int i, j;

	for (j = nm/2; j < ny/2; j++)
		A[j][nm/2] = bc[1][0]; /* left value */

	for (j = ny/2; j < ny-nm/2; j++)
		A[j][nx-nm/2-1] = bc[1][1]; /* right value */

	for (j = nm/2; j < ny-nm/2; j++) {
		A[j][nm/2-1] = A[j][nm/2]; /* left condition */
		A[j][nx-nm/2] = A[j][nx-nm/2-1]; /* right condition */
	}

	for (i = nm/2; i < nx-nm/2; i++) {
		A[nm/2-1][i] = A[nm/2][i]; /* bottom condition */
		A[ny-nm/2][i] = A[ny-nm/2-1][i]; /* top condition */
	}
}
