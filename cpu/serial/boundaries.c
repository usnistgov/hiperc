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

void apply_initial_conditions(fp_t** conc, int nx, int ny, int nm, fp_t bc[2][2])
{
	int i, j;

	for (j = 0; j < ny; j++)
		for (i = 0; i < nx; i++)
			conc[j][i] = bc[0][0];

	for (j = 0; j < ny/2; j++)
		for (i = 0; i < 1+nm/2; i++)
			conc[j][i] = bc[1][0]; /* left half-wall */

	for (j = ny/2; j < ny; j++)
		for (i = nx-1-nm/2; i < nx; i++)
			conc[j][i] = bc[1][1]; /* right half-wall */
}

void apply_boundary_conditions(fp_t** conc, int nx, int ny, int nm, fp_t bc[2][2])
{
	/* Set fixed value (c=1) along left and bottom, zero-flux elsewhere */
	int i, j;

	for (j = 0; j < ny/2; j++)
		for (i = 0; i < 1+nm/2; i++)
			conc[j][i] = bc[1][0]; /* left value */

	for (j = ny/2; j < ny; j++)
		for (i = nx-1-nm/2; i < nx; i++)
			conc[j][i] = bc[1][1]; /* right value */

	for (j = 0; j < ny; j++) {
		for (i = nm/2; i > 0; i--)
			conc[j][i-1] = conc[j][i]; /* left condition */
		for (i = nx-1-nm/2; i < nx-1; i++)
			conc[j][i+1] = conc[j][i]; /* right condition */
	}

	for (i = 0; i < nx; i++) {
		for (j = nm/2; j > 0; j--)
			conc[j-1][i] = conc[j][i]; /* bottom condition */
		for (j = ny-1-nm/2; j < ny-1; j++)
			conc[j+1][i] = conc[j][i]; /* top condition */
	}
}
