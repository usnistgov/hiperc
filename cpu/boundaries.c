/*
	File: boundaries.c
	Role: implementation of boundary condition functions

	Questions/comments to trevor.keller@nist.gov
	Bugs/requests to https://github.com/tkphd/accelerator-testing
*/

#include <math.h>

#include "diffusion.h"

void set_boundaries(double* c0, double bc[2][2])
{
	*c0 = 0.0; /*initial flat composition */

	/* C indexing is A[y][x], so bc = [[ylo,yhi], [xlo,xhi]] */
	bc[0][0] = *c0; /* bottom boundary */
	bc[0][1] = *c0; /* top boundary */
	bc[1][0] = 1.0; /* left boundary */
	bc[1][1] = *c0; /* right boundary */
}

void apply_initial_conditions(double** A, int nx, int ny, double c0, double bc[2][2])
{
	/* bulk values only */
	int i, j;

	for (j = 1; j < ny-1; j++) {
		A[j][1] = bc[1][0];
		for (i = 2; i < nx-1; i++) {
			A[j][i] = c0;
		}
	}

}

void apply_boundary_conditions(double** A, int nx, int ny, double bc[2][2])
{
	/* boundary values only */
	int i, j;

	for (j = 0; j < ny; j++) {
		A[j][0] = bc[1][0]; /* left */
		A[j][nx-1] = A[j][nx-2]; /* right */
	}

	for (i = 0; i < nx; i++) {
		A[0][i] = A[1][i]; /* bottom */
		A[ny-1][i] = A[ny-2][i]; /* top */
	}

}
