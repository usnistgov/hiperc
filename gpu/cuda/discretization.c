/*
	File: discretization.c
	Role: implementation of discretized mathematical operations with CUDA acceleration

	Questions/comments to trevor.keller@nist.gov
	Bugs/requests to https://github.com/tkphd/accelerator-testing
*/

#include <math.h>

#include "diffusion.h"

void set_threads(int n)
{
	/* nothing to do here */
}

void set_mask(double dx, double dy, int* nm, double** M)
{
	/* M is initialized to zero, so corners can be ignored */
	*nm = 1;

	M[0][1] =  1.0 / (dy * dy); /* up */
	M[1][0] =  1.0 / (dx * dx); /* left */
	M[1][1] = -2.0 * (dx*dx + dy*dy) / (dx*dx * dy*dy); /* middle */
	M[1][2] =  1.0 / (dx * dx); /* right */
	M[2][1] =  1.0 / (dy * dy); /* down */
}

void compute_convolution(double** A, double** C, double** M, int nx, int ny, int nm)
{
	int i, j, mi, mj;
	double value;

	for (j = 1; j < ny-1; j++) {
		for (i = 1; i < nx-1; i++) {
			value = 0.0;
			for (mj = -nm; mj < nm+1; mj++) {
				for (mi = -nm; mi < nm+1; mi++) {
					value += M[mj+nm][mi+nm] * A[j+mj][i+mi];
				}
			}
			C[j][i] = value;
		}
	}
}

void step_in_time(double** A, double** B, double** C, int nx, int ny, double D, double dt, double* elapsed)
{
	int i, j;

	for (j = 1; j < ny-1; j++)
		for (i = 1; i < nx-1; i++)
			B[j][i] = A[j][i] + dt * D * C[j][i];

	*elapsed += dt;
}

void analytical_value(double x, double t, double D, double bc[2][2], double* c)
{
	*c = bc[1][0] * (1.0 - erf(x / sqrt(4.0 * D * t)));
}

void check_solution(double** A, int nx, int ny, double dx, double dy, double elapsed, double D, double bc[2][2], double* rss)
{
	int i, j;
	double r, cal, car, ca, cn;
	*rss = 0.0;

	for (j = 1; j < ny-1; j++) {
		for (i = 1; i < nx-1; i++) {
			/* numerical solution */
			cn = A[j][i];

			/* shortest distance to left-wall source */
			r = (j < ny/2) ? dx * (i - 1) : sqrt(dx*dx * (i - 1) * (i - 1) + dy*dy * (j - ny/2) * (j - ny/2));
			analytical_value(r, elapsed, D, bc, &cal);

			/* shortest distance to right-wall source */
			r = (j >= ny/2) ? dx * (nx-2 - i) : sqrt(dx*dx * (nx-2 - i)*(nx-2 - i) + dy*dy * (ny/2 - j)*(ny/2 - j));
			analytical_value(r, elapsed, D, bc, &car);

			/* superposition of analytical solutions */
			ca = cal + car;

			/* residual sum of squares (RSS) */
			*rss += (ca - cn) * (ca - cn) / (double)((nx-2) * (ny-2));
		}
	}
}
