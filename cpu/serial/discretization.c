/*
	File: discretization.c
	Role: implementation of discretized mathematical operations without threading

	Questions/comments to trevor.keller@nist.gov
	Bugs/requests to https://github.com/usnistgov/phasefield-accelerator-benchmarks
*/

#include <math.h>

#include "diffusion.h"

void set_threads(int n)
{
	/* nothing to do here */
}

void five_point_Laplacian_stencil(fp_t dx, fp_t dy, fp_t** mask_lap)
{
	/* 3x3 mask, 5 values, truncation error O(dx^2) */
	mask_lap[0][1] =  1. / (dy * dy); /* up */
	mask_lap[1][0] =  1. / (dx * dx); /* left */
	mask_lap[1][1] = -2. * (dx*dx + dy*dy) / (dx*dx * dy*dy); /* middle */
	mask_lap[1][2] =  1. / (dx * dx); /* right */
	mask_lap[2][1] =  1. / (dy * dy); /* down */
}

void nine_point_Laplacian_stencil(fp_t dx, fp_t dy, fp_t** mask_lap)
{
	/* 3x3 mask, 9 values, truncation error O(dx^4) */
	mask_lap[0][0] =   1. / (6. * dx * dy);
	mask_lap[0][1] =   4. / (6. * dy * dy);
	mask_lap[0][2] =   1. / (6. * dx * dy);

	mask_lap[1][0] =   4. / (6. * dx * dx);
	mask_lap[1][1] = -10. * (dx*dx + dy*dy) / (6. * dx*dx * dy*dy);
	mask_lap[1][2] =   4. / (6. * dx * dx);

	mask_lap[2][0] =   1. / (6. * dx * dy);
	mask_lap[2][1] =   4. / (6. * dy * dy);
	mask_lap[2][2] =   1. / (6. * dx * dy);
}

void expensive_nine_point_Laplacian_stencil(fp_t dx, fp_t dy, fp_t** mask_lap)
{
	/* 4x4 mask, 9 values, truncation error O(dx^4)
	   Provided for testing and demonstration of scalability, only:
	   as the name indicates, this 9-point stencil is computationally
	   more expensive than the 3x3 version. If your code requires O(dx^4)
	   accuracy, please use nine_point_Laplacian_stencil. */

	mask_lap[0][2] = -1. / (12. * dy * dy);

	mask_lap[1][2] =  4. / (3. * dy * dy);

	mask_lap[2][0] = -1. / (12. * dx * dx);
	mask_lap[2][1] =  4. / (3. * dx * dx);
	mask_lap[2][2] = -5. * (dx*dx + dy*dy) / (2. * dx*dx * dy*dy);
	mask_lap[2][3] =  4. / (3. * dx * dx);
	mask_lap[2][4] = -1. / (12. * dx * dx);

	mask_lap[3][2] =  4. / (3. * dy * dy);

	mask_lap[4][2] = -1. / (12. * dy * dy);
}

void set_mask(fp_t dx, fp_t dy, int nm, fp_t** mask_lap)
{
	five_point_Laplacian_stencil(dx, dy, mask_lap);
}

void compute_convolution(fp_t** conc_old, fp_t** conc_lap, fp_t** mask_lap, int nx, int ny, int nm)
{
	int i, j, mi, mj;
	fp_t value;

	for (j = nm/2; j < ny-nm/2; j++) {
		for (i = nm/2; i < nx-nm/2; i++) {
			value = 0.0;
			for (mj = -nm/2; mj < nm/2+1; mj++) {
				for (mi = -nm/2; mi < nm/2+1; mi++) {
					value += mask_lap[mj+nm/2][mi+nm/2] * conc_old[j+mj][i+mi];
				}
			}
			conc_lap[j][i] = value;
		}
	}
}

void solve_diffusion_equation(fp_t** conc_old, fp_t** conc_new, fp_t** conc_lap, int nx, int ny, int nm, fp_t D, fp_t dt, fp_t* elapsed)
{
	int i, j;

	for (j = nm/2; j < ny-nm/2; j++)
		for (i = nm/2; i < nx-nm/2; i++)
			conc_new[j][i] = conc_old[j][i] + dt * D * conc_lap[j][i];

	*elapsed += dt;
}

void analytical_value(fp_t x, fp_t t, fp_t D, fp_t bc[2][2], fp_t* c)
{
	*c = bc[1][0] * (1.0 - erf(x / sqrt(4.0 * D * t)));
}

void check_solution(fp_t** conc_new, int nx, int ny, fp_t dx, fp_t dy, int nm, fp_t elapsed, fp_t D, fp_t bc[2][2], fp_t* rss)
{
	int i, j;
	fp_t r, cal, car, ca, cn;
	*rss = 0.0;

	for (j = nm/2; j < ny-nm/2; j++) {
		for (i = nm/2; i < nx-nm/2; i++) {
			/* numerical solution */
			cn = conc_new[j][i];

			/* shortest distance to left-wall source */
			r = (j < ny/2) ? dx * (i - nm/2) : sqrt(dx*dx * (i - nm/2) * (i - nm/2) + dy*dy * (j - ny/2) * (j - ny/2));
			analytical_value(r, elapsed, D, bc, &cal);

			/* shortest distance to right-wall source */
			r = (j >= ny/2) ? dx * (nx-1-nm/2 - i) : sqrt(dx*dx * (nx-1-nm/2 - i)*(nx-1-nm/2 - i) + dy*dy * (ny/2 - j)*(ny/2 - j));
			analytical_value(r, elapsed, D, bc, &car);

			/* superposition of analytical solutions */
			ca = cal + car;

			/* residual sum of squares (RSS) */
			*rss += (ca - cn) * (ca - cn) / (fp_t)((nx-1-nm/2) * (ny-1-nm/2));
		}
	}
}
