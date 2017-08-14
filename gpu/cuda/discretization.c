/*
	File: discretization.c
	Role: implementation of discretized mathematical operations with OpenMP threading

	Questions/comments to trevor.keller@nist.gov
	Bugs/requests to https://github.com/tkphd/accelerator-testing
*/

#include <math.h>
#include <omp.h>

#include "diffusion.h"

void set_threads(int n)
{
	omp_set_num_threads(n);
}

void set_mask(double dx, double dy, int* nm, double** M)
{
	/* M is initialized to zero, so corners can be ignored */
	*nm = 1;

	M[0][1] =  1. / (dy * dy); /* up */
	M[1][0] =  1. / (dx * dx); /* left */
	M[1][1] = -2. * (dx*dx + dy*dy) / (dx*dx * dy*dy); /* middle */
	M[1][2] =  1. / (dx * dx); /* right */
	M[2][1] =  1. / (dy * dy); /* down */
}

void compute_convolution(double** A, double** C, double** M, int nx, int ny, int nm)
{
	#pragma omp parallel
	{
		int i, j, mi, mj;
		double value;

		#pragma omp for collapse(2)
		for (j = 1; j < ny-1; j++) {
			for (i = 1; i < nx-1; i++) {
				value = 0.;
				for (mj = -nm; mj < nm+1; mj++) {
					for (mi = -nm; mi < nm+1; mi++) {
						value += M[mj+nm][mi+nm] * A[j+mj][i+mi];
					}
				}
				C[j][i] = value;
			}
		}
	}
}

void step_in_time(double** A, double** B, double** C, int nx, int ny, double D, double dt, double* elapsed)
{
	#pragma omp parallel
	{
		int i, j;

		#pragma omp for collapse(2)
		for (j = 1; j < ny-1; j++)
			for (i = 1; i < nx-1; i++)
				B[j][i] = A[j][i] + dt * D * C[j][i];
	}

	*elapsed += dt;
}

void check_solution(double** A, int nx, int ny, double dx, double dy, double elapsed, double D, double bc[2][2], double* rss)
{
	/* OpenCL does not have a GPU-based erf() definition, using Maclaurin series approximation */
	double sum=0.;
	#pragma omp parallel reduction(+:sum)
	{
		int i, j;
		double ca, cal, car, cn, r, trss, z, z2;

		#pragma omp for collapse(2)
		for (j = 1; j < ny-1; j++) {
			for (i = 1; i < nx-1; i++) {
				/* numerical solution */
				cn = A[j][i];

				/* shortest distance to left-wall source */
				r = (j < ny/2) ? dx * (i - 1) : sqrt(dx*dx * (i - 1) * (i - 1) + dy*dy * (j - ny/2) * (j - ny/2));
				z2 = z * z;
				poly_erf = (z < 1.) ? 2. * z * (1. + z2 * (-1./3 + z2 * (1./10 + z2 * (-1./42 + z2 / 216)))) / M_PI : 1.;
				cal = bc[1][0] * (1. - poly_erf);

				/* shortest distance to right-wall source */
				r = (j >= ny/2) ? dx * (nx-2 - i) : sqrt(dx*dx * (nx-2 - i)*(nx-2 - i) + dy*dy * (ny/2 - j)*(ny/2 - j));
				z2 = z * z;
				poly_erf = (z < 1.) ? 2. * z * (1. + z2 * (-1./3 + z2 * (1./10 + z2 * (-1./42 + z2 / 216)))) / M_PI : 1.;
				car = bc[1][0] * (1. - poly_erf);

				/* superposition of analytical solutions */
				ca = cal + car;

				/* residual sum of squares (RSS) */
				trss = (ca - cn) * (ca - cn) / (double)((nx-2) * (ny-2));
				sum += trss;
			}
		}
	}

	*rss = sum;
}
