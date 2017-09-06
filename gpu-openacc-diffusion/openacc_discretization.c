/**********************************************************************************
 This file is part of Phase-field Accelerator Benchmarks, written by Trevor Keller
 and available from https://github.com/usnistgov/phasefield-accelerator-benchmarks.

 This software was developed at the National Institute of Standards and Technology
 by employees of the Federal Government in the course of their official duties.
 Pursuant to title 17 section 105 of the United States Code this software is not
 subject to copyright protection and is in the public domain. NIST assumes no
 responsibility whatsoever for the use of this software by other parties, and makes
 no guarantees, expressed or implied, about its quality, reliability, or any other
 characteristic. We would appreciate acknowledgement if the software is used.

 This software can be redistributed and/or modified freely provided that any
 derivative works bear some notice that they are derived from it, and any modified
 versions bear some notice that they have been modified.

 Questions/comments to Trevor Keller (trevor.keller@nist.gov)
 **********************************************************************************/

/**
 \file  openacc_discretization.c
 \brief Implementation of boundary condition functions with OpenACC threading
*/

#include <math.h>
#include <omp.h>
#include <openacc.h>
#include "discretization.h"
#include "mesh.h"
#include "openacc_kernels.h"

void convolution_kernel(fp_t** conc_old, fp_t** conc_lap, fp_t** mask_lap, int nx, int ny, int nm)
{
	#pragma acc parallel
	{
		#pragma acc loop
		for (int j = nm/2; j < ny-nm/2; j++) {
			#pragma acc loop
			for (int i = nm/2; i < nx-nm/2; i++) {
				fp_t value = 0.;
				for (int mj = -nm/2; mj < 1+nm/2; mj++) {
					for (int mi = -nm/2; mi < 1+nm/2; mi++) {
						value += mask_lap[mj+nm/2][mi+nm/2] * conc_old[j+mj][i+mi];
					}
				}
				conc_lap[j][i] = value;
			}
		}
	}
}

void diffusion_kernel(fp_t** conc_old, fp_t** conc_new, fp_t** conc_lap,
                      int nx, int ny, int nm, fp_t D, fp_t dt)
{
	#pragma acc parallel
	{
		#pragma acc loop
		for (int j = nm/2; j < ny-nm/2; j++) {
			#pragma acc loop
			for (int i = nm/2; i < nx-nm/2; i++) {
				conc_new[j][i] = conc_old[j][i] + dt * D * conc_lap[j][i];
			}
		}
	}
}


void compute_convolution(fp_t** conc_old, fp_t** conc_lap, fp_t** mask_lap,
                         int nx, int ny, int nm)
{
	/* If you must compute the convolution separately, do so here. */
	#pragma acc data copyin(conc_old[0:ny][0:nx], mask_lap[0:nm][0:nm]) create(conc_lap[0:ny][0:nx]) copyout(conc_lap[0:ny][0:nx])
	{
		convolution_kernel(conc_old, conc_lap, mask_lap, nx, ny, nm);
	}
}

void solve_diffusion_equation(fp_t** conc_old, fp_t** conc_new, fp_t** conc_lap,
                              fp_t** mask_lap, int nx, int ny, int nm,
                              fp_t bc[2][2], fp_t D, fp_t dt, fp_t* elapsed,
                              struct Stopwatch* sw, int checks)
{
	#pragma acc data copy(conc_old[0:ny][0:nx], mask_lap[0:nm][0:nm], bc[0:2][0:2]) create(conc_lap[0:ny][0:nx], conc_new[0:ny][0:nx])
	{
		double start_time=0.;
		int check=0;

		for (check = 0; check < checks; check++) {
			boundary_kernel(conc_old, nx, ny, nm, bc);

			start_time = GetTimer();
			convolution_kernel(conc_old, conc_lap, mask_lap, nx, ny, nm);
			sw->conv += GetTimer() - start_time;

			start_time = GetTimer();
			diffusion_kernel(conc_old, conc_new, conc_lap, nx, ny, nm, D, dt);
			sw->step += GetTimer() - start_time;

			swap_pointers(&conc_old, &conc_new);
		}
	}

	*elapsed += dt * checks;
}

void check_solution(fp_t** conc_new, fp_t** conc_lap,  int nx, int ny,
                    fp_t dx, fp_t dy, int nm, fp_t elapsed, fp_t D,
                    fp_t bc[2][2], fp_t* rss)
{
	fp_t sum=0.;

	#pragma omp parallel reduction(+:sum)
	{
		int i, j;
		fp_t r, cal, car, ca, cn;

		#pragma omp for collapse(2) private(ca,cal,car,cn,i,j,r)
		for (j = nm/2; j < ny-nm/2; j++) {
			for (i = nm/2; i < nx-nm/2; i++) {
				/* numerical solution */
				cn = conc_new[j][i];

				/* shortest distance to left-wall source */
				r = distance_point_to_segment(dx * (nm/2), dy * (nm/2),
				                              dx * (nm/2), dy * (ny/2),
				                              dx * i, dy * j);
				analytical_value(r, elapsed, D, bc, &cal);

				/* shortest distance to right-wall source */
				r = distance_point_to_segment(dx * (nx-1-nm/2), dy * (ny/2),
				                              dx * (nx-1-nm/2), dy * (ny-1-nm/2),
				                              dx * i, dy * j);
				analytical_value(r, elapsed, D, bc, &car);

				/* superposition of analytical solutions */
				ca = cal + car;

				/* residual sum of squares (RSS) */
				conc_lap[j][i] = (ca - cn) * (ca - cn) / (fp_t)((nx-1-nm/2) * (ny-1-nm/2));
			}
		}

		#pragma omp for collapse(2) private(i,j)
		for (j = nm/2; j < ny-nm/2; j++) {
			for (i = nm/2; i < nx-nm/2; i++) {
				sum += conc_lap[j][i];
			}
		}
	}

	*rss = sum;
}
