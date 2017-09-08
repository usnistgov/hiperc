/**********************************************************************************
 HIPERC: High Performance Computing Strategies for Boundary Value Problems
 written by Trevor Keller and available from https://github.com/usnistgov/hiperc

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
 \file  serial_discretization.c
 \brief Implementation of boundary condition functions without threading
*/

#include <math.h>
#include "boundaries.h"
#include "discretization.h"
#include "mesh.h"
#include "numerics.h"
#include "timer.h"

void compute_convolution(fp_t** conc_old, fp_t** conc_lap, fp_t** mask_lap,
                         int nx, int ny, int nm)
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

void solve_diffusion_equation(fp_t** conc_old, fp_t** conc_new, fp_t** conc_lap,
                              fp_t** mask_lap, int nx, int ny, int nm,
                              fp_t bc[2][2], fp_t D, fp_t dt, int checks,
                              fp_t* elapsed, struct Stopwatch* sw)
{
	int i, j, check;
	double start_time=0.;

	for (check = 0; check < checks; check++) {
		apply_boundary_conditions(conc_old, nx, ny, nm, bc);

		start_time = GetTimer();
		compute_convolution(conc_old, conc_lap, mask_lap, nx, ny, nm);
		sw->conv += GetTimer() - start_time;

		start_time = GetTimer();
		for (j = nm/2; j < ny-nm/2; j++)
			for (i = nm/2; i < nx-nm/2; i++)
				conc_new[j][i] = conc_old[j][i] + dt * D * conc_lap[j][i];

		*elapsed += dt;
		sw->step += GetTimer() - start_time;

		swap_pointers(&conc_old, &conc_new);
	}
}

void check_solution(fp_t** conc_new, fp_t** conc_lap, int nx, int ny,
                    fp_t dx, fp_t dy, int nm, fp_t elapsed, fp_t D,
                    fp_t bc[2][2], fp_t* rss)
{
	int i, j;
	fp_t r, cal, car, ca, cn, sum=0.;

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

	for (j = nm/2; j < ny-nm/2; j++) {
		for (i = nm/2; i < nx-nm/2; i++) {
			sum += conc_lap[j][i];
		}
	}

	*rss = sum;
}
