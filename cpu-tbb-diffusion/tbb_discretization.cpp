/**********************************************************************************
 HiPerC: High Performance Computing Strategies for Boundary Value Problems
 Written by Trevor Keller and available from https://github.com/usnistgov/hiperc
 **********************************************************************************/

/**
 \file  tbb_discretization.cpp
 \brief Implementation of boundary condition functions with TBB threading
*/

#include <math.h>
#include <tbb/tbb.h>
#include <tbb/task_scheduler_init.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include <tbb/blocked_range2d.h>
#include "boundaries.h"
#include "mesh.h"
#include "numerics.h"
#include "timer.h"

void compute_convolution(fp_t** conc_old, fp_t** conc_lap, fp_t** mask_lap,
                         const int nx, const int ny, const int nm)
{
	/* Lambda function executed on each thread, solving convolution	*/
	tbb::parallel_for(tbb::blocked_range2d<int>(nm/2, nx-nm/2, nm/2, ny-nm/2),
		[=](const tbb::blocked_range2d<int>& r) {
			for (int j = r.cols().begin(); j != r.cols().end(); j++) {
				for (int i = r.rows().begin(); i != r.rows().end(); i++) {
					fp_t value = 0.0;
					for (int mj = -nm/2; mj < nm/2+1; mj++) {
						for (int mi = -nm/2; mi < nm/2+1; mi++) {
							value += mask_lap[mj+nm/2][mi+nm/2] * conc_old[j+mj][i+mi];
						}
					}
					conc_lap[j][i] = value;
				}
			}
		}
	);
}

void update_composition(fp_t** conc_old, fp_t** conc_lap, fp_t** conc_new,
                        const int nx, const int ny, const int nm,
						const fp_t D, const fp_t dt)
{
	/* Lambda function executed on each thread, updating diffusion equation */
	tbb::parallel_for(tbb::blocked_range2d<int>(nm/2, nx-nm/2, nm/2, ny-nm/2),
		[=](const tbb::blocked_range2d<int>& r) {
			for (int j = r.cols().begin(); j != r.cols().end(); j++) {
				for (int i = r.rows().begin(); i != r.rows().end(); i++) {
					conc_new[j][i] = conc_old[j][i] + dt * D * conc_lap[j][i];
				}
			}
		}
	);
}

void check_solution_lambda(fp_t** conc_new, fp_t** conc_lap, const int nx, const int ny,
						   const fp_t dx, const fp_t dy, const int nm, const fp_t elapsed, const fp_t D,
						   fp_t* rss)
{
	/* Note: tbb::parallel_reduce can only operate on a blocked_range, */
	/* *not* a blocked_range2d. This requires some creativity to get around. */

	/* Lambda function executed on each thread, zeroing conc_lap */
	tbb::parallel_for
	(
		tbb::blocked_range2d<int>(0, nx, 0, ny),
		[=](const tbb::blocked_range2d<int>& r) {
			for (int j = r.cols().begin(); j != r.cols().end(); j++) {
				for (int i = r.rows().begin(); i != r.rows().end(); i++) {
					conc_lap[j][i] = 0.;
				}
			}
		}
	);

	/* Lambda function executed on each thread, checking local values */
	tbb::parallel_for
	(
		tbb::blocked_range2d<int>(nm/2, nx-nm/2, nm/2, ny-nm/2),
		[=](const tbb::blocked_range2d<int>& r) {
			for (int j = r.cols().begin(); j != r.cols().end(); j++) {
				for (int i = r.rows().begin(); i != r.rows().end(); i++) {
					fp_t cal, car, r;

					/* numerical solution */
					const fp_t cn = conc_new[j][i];

					/* shortest distance to left-wall source */
					r = distance_point_to_segment(dx * (nm/2), dy * (nm/2),
					                              dx * (nm/2), dy * (ny/2),
					                              dx * i, dy * j);
					analytical_value(r, elapsed, D, &cal);

					/* shortest distance to right-wall source */
					r = distance_point_to_segment(dx * (nx-1-nm/2), dy * (ny/2),
					                              dx * (nx-1-nm/2), dy * (ny-1-nm/2),
					                              dx * i, dy * j);
					analytical_value(r, elapsed, D, &car);

					/* superposition of analytical solutions */
					const fp_t ca = cal + car;

					/* residual sum of squares (RSS) */
					conc_lap[j][i] = (ca - cn) * (ca - cn) / (fp_t)((nx-1-nm/2) * (ny-1-nm/2));
				}
			}
		}
	);

	/* Lambda function executed on each thread, summing up the vector */
	*rss = tbb::parallel_reduce
	(
		tbb::blocked_range<fp_t*>(conc_lap[0], conc_lap[0] + nx*ny), 0.,
		[](const tbb::blocked_range<fp_t*>& r, fp_t sum)->fp_t {
			for (fp_t* p = r.begin(); p != r.end(); p++) {
				sum += *p;
			}
			return sum;
		},
		[](fp_t x, fp_t y)->fp_t {
			return x+y;
		}
	);
}
