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
#include "discretization.h"
#include "numerics.h"
#include "timer.h"

/**
 \brief Perform the convolution of the mask matrix with the composition matrix

 If the convolution mask is the Laplacian stencil, the convolution evaluates
 the discrete Laplacian of the composition field. Other masks are possible, for
 example the Sobel filters for edge detection. This function is general
 purpose: as long as the dimensions \c nx, \c ny, and \c nm are properly specified,
 the convolution will be correctly computed.
*/
void compute_convolution(fp_t** conc_old, fp_t** conc_lap, fp_t** mask_lap,
                         int nx, int ny, int nm)
{
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

/**
 \brief Update the scalar composition field using old and Laplacian values
*/
void solve_diffusion_equation(fp_t** conc_old, fp_t** B, fp_t** conc_lap,
                              fp_t** mask_lap, int nx, int ny, int nm,
                              fp_t bc[2][2], fp_t D, fp_t dt, fp_t* elapsed,
                              struct Stopwatch* sw)
{
	double start_time=0.;

	apply_boundary_conditions(conc_old, nx, ny, nm, bc);

	start_time = GetTimer();
	compute_convolution(conc_old, conc_lap, mask_lap, nx, ny, nm);
	sw->conv += GetTimer() - start_time;

	start_time = GetTimer();
	tbb::parallel_for(tbb::blocked_range2d<int>(nm/2, nx-nm/2, nm/2, ny-nm/2),
		[=](const tbb::blocked_range2d<int>& r) {
			for (int j = r.cols().begin(); j != r.cols().end(); j++) {
				for (int i = r.rows().begin(); i != r.rows().end(); i++) {
					B[j][i] = conc_old[j][i] + dt * D * conc_lap[j][i];
				}
			}
		}
	);

	*elapsed += dt;
	sw->step += GetTimer() - start_time;
}

/**
 \brief Comparison algorithm for execution on the block of threads
*/
class ResidualSumOfSquares2D {
	fp_t** my_conc_new;
	int my_nx;
	int my_ny;
	fp_t my_dx;
	fp_t my_dy;
	int my_nm;
	fp_t my_elapsed;
	fp_t my_D;
	fp_t my_c;

	public:
		fp_t my_rss;

		/* constructors */
		ResidualSumOfSquares2D(fp_t** conc_new, int nx, int ny, fp_t dx, fp_t dy, int nm,
		                       fp_t elapsed, fp_t D, fp_t c)
		                      : my_conc_new(conc_new), my_nx(nx), my_ny(ny),
		                        my_dx(dx), my_dy(dy), my_nm(nm),
		                        my_elapsed(elapsed), my_D(D), my_c(c), my_rss(0.0) {}
		ResidualSumOfSquares2D(ResidualSumOfSquares2D& a, tbb::split)
		                      : my_conc_new(a.my_conc_new), my_nx(a.my_nx), my_ny(a.my_ny),
		                        my_dx(a.my_dx), my_dy(a.my_dy), my_nm(a.my_nm),
		                        my_elapsed(a.my_elapsed), my_D(a.my_D), my_c(a.my_c),
		                        my_rss(0.0) {}

		/* modifier */
		void operator()(const tbb::blocked_range2d<int>& r)
		{
			fp_t** conc_new = my_conc_new;
			int nx = my_nx;
			int ny = my_ny;
			fp_t dx = my_dx;
			fp_t dy = my_dy;
			int nm = my_nm;
			fp_t elapsed = my_elapsed;
			fp_t D = my_D;
			fp_t c = my_c;
			fp_t sum = my_rss;
			fp_t bc[2][2] = {{c, c}, {c, c}};

			for (int j = r.cols().begin(); j != r.cols().end(); j++) {
				for (int i = r.rows().begin(); i != r.rows().end(); i++) {
					fp_t r, cal, car, ca, cn;

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
					sum += (ca - cn) * (ca - cn) / (fp_t)((nx-1-nm/2) * (ny-1-nm/2));
				}
			}
			my_rss = sum;
		}

		/* reduction */
		void join(const ResidualSumOfSquares2D& a)
		{
			my_rss += a.my_rss;
		}
};

/**
 \brief Compare numerical and analytical solutions of the diffusion equation

 Returns the residual sum of squares (RSS), normalized to the domain size.
*/
void check_solution(fp_t** conc_new, int nx, int ny, fp_t dx, fp_t dy, int nm,
                    fp_t elapsed, fp_t D, fp_t bc[2][2], fp_t* rss)
{
	ResidualSumOfSquares2D R(conc_new, nx, ny, dx, dy, nm, elapsed, D, bc[1][0]);

	tbb::parallel_reduce(tbb::blocked_range2d<int>(nm/2, nx-nm/2, nm/2, ny-nm/2), R);

	*rss = R.my_rss;
}
