/*
	File: discretization.c
	Role: implementation of discretized mathematical operations with TBB threading

	Questions/comments to trevor.keller@nist.gov
	Bugs/requests to https://github.com/usnistgov/phasefield-accelerator-benchmarks
*/

#include <math.h>
#include <tbb/tbb.h>
#include <tbb/task_scheduler_init.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include <tbb/blocked_range2d.h>

#include "diffusion.h"

void set_threads(int n)
{
	tbb::task_scheduler_init init(n);
}

void five_point_Laplacian_stencil(fp_t dx, fp_t dy, fp_t** mask_lap)
{
	mask_lap[0][1] =  1. / (dy * dy); /* up */
	mask_lap[1][0] =  1. / (dx * dx); /* left */
	mask_lap[1][1] = -2. * (dx*dx + dy*dy) / (dx*dx * dy*dy); /* middle */
	mask_lap[1][2] =  1. / (dx * dx); /* right */
	mask_lap[2][1] =  1. / (dy * dy); /* down */
}

void nine_point_Laplacian_stencil(fp_t dx, fp_t dy, fp_t** mask_lap)
{
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

void slow_nine_point_Laplacian_stencil(fp_t dx, fp_t dy, fp_t** mask_lap)
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

void solve_diffusion_equation(fp_t** conc_old, fp_t** B, fp_t** conc_lap,
                              int nx, int ny, int nm, fp_t D, fp_t dt, fp_t* elapsed)
{
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
}

void analytical_value(fp_t x, fp_t t, fp_t D, fp_t chi, fp_t* c)
{
	*c = chi * (1.0 - erf(x / sqrt(4.0 * D * t)));
}

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
		ResidualSumOfSquares2D(fp_t** conc_new, int nx, int ny, fp_t dx, fp_t dy, int nm, fp_t elapsed, fp_t D, fp_t c)
		                      : my_conc_new(conc_new), my_nx(nx), my_ny(ny), my_dx(dx), my_dy(dy), my_nm(nm), my_elapsed(elapsed), my_D(D), my_c(c), my_rss(0.0) {}
		ResidualSumOfSquares2D(ResidualSumOfSquares2D& a, tbb::split)
		                      : my_conc_new(a.my_conc_new), my_nx(a.my_nx), my_ny(a.my_ny), my_dx(a.my_dx), my_dy(a.my_dy), my_nm(a.my_nm), my_elapsed(a.my_elapsed), my_D(a.my_D), my_c(a.my_c), my_rss(0.0) {}

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

			for (int j = r.cols().begin(); j != r.cols().end(); j++) {
				for (int i = r.rows().begin(); i != r.rows().end(); i++) {
					fp_t x, cal, car, ca, cn;

					/* numerical solution */
					cn = conc_new[j][i];

					/* shortest distance to left-wall source */
					x = (j < ny/2) ?
					    dx * (i - nm/2) :
					    sqrt(dx*dx * (i - nm/2) * (i - nm/2) + dy*dy * (j - ny/2) * (j - ny/2));
					analytical_value(x, elapsed, D, c, &cal);

					/* shortest distance to right-wall source */
					x = (j >= ny/2) ?
					    dx * (nx-1-nm/2 - i) :
					    sqrt(dx*dx * (nx-1-nm/2 - i)*(nx-1-nm/2 - i) + dy*dy * (ny/2 - j)*(ny/2 - j));
					analytical_value(x, elapsed, D, c, &car);

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

void check_solution(fp_t** conc_new, int nx, int ny, fp_t dx, fp_t dy, int nm, fp_t elapsed, fp_t D, fp_t bc[2][2], fp_t* rss)
{
	ResidualSumOfSquares2D R(conc_new, nx, ny, dx, dy, nm, elapsed, D, bc[1][0]);

	tbb::parallel_reduce(tbb::blocked_range2d<int>(nm/2, nx-nm/2, nm/2, ny-nm/2), R);

	*rss = R.my_rss;
}
