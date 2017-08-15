/*
	File: boundaries.c
	Role: implementation of boundary condition functions with TBB threading

	Questions/comments to trevor.keller@nist.gov
	Bugs/requests to https://github.com/usnistgov/phasefield-accelerator-benchmarks
*/

#include <math.h>
#include <tbb/tbb.h>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <tbb/blocked_range2d.h>

#include "diffusion.h"

void set_boundaries(double bc[2][2])
{
	/* indexing is A[y][x], so bc = [[ylo,yhi], [xlo,xhi]] */
	double clo = 0.0, chi = 1.0;
	bc[0][0] = clo; /* bottom boundary */
	bc[0][1] = clo; /* top boundary */
	bc[1][0] = chi; /* left boundary */
	bc[1][1] = chi; /* right boundary */
}

void apply_initial_conditions(double** A, int nx, int ny, int nm, double bc[2][2])
{
	const int tbb_bs = 16;

	/* apply flat field values  (lambda function) */
	tbb::parallel_for(tbb::blocked_range2d<int>(nm/2, ny, tbb_bs, nm/2, nx, tbb_bs),
		[=](const tbb::blocked_range2d<int>& r) {
			for (int j = r.cols().begin(); j != r.cols().end(); j++) {
				for (int i = r.rows().begin(); i != r.rows().end(); i++) {
					A[j][i] = bc[0][0];
				}
			}
		}
	);

	/* apply left boundary values  (lambda function) */
	tbb::parallel_for(tbb::blocked_range<int>(nm/2, ny/2, tbb_bs),
		[=](const tbb::blocked_range<int>& r) {
			for (int j = r.begin(); j != r.end(); j++) {
				A[j][nm/2] = bc[1][0];
			}
		}
	);

	/* apply right boundary values  (lambda function) */
	tbb::parallel_for( tbb::blocked_range<int>(ny/2, ny-nm/2, tbb_bs),
		[=](const tbb::blocked_range<int>& r) {
			for (int j = r.begin(); j != r.end(); j++) {
				A[j][nx-nm/2-1] = bc[1][1];
			}
		}
	);
}

void apply_boundary_conditions(double** A, int nx, int ny, int nm, double bc[2][2])
{
	const int tbb_bs = 16;

	/* apply left boundary values  (lambda function) */
	tbb::parallel_for(tbb::blocked_range<int>(nm/2, ny/2, tbb_bs),
		[=](const tbb::blocked_range<int>& r) {
			for (int j = r.begin(); j != r.end(); j++) {
				A[j][nm/2] = bc[1][0];
			}
		}
	);

	/* apply right boundary values  (lambda function) */
	tbb::parallel_for( tbb::blocked_range<int>(ny/2, ny-nm/2, tbb_bs),
		[=](const tbb::blocked_range<int>& r) {
			for (int j = r.begin(); j != r.end(); j++) {
				A[j][nx-nm/2-1] = bc[1][1];
			}
		}
	);

	/* apply no-flux boundary conditions  (lambda function) */
	tbb::parallel_for(tbb::blocked_range2d<int>(nm/2, ny-nm/2, tbb_bs, nm/2, nx-nm/2, tbb_bs),
		[=](const tbb::blocked_range2d<int>& r) {
			for (int j = r.cols().begin(); j != r.cols().end(); j++) {
				A[j][nm/2-1] = A[j][nm/2]; /* left boundary */
				A[j][nx-nm/2] = A[j][nx-nm/2-1]; /* right boundary */
			}

			for (int i = r.rows().begin(); i != r.rows().end(); i++) {
				A[nm/2-1][i] = A[nm/2][i]; /* bottom boundary */
				A[ny-nm/2][i] = A[ny-nm/2-1][i]; /* top boundary */
			}
		}
	);
}
