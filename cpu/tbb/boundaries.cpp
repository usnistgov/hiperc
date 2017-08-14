/*
	File: boundaries.c
	Role: implementation of boundary condition functions with TBB threading

	Questions/comments to trevor.keller@nist.gov
	Bugs/requests to https://github.com/tkphd/accelerator-testing
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

class ApplyInitialConditions2D {
	double** my_A;
	double my_c;

	public:
		/* constructor */
		ApplyInitialConditions2D(double** A, double c) : my_A(A), my_c(c) {}

		/* modifier */
		void operator()(const tbb::blocked_range2d<int>& r) const
		{
			double** A = my_A;
			double c = my_c;

			for (int j = r.cols().begin(); j != r.cols().end(); j++)
				for (int i = r.rows().begin(); i != r.rows().end(); i++)
					A[j][i] = c;
		}
};

class ApplyLeftBoundaryValues {
	double** my_A;
	double my_c;
	int my_x;

	public:
		/* constructor */
		ApplyLeftBoundaryValues(double** A, double c, int x) : my_A(A), my_c(c), my_x(x) {}

		/* modifier */
		void operator()(const tbb::blocked_range<int>& r) const
		{
			double** A = my_A;
			double c = my_c;
			int x = my_x;

			for (int j = r.begin(); j != r.end(); j++)
				A[j][x] = c;
		}
};

class ApplyRightBoundaryValues {
	double** my_A;
	double my_c;
	double my_x;

	public:
		/* constructor */
		ApplyRightBoundaryValues(double** A, double c, int x) : my_A(A), my_c(c), my_x(x) {}

		/* modifier */
		void operator()(const tbb::blocked_range<int>& r) const
		{
			double** A = my_A;
			double c = my_c;
			int x = my_x;

			for (int j = r.begin(); j != r.end(); j++)
				A[j][x] = c;
		}
};

void apply_initial_conditions(double** A, int nx, int ny, double bc[2][2])
{
	const int tbb_bs = 16;

	tbb::parallel_for( tbb::blocked_range2d<int>(1, ny, tbb_bs, 1, nx, tbb_bs), ApplyInitialConditions2D(A, bc[0][0]) );

	tbb::parallel_for( tbb::blocked_range<int>(1, ny/2, tbb_bs), ApplyLeftBoundaryValues(A, bc[1][0], 1) );

	tbb::parallel_for( tbb::blocked_range<int>(ny/2, ny-1, tbb_bs), ApplyRightBoundaryValues(A, bc[1][1], nx-2) );
}

class ApplyBoundaryConditions {
	double** my_A;
	int my_nx;
	int my_ny;

	public:
		/* constructor */
		ApplyBoundaryConditions(double** A, int nx, int ny) : my_A(A), my_nx(nx), my_ny(ny) {}

		/* modifier */
		void operator()(const tbb::blocked_range2d<int>& r) const
		{
			double** A = my_A;
			int nx = my_nx;
			int ny = my_ny;

			for (int j = r.cols().begin(); j != r.cols().end(); j++) {
				A[j][0] = A[j][1]; /* left boundary */
				A[j][nx-1] = A[j][nx-2]; /* right boundary */
			}

			for (int i = r.rows().begin(); i != r.rows().end(); i++) {
				A[0][i] = A[1][i]; /* bottom boundary */
				A[ny-1][i] = A[ny-2][i]; /* top boundary */
			}

		}
};

void apply_boundary_conditions(double** A, int nx, int ny, double bc[2][2])
{
	const int tbb_bs = 16;

	tbb::parallel_for( tbb::blocked_range<int>(1, ny/2, tbb_bs), ApplyLeftBoundaryValues(A, bc[1][0], 1) );

	tbb::parallel_for( tbb::blocked_range<int>(ny/2, ny-1, tbb_bs), ApplyRightBoundaryValues(A, bc[1][1], nx-2) );

	tbb::parallel_for( tbb::blocked_range2d<int>(1, ny-1, tbb_bs, 1, nx-1, tbb_bs), ApplyBoundaryConditions(A, nx, ny) );
}
