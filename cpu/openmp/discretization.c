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

/** \addtogroup openmp
 \{
*/

/**
 \file  cpu/openmp/discretization.c
 \brief Implementation of boundary condition functions with OpenMP threading
*/

#include <math.h>
#include <omp.h>
#include "discretization.h"

/**
 \brief Set number of OpenMP threads to use in parallel code sections
*/
void set_threads(int n)
{
	omp_set_num_threads(n);
}

/**
 \brief Write 5-point Laplacian stencil into convolution mask

 \f$3\times3\f$ mask, 5 values, truncation error \f$\mathcal{O}(\Delta x^2)\f$
*/
void five_point_Laplacian_stencil(fp_t dx, fp_t dy, fp_t** mask_lap)
{
	mask_lap[0][1] =  1. / (dy * dy); /* up */
	mask_lap[1][0] =  1. / (dx * dx); /* left */
	mask_lap[1][1] = -2. * (dx*dx + dy*dy) / (dx*dx * dy*dy); /* middle */
	mask_lap[1][2] =  1. / (dx * dx); /* right */
	mask_lap[2][1] =  1. / (dy * dy); /* down */
}

/**
 \brief Write 9-point Laplacian stencil into convolution mask

 \f$3\times3\f$ mask, 9 values, truncation error \f$\mathcal{O}(\Delta x^4)\f$
*/
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

/**
 \brief Write 9-point Laplacian stencil into convolution mask

 \f$4\times4\f$ mask, 9 values, truncation error \f$\mathcal{O}(\Delta x^4)\f$
 Provided for testing and demonstration of scalability, only:
 as the name indicates, this 9-point stencil is computationally
 more expensive than the \f$3\times3\f$ version. If your code requires
 \f$\mathcal{O}(\Delta x^4)\f$ accuracy, please use nine_point_Laplacian_stencil().
*/
void slow_nine_point_Laplacian_stencil(fp_t dx, fp_t dy, fp_t** mask_lap)
{
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

/**
 \brief Specify which stencil to use for the Laplacian
*/
void set_mask(fp_t dx, fp_t dy, int nm, fp_t** mask_lap)
{
	five_point_Laplacian_stencil(dx, dy, mask_lap);
}

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
	#pragma omp parallel
	{
		int i, j, mi, mj;
		fp_t value;

		#pragma omp for collapse(2)
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
}

/**
 \brief Update the scalar composition field using old and Laplacian values
*/
void solve_diffusion_equation(fp_t** conc_old, fp_t** conc_new, fp_t** conc_lap,
                              int nx, int ny, int nm, fp_t D, fp_t dt, fp_t* elapsed)
{
	#pragma omp parallel
	{
		int i, j;

		#pragma omp for collapse(2)
		for (j = nm/2; j < ny-nm/2; j++)
			for (i = nm/2; i < nx-nm/2; i++)
				conc_new[j][i] = conc_old[j][i] + dt * D * conc_lap[j][i];
	}

	*elapsed += dt;
}

/**
 \brief Compute Euclidean distance between two points, \c a and \c b
*/
fp_t euclidean_distance(fp_t ax, fp_t ay, fp_t bx, fp_t by)
{
	return sqrt((ax - bx) * (ax - bx) + (ay - by) * (ay - by));
}

/**
 \brief Compute Manhattan distance between two points, \c a and \c b
*/
fp_t manhattan_distance(fp_t ax, fp_t ay, fp_t bx, fp_t by)
{
	return fabs(ax - bx) + fabs(ay - by);
}

/**
 \brief Compute minimum distance from point \c p to a line segment bounded by points \c a and \c b

 This function computes the projection of \c p onto \c ab, limiting the
 projected range to [0, 1] to handle projections that fall outside of \c ab.
 Implemented after Grumdrig on Stackoverflow, https://stackoverflow.com/a/1501725.
*/
fp_t distance_point_to_segment(fp_t ax, fp_t ay, fp_t bx, fp_t by, fp_t px, fp_t py)
{
	fp_t L2, t, zx, zy;

	L2 = (ax - bx) * (ax - bx) + (ay - by) * (ay - by);
	if (L2 == 0.) /* line segment is just a point */
		return euclidean_distance(ax, ay, px, py);
	t = fmax(0., fmin(1., ((px - ax) * (bx - ax) + (py - ay) * (by - ay)) / L2));
	zx = ax + t * (bx - ax);
	zy = ay + t * (by - ay);
	return euclidean_distance(px, py, zx, zy);
}

/**
 \brief Analytical solution of the diffusion equation for a carburizing process

 For 1D diffusion through a semi-infinite domain with initial and far-field
 composition \f$ c_{\infty} \f$ and boundary value \f$ c(x=0, t) = c_0 \f$
 with constant diffusivity \e D, the solution to Fick's second law is
 \f[ c(x,t) = c_0 - (c_0 - c_{\infty})\mathrm{erf}\left(\frac{x}{\sqrt{4Dt}}\right) \f]
 which reduces, when \f$ c_{\infty} = 0 \f$, to
 \f[ c(x,t) = c_0\left[1 - \mathrm{erf}\left(\frac{x}{\sqrt{4Dt}}\right)\right]. \f]
*/
void analytical_value(fp_t x, fp_t t, fp_t D, fp_t bc[2][2], fp_t* c)
{
	*c = bc[1][0] * (1.0 - erf(x / sqrt(4.0 * D * t)));
}

/**
 \brief Compare numerical and analytical solutions of the diffusion equation

 Returns the residual sum of squares (RSS), normalized to the domain size.
*/
void check_solution(fp_t** conc_new, int nx, int ny, fp_t dx, fp_t dy, int nm,
                    fp_t elapsed, fp_t D, fp_t bc[2][2], fp_t* rss)
{
	fp_t sum=0.;
	#pragma omp parallel reduction(+:sum)
	{
		int i, j;
		fp_t r, cal, car, ca, cn, trss;

		#pragma omp for collapse(2)
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
				trss = (ca - cn) * (ca - cn) / (fp_t)((nx-1-nm/2) * (ny-1-nm/2));
				sum += trss;
			}
		}
	}

	*rss = sum;
}

/** \} */
