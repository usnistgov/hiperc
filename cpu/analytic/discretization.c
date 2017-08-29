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

/** \addtogroup analytic
 \{
*/

/**
 \file  cpu/analytic/discretization.c
 \brief Implementation of analytical solution functions
*/

#include <math.h>
#include "discretization.h"

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
 which reduces, when \f$ c_{\infty} = 0 \f$ and \f$ c_0 = 1 \f$, to
 \f[ c(x,t) = 1 - \mathrm{erf}\left(\frac{x}{\sqrt{4Dt}}\right). \f]
*/
void analytical_value(fp_t x, fp_t t, fp_t D, fp_t* c)
{
	*c = 1.0 - erf(x / sqrt(4.0 * D * t));
}

/**
 \brief Update the scalar composition field using analytical solution
*/
void solve_diffusion_equation(fp_t** conc_old, fp_t** conc_new, fp_t** conc_lap, int nx,
                              int ny, fp_t dx, fp_t dy, int nm, fp_t D, fp_t dt, fp_t elapsed)
{
	int i, j;
	fp_t r, cal, car;

	for (j = nm/2; j < ny-nm/2; j++) {
		for (i = nm/2; i < nx-nm/2; i++) {
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
			conc_new[j][i] = cal + car;
		}
	}
}

/** \} */
