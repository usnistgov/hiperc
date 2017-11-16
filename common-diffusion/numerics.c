/**********************************************************************************
 HiPerC: High Performance Computing Strategies for Boundary Value Problems
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
 \file  numerics.c
 \brief Implementation of Laplacian operator and analytical solution functions
*/

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "numerics.h"

void set_mask(const fp_t dx, const fp_t dy, const int code, fp_t** mask_lap, const int nm)
{
    switch(code) {
	    case 53:
	    	five_point_Laplacian_stencil(dx, dy, mask_lap, nm);
	    	break;
	    case 93:
	    	nine_point_Laplacian_stencil(dx, dy, mask_lap, nm);
	    	break;
	    case 95:
	    	slow_nine_point_Laplacian_stencil(dx, dy, mask_lap, nm);
	    	break;
	    default :
	    	five_point_Laplacian_stencil(dx, dy, mask_lap, nm);
    }

	assert(nm <= MAX_MASK_W);
	assert(nm <= MAX_MASK_H);
}

void five_point_Laplacian_stencil(const fp_t dx, const fp_t dy, fp_t** mask_lap, const int nm)
{
	assert(nm == 3);

	mask_lap[0][1] =  1. / (dy * dy); /* upper */
	mask_lap[1][0] =  1. / (dx * dx); /* left */
	mask_lap[1][1] = -2. * (dx*dx + dy*dy) / (dx*dx * dy*dy); /* middle */
	mask_lap[1][2] =  1. / (dx * dx); /* right */
	mask_lap[2][1] =  1. / (dy * dy); /* lower */
}

void nine_point_Laplacian_stencil(const fp_t dx, const fp_t dy, fp_t** mask_lap, const int nm)
{
	assert(nm == 3);

	mask_lap[0][0] =   1. / (6. * dx * dy); /* upper-left */
	mask_lap[0][1] =   4. / (6. * dy * dy); /* upper-middle */
	mask_lap[0][2] =   1. / (6. * dx * dy); /* upper-right */

	mask_lap[1][0] =   4. / (6. * dx * dx); /* middle-left */
	mask_lap[1][1] = -10. * (dx*dx + dy*dy) / (6. * dx*dx * dy*dy); /* middle */
	mask_lap[1][2] =   4. / (6. * dx * dx); /* middle-right */

	mask_lap[2][0] =   1. / (6. * dx * dy); /* lower-left */
	mask_lap[2][1] =   4. / (6. * dy * dy); /* lower-middle */
	mask_lap[2][2] =   1. / (6. * dx * dy); /* lower-right */
}

void slow_nine_point_Laplacian_stencil(const fp_t dx, const fp_t dy, fp_t** mask_lap, const int nm)
{
	assert(nm == 5);

	mask_lap[0][2] = -1. / (12. * dy * dy); /* upper-upper-middle */

	mask_lap[1][2] =  4. / (3. * dy * dy); /* upper-middle */

	mask_lap[2][0] = -1. / (12. * dx * dx); /* middle-left-left */
	mask_lap[2][1] =  4. / (3. * dx * dx); /* middle-left */
	mask_lap[2][2] = -5. * (dx*dx + dy*dy) / (2. * dx*dx * dy*dy); /* middle */
	mask_lap[2][3] =  4. / (3. * dx * dx); /* middle-right */
	mask_lap[2][4] = -1. / (12. * dx * dx); /* middle-right-right */

	mask_lap[3][2] =  4. / (3. * dy * dy); /* lower-middle */

	mask_lap[4][2] = -1. / (12. * dy * dy); /* lower-lower-middle */
}

fp_t euclidean_distance(const fp_t ax, const fp_t ay,
						const fp_t bx, const fp_t by)
{
	return sqrt((ax - bx) * (ax - bx) + (ay - by) * (ay - by));
}

fp_t manhattan_distance(const fp_t ax, const fp_t ay,
						const fp_t bx, const fp_t by)
{
	return fabs(ax - bx) + fabs(ay - by);
}

fp_t distance_point_to_segment(const fp_t ax, const fp_t ay,
							   const fp_t bx, const fp_t by,
							   const fp_t px, const fp_t py)
{
	const fp_t L2 = (ax - bx) * (ax - bx) + (ay - by) * (ay - by);
	if (L2 == 0.) /* line segment is just a point */
		return euclidean_distance(ax, ay, px, py);
	const fp_t t = fmax(0., fmin(1., ((px - ax) * (bx - ax) + (py - ay) * (by - ay)) / L2));
	const fp_t zx = ax + t * (bx - ax);
	const fp_t zy = ay + t * (by - ay);
	return euclidean_distance(px, py, zx, zy);
}

void analytical_value(const fp_t x, const fp_t t, const fp_t D, fp_t* c)
{
	*c = erfc(x / sqrt(4.0 * D * t));
}

void check_solution(fp_t** conc_new, fp_t** conc_lap, const int nx, const int ny, const fp_t dx, const fp_t dy, const int nm,
                    const fp_t elapsed, const fp_t D, fp_t* rss)
{
	fp_t sum=0.;

	#ifdef __OPENMP
	#pragma omp parallel reduction(+:sum)
	{
		#pragma omp for collapse(2)
	#endif
		for (int j = nm/2; j < ny-nm/2; j++) {
			for (int i = nm/2; i < nx-nm/2; i++) {
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

		#ifdef __OPENMP
		#pragma omp for collapse(2)
		#endif
		for (int j = nm/2; j < ny-nm/2; j++) {
			for (int i = nm/2; i < nx-nm/2; i++) {
				sum += conc_lap[j][i];
			}
		}
	#ifdef __OPENMP
	}
	#endif

	*rss = sum;
}
