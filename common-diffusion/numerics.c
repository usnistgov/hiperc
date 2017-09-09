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

void set_mask(fp_t dx, fp_t dy, int code, fp_t** mask_lap, int nm)
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

void five_point_Laplacian_stencil(fp_t dx, fp_t dy, fp_t** mask_lap, int nm)
{
	assert(nm == 3);

	mask_lap[0][1] =  1. / (dy * dy); /* upper */
	mask_lap[1][0] =  1. / (dx * dx); /* left */
	mask_lap[1][1] = -2. * (dx*dx + dy*dy) / (dx*dx * dy*dy); /* middle */
	mask_lap[1][2] =  1. / (dx * dx); /* right */
	mask_lap[2][1] =  1. / (dy * dy); /* lower */
}

void nine_point_Laplacian_stencil(fp_t dx, fp_t dy, fp_t** mask_lap, int nm)
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

void slow_nine_point_Laplacian_stencil(fp_t dx, fp_t dy, fp_t** mask_lap, int nm)
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

fp_t euclidean_distance(fp_t ax, fp_t ay, fp_t bx, fp_t by)
{
	return sqrt((ax - bx) * (ax - bx) + (ay - by) * (ay - by));
}

fp_t manhattan_distance(fp_t ax, fp_t ay, fp_t bx, fp_t by)
{
	return fabs(ax - bx) + fabs(ay - by);
}

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

void analytical_value(fp_t x, fp_t t, fp_t D, fp_t bc[2][2], fp_t* c)
{
	*c = bc[1][0] * (1.0 - erf(x / sqrt(4.0 * D * t)));
}
