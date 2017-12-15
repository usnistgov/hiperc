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
	    case 135:
	    	biharmonic_stencil(dx, dy, mask_lap, nm);
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

void biharmonic_stencil(const fp_t dx, const fp_t dy, fp_t** mask_lap, const int nm)
{
	assert(nm == 5);

	mask_lap[0][2] =  1. / (dy * dy); /* upper-upper-middle */

	mask_lap[1][1] =  2. / (dx * dy); /* upper-left */
	mask_lap[1][2] = -8. / (dy * dy); /* upper-middle */
	mask_lap[1][3] =  2. / (dx * dy); /* upper-right */

	mask_lap[2][0] =  1. / (dx * dx); /* middle-left-left */
	mask_lap[2][1] = -8. / (dx * dx); /* middle-left */
	mask_lap[2][2] = 10. * (dx*dx + dy*dy) / (dx*dx * dy*dy); /* middle */
	mask_lap[2][3] = -8. / (dx * dx); /* middle-right */
	mask_lap[2][4] =  1. / (dx * dx); /* middle-right-right */

	mask_lap[3][1] =  2. / (dx * dy); /* lower-left */
	mask_lap[3][2] = -8. / (dy * dy); /* lower-middle */
	mask_lap[3][3] =  2. / (dx * dy); /* lower-right */

	mask_lap[4][2] =  1. / (dy * dy); /* lower-lower-middle */
}

fp_t grad_sq(fp_t** conc, const int x, const int y,
			 const fp_t dx, const fp_t dy,
			 const int nx, const int ny)
{
	assert(x > 0 && x < nx-1);
	assert(y > 0 && y < ny-1);

	int i=x, j=y;
	fp_t gsq=0.;

	/* gradient in x-direction */
	i += 1;
	const fp_t xhi = conc[j][i];
	i -= 2;
	const fp_t xlo = conc[j][i];
	i += 1;

	gsq += (xhi - xlo) * (xhi - xlo) / (4. * dx * dx);

	/* gradient in y-direction */
	j += 1;
	const fp_t yhi = conc[j][i];
	j -= 2;
	const fp_t ylo = conc[j][i];
	j += 1;

	gsq += (yhi - ylo) * (yhi - ylo) / (4. * dy * dy);

	return gsq;
}

fp_t chem_energy(const fp_t C)
{
	const fp_t Ca  = 0.3;
	const fp_t Cb  = 0.7;
	const fp_t rho = 5.0;

	const fp_t A = C - Ca;
	const fp_t B = Cb - C;

	return rho * A*A * B*B;
}

void free_energy(fp_t** conc_new, fp_t** conc_lap,
				 const fp_t dx, const fp_t dy,
				 const int nx, const int ny, const int nm,
				 const fp_t kappa, fp_t* energy)
{
	const fp_t dV = dx * dy;
	int i, j;
	fp_t sum=0.;

	#ifdef __OPENMP
	#pragma omp parallel reduction(+:sum)
	{
		#pragma omp for collapse(2) private (i,j)
	#endif
		for (j = nm/2; j < ny-nm/2; j++) {
			for (i = nm/2; i < nx-nm/2; i++) {
				const fp_t f = chem_energy(conc_new[j][i]);
				const fp_t g = grad_sq(conc_new, i, j, dx, dy, nx, ny);
				conc_lap[j][i] = dV * (f + 0.5 * kappa * g);
			}
		}

		#ifdef __OPENMP
		#pragma omp for collapse(2) private(i,j)
		#endif
		for (j = nm/2; j < ny-nm/2; j++) {
			for (i = nm/2; i < nx-nm/2; i++) {
				sum += conc_lap[j][i];
			}
		}
	#ifdef __OPENMP
	}
	#endif

	*energy = sum;
}
