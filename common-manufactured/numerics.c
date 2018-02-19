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
 \brief Implementation of Laplacian operator and manufactured solution functions
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

void manufactured_solution(const fp_t x,  const fp_t y, const fp_t t,
                           const fp_t A1, const fp_t A2,
                           const fp_t B1, const fp_t B2,
                           const fp_t C2, const fp_t kappa,
                           fp_t* eta)
{
	/* Equation 2 */
	fp_t alpha = 0.25 + A1 * t * sin(B1 * x) + A2 * sin(B2 * x + C2 * t);
	*eta = 0.5 * (1. - tanh((y - alpha)/sqrt(2. * kappa)));
}

void compute_L2_norm(fp_t** conc_new, fp_t** conc_lap,
					 const fp_t dx, const fp_t dy,
					 const fp_t elapsed,
					 const int  nx, const int  ny, const int  nm,
					 const fp_t A1, const fp_t A2,
					 const fp_t B1, const fp_t B2,
					 const fp_t C2, const fp_t kappa,
					 fp_t* L2)
{
	/* Equation 9 */
	fp_t sum = 0.;
	int i, j;

	#ifdef __OPENMP
	#pragma omp parallel reduction(+:sum)
	{
		#pragma omp for collapse(2) private (i,j)
	#endif
		for (j = nm/2; j < ny-nm/2; j++) {
			for (i = nm/2; i < nx-nm/2; i++) {
				const fp_t x = dx * (i - nm/2);
				const fp_t y = dy * (j - nm/2);

				/* numerical solution */
				const fp_t etaN = conc_new[j][i];

				/* manufactured solution */
				fp_t etaM;
				manufactured_solution(x, y, elapsed, A1, A2, B1, B2, C2, kappa, &etaM);

				/* error */
				conc_lap[j][i] = (etaN - etaM) * (etaN - etaM);
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

	*L2 = sqrt(dx * dy * sum);
}
