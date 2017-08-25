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

/** \addtogroup GPU
 \{
*/

/** \addtogroup openacc
 \{
*/

/**
 \file  gpu/openacc/discretization.c
 \brief Implementation of boundary condition functions with OpenACC threading
*/

#include <math.h>
#include <omp.h>
#include <openacc.h>
#include "discretization.h"

/**
 \brief Set number of OpenMP threads to use in CPU code sections
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
 more expensive than the \f$3\times3\f$ version. If your code requires \f$\mathcal{O}(\Delta x^4)\f$
 accuracy, please use nine_point_Laplacian_stencil().
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
 purpose: as long as the dimensions \c nx, \c ny, and \c nm are properly specified, the
 convolution will be correctly computed.
*/
void compute_convolution(fp_t** conc_old, fp_t** conc_lap, fp_t** mask_lap, int nx, int ny, int nm, int bs)
{
	#pragma acc data copyin(conc_old[0:ny][0:nx], mask_lap[0:nm][0:nm]) copyout(conc_lap[0:ny][0:nx])
	{
		#pragma acc parallel
		{
			int i, j, mi, mj;
			fp_t value;

			#pragma acc loop
			for (j = nm/2; j < ny-nm/2; j++) {
				#pragma acc loop
				for (i = nm/2; i < nx-nm/2; i++) {
					value = 0.;
					for (mj = -nm/2; mj < 1+nm/2; mj++) {
						for (mi = -nm/2; mi < 1+nm/2; mi++) {
							value += mask_lap[mj+nm/2][mi+nm/2] * conc_old[j+mj][i+mi];
						}
					}
					conc_lap[j][i] = value;
				}
			}
		}
	}
}

/**
 \brief Update the scalar composition field using old and Laplacian values
*/
void solve_diffusion_equation(fp_t** conc_old, fp_t** conc_new, fp_t** conc_lap,
                              int nx, int ny, int nm, int bs,
                              fp_t D, fp_t dt, fp_t* elapsed)
{
	#pragma acc data copyin(conc_old[0:ny][0:nx], conc_lap[0:ny][0:nx]) copyout(conc_new[0:ny][0:nx])
	{
		#pragma acc parallel
		{
			int i, j;

			#pragma acc loop
			for (j = nm/2; j < ny-nm/2; j++) {
				#pragma acc loop
				for (i = nm/2; i < nx-nm/2; i++) {
					conc_new[j][i] = conc_old[j][i] + dt * D * conc_lap[j][i];
				}
			}
		}
	}

	*elapsed += dt;
}

/**
 \brief Compare numerical and analytical solutions of the diffusion equation

 Returns the residual sum of squares (RSS), normalized to the domain size.
*/
void check_solution(fp_t** conc_new,
                    int nx, int ny, fp_t dx, fp_t dy, int nm, int bs,
                    fp_t elapsed, fp_t D, fp_t bc[2][2], fp_t* rss)
{
	/* OpenCL does not have a GPU-based erf() definition, using Maclaurin series approximation */
	fp_t sum=0.;
	#pragma acc data copyin(conc_new[0:ny][0:nx], bc[0:2][0:2]) copy(sum)
	{
		#pragma acc parallel reduction(+:sum)
		{
			int i, j;
			fp_t ca, cal, car, cn, poly_erf, r, trss, z, z2;

			#pragma acc loop
			for (j = nm/2; j < ny-nm/2; j++) {
				#pragma acc loop
				for (i = nm/2; i < nx-nm/2; i++) {
					/* numerical solution */
					cn = conc_new[j][i];

					/* shortest distance to left-wall source */
					r = (j < ny/2) ? dx * (i - nm/2) : sqrt(dx*dx * (i - nm/2) * (i - nm/2) + dy*dy * (j - ny/2) * (j - ny/2));
					z = r / sqrt(4. * D * elapsed);
					z2 = z * z;
					poly_erf = (z < 1.5) ? 2. * z * (1. + z2 * (-1./3 + z2 * (1./10 + z2 * (-1./42 + z2 / 216)))) / sqrt(M_PI) : 1.;
					cal = bc[1][0] * (1. - poly_erf);

					/* shortest distance to right-wall source */
					r = (j >= ny/2) ? dx * (nx-1-nm/2 - i) : sqrt(dx*dx * (nx-1-nm/2 - i)*(nx-1-nm/2 - i) + dy*dy * (ny/2 - j)*(ny/2 - j));
					z = r / sqrt(4. * D * elapsed);
					z2 = z * z;
					poly_erf = (z < 1.5) ? 2. * z * (1. + z2 * (-1./3 + z2 * (1./10 + z2 * (-1./42 + z2 / 216)))) / sqrt(M_PI) : 1.;
					car = bc[1][0] * (1. - poly_erf);

					/* superposition of analytical solutions */
					ca = cal + car;

					/* residual sum of squares (RSS) */
					trss = (ca - cn) * (ca - cn) / (fp_t)((nx-1-nm/2) * (ny-1-nm/2));
					sum += trss;
				}
			}
		}
	}

	*rss = sum;
}

/** \} */
/** \} */
