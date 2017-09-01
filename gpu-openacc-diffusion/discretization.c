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

/**
 \file  gpu-openacc-diffusion/discretization.c
 \brief Implementation of boundary condition functions with OpenACC threading
*/

#include <math.h>
#include <omp.h>
#include <openacc.h>
#include "discretization.h"
#include "numerics.h"

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
                              fp_t** mask_lap, int nx, int ny, int nm,
                              fp_t bc[2][2], fp_t D, fp_t dt, fp_t* elapsed,
                              struct Stopwatch* sw)
{
	int i, j;
	double start_time=0.;

	apply_boundary_conditions(conc_old, nx, ny, nm, bc);

	start_time = GetTimer();
	compute_convolution(conc_old, conc_lap, mask_lap, nx, ny, nm);
	sw->conv += GetTimer() - start_time;

	start_time = GetTimer();
	#pragma acc data copyin(conc_old[0:ny][0:nx], conc_lap[0:ny][0:nx]) copyout(conc_new[0:ny][0:nx])
	{
		#pragma acc parallel
		{
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
	sw->step += GetTimer() - start_time;
}

/**
 \brief Compare numerical and analytical solutions of the diffusion equation

 Returns the residual sum of squares (RSS), normalized to the domain size.

 For 1D diffusion through a semi-infinite domain with initial and far-field
 composition \f$ c_{\infty} \f$ and boundary value \f$ c(x=0, t) = c_0 \f$
 with constant diffusivity \e D, the solution to Fick's second law is
 \f[ c(x,t) = c_0 - (c_0 - c_{\infty})\mathrm{erf}\left(\frac{x}{\sqrt{4Dt}}\right) \f]
 which reduces, when \f$ c_{\infty} = 0 \f$, to
 \f[ c(x,t) = c_0\left[1 - \mathrm{erf}\left(\frac{x}{\sqrt{4Dt}}\right)\right]. \f]
*/
void check_solution(fp_t** conc_new, int nx, int ny, fp_t dx, fp_t dy, int nm,
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
					r = distance_point_to_segment(dx * (nm/2), dy * (nm/2),
					                              dx * (nm/2), dy * (ny/2),
					                              dx * i, dy * j);
					z = r / sqrt(4. * D * elapsed);
					z2 = z * z;
					poly_erf = (z < 1.5)
					         ? 2. * z * (1. + z2 * (-1./3 + z2 * (1./10 + z2 * (-1./42 + z2 / 216)))) / sqrt(M_PI)
					         : 1.;
					cal = bc[1][0] * (1. - poly_erf);

					/* shortest distance to right-wall source */
					r = distance_point_to_segment(dx * (nx-1-nm/2), dy * (ny/2),
					                              dx * (nx-1-nm/2), dy * (ny-1-nm/2),
					                              dx * i, dy * j);
					z = r / sqrt(4. * D * elapsed);
					z2 = z * z;
					poly_erf = (z < 1.5)
					         ? 2. * z * (1. + z2 * (-1./3 + z2 * (1./10 + z2 * (-1./42 + z2 / 216)))) / sqrt(M_PI)
					         : 1.;
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
