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
 \file  openacc_discretization.c
 \brief Implementation of boundary condition functions with OpenACC threading
*/

#include <math.h>
#include <omp.h>
#include <openacc.h>
#include "mesh.h"
#include "openacc_kernels.h"

void convolution_kernel(fp_t** conc_old, fp_t** conc_lap, fp_t** mask_lap, const int nx, const int ny, const int nm)
{
	#pragma acc declare present(conc_old[0:ny][0:nx], conc_lap[0:ny][0:nx], mask_lap[0:nm][0:nm])
	#pragma acc parallel
	{
		#pragma acc loop collapse(2)
		for (int j = nm/2; j < ny-nm/2; j++) {
			for (int i = nm/2; i < nx-nm/2; i++) {
				fp_t value = 0.;
				#pragma acc loop seq collapse(2)
				for (int mj = -nm/2; mj < 1+nm/2; mj++) {
					for (int mi = -nm/2; mi < 1+nm/2; mi++) {
						value += mask_lap[mj+nm/2][mi+nm/2] * conc_old[j+mj][i+mi];
					}
				}
				conc_lap[j][i] = value;
			}
		}
	}
}

void diffusion_kernel(fp_t** conc_old, fp_t** conc_new, fp_t** conc_lap,
                      const int nx, const int ny, const int nm, const fp_t D, const fp_t dt)
{
	#pragma acc declare present(conc_old[0:ny][0:nx], conc_new[0:ny][0:nx], conc_lap[0:ny][0:nx])
	#pragma acc parallel
	{
		#pragma acc loop collapse(2)
		for (int j = nm/2; j < ny-nm/2; j++) {
			for (int i = nm/2; i < nx-nm/2; i++) {
				conc_new[j][i] = conc_old[j][i] + dt * D * conc_lap[j][i];
			}
		}
	}
}
