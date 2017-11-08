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
 \file  opencl_boundaries.c
 \brief Implementation of boundary condition functions with OpenCL acceleration
*/

#include <math.h>
#include <omp.h>
#include <CL/cl.h>
#include "boundaries.h"
#include "numerics.h"

void set_boundaries(fp_t bc[2][2])
{
	/* Change these values to your liking: */
	fp_t clo = 0.0, chi = 1.0;

	bc[0][0] = clo; /* bottom boundary */
	bc[0][1] = clo; /* top boundary */
	bc[1][0] = chi; /* left boundary */
	bc[1][1] = chi; /* right boundary */
}

void apply_initial_conditions(fp_t** conc, const int nx, const int ny, const int nm, fp_t bc[2][2])
{
	#pragma omp parallel
	{
		int i, j;

		#pragma omp for collapse(2)
		for (j = 0; j < ny; j++)
			for (i = 0; i < nx; i++)
				conc[j][i] = bc[0][0];

		#pragma omp for collapse(2)
		for (j = 0; j < ny/2; j++)
			for (i = 0; i < 1+nm/2; i++)
				conc[j][i] = bc[1][0]; /* left half-wall */

		#pragma omp for collapse(2)
		for (j = ny/2; j < ny; j++)
			for (i = nx-1-nm/2; i < nx; i++)
				conc[j][i] = bc[1][1]; /* right half-wall */
	}
}
