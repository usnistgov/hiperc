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
 \file  cpu-analytic-diffusion/discretization.c
 \brief Implementation of analytical solution functions
*/

#include <math.h>
#include "discretization.h"
#include "numerics.h"

/**
 \brief Update the scalar composition field using analytical solution
*/
void solve_diffusion_equation(fp_t** conc_old, fp_t** conc_new, fp_t** conc_lap, int nx,
                              int ny, fp_t dx, fp_t dy, int nm, fp_t D, fp_t dt, fp_t elapsed)
{
	int i, j;
	fp_t r, cal, car;
	fp_t bc[2][2] = {{1., 1.}, {1., 1.}};

	for (j = nm/2; j < ny-nm/2; j++) {
		for (i = nm/2; i < nx-nm/2; i++) {
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
			conc_new[j][i] = cal + car;
		}
	}
}

/** \} */
