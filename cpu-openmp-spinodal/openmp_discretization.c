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
 \file  openmp_discretization.c
 \brief Implementation of boundary condition functions with OpenMP threading
*/

#include <math.h>
#include <omp.h>
#include "boundaries.h"
#include "mesh.h"
#include "numerics.h"
#include "timer.h"

fp_t dfdc(const fp_t C)
{
	const fp_t Ca  = 0.3;
	const fp_t Cb  = 0.7;
	const fp_t rho = 5.0;

	const fp_t A = C - Ca;
	const fp_t B = Cb - C;

	return 2.0 * rho * A * B * (Ca + Cb - 2.0 * C);
}

void compute_laplacian(fp_t** conc_old, fp_t** conc_lap,
					   fp_t** mask_lap, const fp_t kappa,
					   const int nx, const int ny, const int nm)
{
	#pragma omp parallel for collapse(2)
	for (int j = nm/2; j < ny-nm/2; j++) {
		for (int i = nm/2; i < nx-nm/2; i++) {
			fp_t value = 0.0;
			for (int mj = -nm/2; mj < nm/2+1; mj++) {
				for (int mi = -nm/2; mi < nm/2+1; mi++) {
					value += mask_lap[mj+nm/2][mi+nm/2] * conc_old[j+mj][i+mi];
				}
			}
			conc_lap[j][i] = dfdc(conc_old[j][i]) - kappa * value;
		}
	}
}

void compute_divergence(fp_t** conc_lap, fp_t** conc_div, fp_t** mask_lap,
                         const int nx, const int ny, const int nm)
{
	#pragma omp parallel for collapse(2)
	for (int j = nm/2; j < ny-nm/2; j++) {
		for (int i = nm/2; i < nx-nm/2; i++) {
			fp_t value = 0.0;
			for (int mj = -nm/2; mj < nm/2+1; mj++) {
				for (int mi = -nm/2; mi < nm/2+1; mi++) {
					value += mask_lap[mj+nm/2][mi+nm/2] * conc_lap[j+mj][i+mi];
				}
			}
			conc_div[j][i] = value;
		}
	}
}

void update_composition(fp_t** conc_old, fp_t** conc_div, fp_t** conc_new,
						const int nx, const int ny, const int nm,
						const fp_t M, const fp_t dt)
{
	#pragma omp parallel for collapse(2)
	for (int j = nm/2; j < ny - nm/2; j++) {
		for (int i = nm/2; i < nx - nm/2; i++) {
			conc_new[j][i] = conc_old[j][i] + dt * M * conc_div[j][i];
		}
	}
}
