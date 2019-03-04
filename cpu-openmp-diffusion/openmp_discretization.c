/**********************************************************************************
 HiPerC: High Performance Computing Strategies for Boundary Value Problems
 Written by Trevor Keller and available from https://github.com/usnistgov/hiperc
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

void compute_convolution(fp_t** conc_old, fp_t** conc_lap, fp_t** mask_lap,
                         const int nx, const int ny, const int nm)
{
	#pragma omp parallel
	{
		#pragma omp for collapse(2)
		for (int j = nm/2; j < ny-nm/2; j++) {
			for (int i = nm/2; i < nx-nm/2; i++) {
				fp_t value = 0.0;
				for (int mj = -nm/2; mj < nm/2+1; mj++) {
					for (int mi = -nm/2; mi < nm/2+1; mi++) {
						value += mask_lap[mj+nm/2][mi+nm/2] * conc_old[j+mj][i+mi];
					}
				}
				conc_lap[j][i] = value;
			}
		}
	}
}

void update_composition(fp_t** conc_old, fp_t** conc_lap, fp_t** conc_new,
				   const int nx, const int ny, const int nm,
				   const fp_t D, const fp_t dt)
{
	#pragma omp parallel for collapse(2)
	for (int j = nm/2; j < ny - nm/2; j++) {
		for (int i = nm/2; i < nx - nm/2; i++) {
			conc_new[j][i] = conc_old[j][i] + dt * D * conc_lap[j][i];
		}
	}
}
