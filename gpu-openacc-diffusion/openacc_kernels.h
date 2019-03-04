/**********************************************************************************
 HiPerC: High Performance Computing Strategies for Boundary Value Problems
 Written by Trevor Keller and available from https://github.com/usnistgov/hiperc
 **********************************************************************************/

/**
 \file  openacc_kernels.h
 \brief Declaration of functions to execute on the GPU (OpenACC kernels)
*/

/** \cond SuppressGuard */
#ifndef _OPENACC_KERNELS_H_
#define _OPENACC_KERNELS_H_
/** \endcond */

#include "numerics.h"

/**
 \brief Boundary condition kernel for execution on the GPU
*/
void boundary_kernel(fp_t** conc, const int nx, const int ny, const int nm);

/**
 \brief Tiled convolution algorithm for execution on the GPU
*/
void convolution_kernel(fp_t** conc_old, fp_t** conc_lap, fp_t** mask_lap,
                        const int nx, const int ny, const int nm);

/**
 \brief Vector addition algorithm for execution on the GPU
*/
void diffusion_kernel(fp_t** conc_old, fp_t** conc_new, fp_t** conc_lap,
                      const int nx, const int ny, const int nm, const fp_t D, const fp_t dt);

/** \cond SuppressGuard */
#endif /* _OPENACC_KERNELS_H_ */
/** \endcond */
