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
 \file  cuda_kernels.cuh
 \brief Declaration of functions to execute on the GPU (CUDA kernels)
*/

/** \cond SuppressGuard */
#ifndef _CUDA_KERNELS_H_
#define _CUDA_KERNELS_H_
/** \endcond */

extern "C" {
#include "numerics.h"
}

/**
 \brief Convolution mask array on the GPU, allocated in protected memory
*/
__constant__ extern fp_t d_mask[MAX_MASK_W * MAX_MASK_H];

/**
 \brief Boundary condition kernel for execution on the GPU

 This function accesses 1D data rather than the 2D array representation of the
 scalar composition field
*/
__global__ void boundary_kernel(fp_t* conc,
                                const int nx, const int ny, const int nm);

/**
 \brief Tiled convolution algorithm for execution on the GPU

 This function accesses 1D data rather than the 2D array representation of the
 scalar composition field, mapping into 2D tiles on the GPU with halo cells
 before computing the convolution.

 Note:
 - The source matrix (\a conc_old) and destination matrix (\a conc_lap) must be identical in size
 - One CUDA core operates on one array index: there is no nested loop over matrix elements
 - The halo (\a nm/2 perimeter cells) in \a conc_lap are unallocated garbage
 - The same cells in \a conc_old are boundary values, and contribute to the convolution
 - \a conc_tile is the shared tile of input data, accessible by all threads in this block
*/
__global__ void convolution_kernel(fp_t* conc_old,
                                   fp_t* conc_lap,
                                   const int nx,
                                   const int ny,
                                   const int nm);

/**
 \brief Vector addition algorithm for execution on the GPU

 This function accesses 1D data rather than the 2D array representation of the
 scalar composition field. Memory allocation, data transfer, and array release
 are handled in cuda_init(), with arrays on the host and device managed through
 CudaData, which is a struct passed by reference into the function. In this way,
 device kernels can be called in isolation without incurring the cost of data
 transfers and with reduced risk of memory leaks.
*/
__global__ void diffusion_kernel(fp_t* conc_old,
                                 fp_t* conc_new,
                                 fp_t* conc_lap,
                                 const int nx,
                                 const int ny,
                                 const int nm,
                                 const fp_t D,
                                 const fp_t dt);

/** \cond SuppressGuard */
#endif /* _CUDA_KERNELS_H_ */
/** \endcond */
