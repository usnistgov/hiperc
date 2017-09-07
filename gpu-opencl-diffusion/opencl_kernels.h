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
 \file  opencl_kernels.h
 \brief Declaration of functions to execute on the GPU (OpenCL kernels)
*/

/** \cond SuppressGuard */
#ifndef _OPENCL_KERNELS_H_
#define _OPENCL_KERNELS_H_
/** \endcond */

#include <CL/cl.h>

#include "numerics.h"

/**
 \brief Width of an input tile, including halo cells, for GPU memory allocation
*/
#define TILE_W 32

/**
 \brief Height of an input tile, including halo cells, for GPU memory allocation
*/
#define TILE_H 32

/* OpenCL requires initializers for __constant arrays
 \brief Convolution mask array on the GPU, allocated in protected memory
__constant extern fp_t d_mask[MAX_MASK_W * MAX_MASK_H];
*/

/*
 \brief Boundary condition array on the GPU, allocated in protected memory
__constant extern fp_t d_bc[2][2];
*/

/**
 \brief Build kernel program from text input

 Source follows the OpenCL Programming Book,
 https://www.fixstars.com/en/opencl/book/OpenCLProgrammingBook/calling-the-kernel/
*/
void build_program(const char* filename,
                  cl_context context,
                  cl_device_id* gpu,
                  cl_program* program,
                  cl_int* status);

/**
 \brief Boundary condition kernel for execution on the GPU

 This function accesses 1D data rather than the 2D array representation of the
 scalar composition field
*/
void boundary_kernel(fp_t* d_conc, fp_t d_bc[2][2],
                     int nx, int ny, int nm);

/**
 \brief Tiled convolution algorithm for execution on the GPU

 This function accesses 1D data rather than the 2D array representation of the
 scalar composition field, mapping into 2D tiles on the GPU with halo cells
 before computing the convolution.

 Note:
 - The source matrix (\a d_conc_old) and destination matrix (\a d_conc_lap)
   must be identical in size
 - One OpenCL worker operates on one array index: there is no nested loop over
   matrix elements
 - The halo (\a nm/2 perimeter cells) in \a d_conc_lap are unallocated garbage
 - The same cells in \a d_conc_old are boundary values, and contribute to the
   convolution
 - \a d_conc_tile is the shared tile of input data, accessible by all threads
   in this block
 - The \a __local specifier allocates the small \a d_conc_tile array in cache
 - The \a __constant specifier allocates the small \a d_mask array in cache
*/
void convolution_kernel(fp_t* d_conc_old, fp_t* d_conc_lap, fp_t** d_mask,
                        int nx, int ny, int nm);

/**
 \brief Diffusion equation kernel for execution on the GPU

 This function accesses 1D data rather than the 2D array representation of the
 scalar composition field
*/
void diffusion_kernel(fp_t* d_conc_old, fp_t* d_conc_new, fp_t* d_conc_lap,
                      int nx, int ny, int nm,
                      fp_t D, fp_t dt);


/** \cond SuppressGuard */
#endif /* _OPENCL_KERNELS_H_ */
/** \endcond */
