/**********************************************************************************
 HIPERC: High Performance Computing Strategies for Boundary Value Problems
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
 \file  opencl_kernels.h
 \brief Declaration of functions to execute on the GPU (OpenCL kernels)
*/

/** \cond SuppressGuard */
#ifndef _OPENCL_KERNELS_H_
#define _OPENCL_KERNELS_H_
/** \endcond */

#include "numerics.h"

/**
 \brief Width of an input tile, including halo cells, for GPU memory allocation
*/
#define TILE_W 32

/**
 \brief Height of an input tile, including halo cells, for GPU memory allocation
*/
#define TILE_H 32

/** \cond SuppressGuard */
#endif /* _OPENCL_KERNELS_H_ */
/** \endcond */
