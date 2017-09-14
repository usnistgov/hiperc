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
 \brief OpenCL version 1.0 does not support the 'static' storage class specifier
*/
#pragma OPENCL EXTENSION cl_clang_storage_class_specifiers : enable

#include "opencl_kernels.h"

/**
 \brief Diffusion equation kernel for execution on the GPU

 This function accesses 1D data rather than the 2D array representation of the
 scalar composition field
*/
__kernel void diffusion_kernel(__global fp_t* d_conc_old,
                               __global fp_t* d_conc_new,
                               __global fp_t* d_conc_lap,
                               int nx,
                               int ny,
                               int nm,
                               fp_t D,
                               fp_t dt)
{
	int col, idx, row;

	/* determine indices on which to operate */
	col = get_global_id(0);
	row = get_global_id(1);
	idx = row * nx + col;

	/* explicit Euler solution to the equation of motion */
	if (row < ny && col < nx) {
		d_conc_new[idx] = d_conc_old[idx] + dt * D * d_conc_lap[idx];
	}

	/* wait for all threads to finish writing */
	barrier(CLK_GLOBAL_MEM_FENCE);
}
