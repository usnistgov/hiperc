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
 \brief Enable double-precision floats
*/
#pragma OPENCL EXTENSION cl_khr_fp64: enable

#include "numerics.h"

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
__kernel void convolution_kernel(__global   fp_t* d_conc_old,
                                 __global   fp_t* d_conc_lap,
                                 __constant fp_t* d_mask,
                                 __local    fp_t* d_conc_tile,
                                 int nx,
                                 int ny,
                                 int nm)
{
	int i, j;
	int dst_nx, dst_ny, dst_x, dst_y;
	int src_nx, src_ny, src_x, src_y;
	int til_nx, til_x, til_y;
	fp_t value = 0.;

	/* source tile includes the halo cells, destination tile does not */
	src_ny = get_local_size(0);
	src_nx = get_local_size(1);
	til_nx = src_nx;

	dst_ny = src_ny - nm + 1;
	dst_nx = src_nx - nm + 1;

	/* determine indices on which to operate */
	til_x = get_local_id(0);
	til_y = get_local_id(1);

	dst_x = get_group_id(0) * dst_ny + til_x;
	dst_y = get_group_id(1) * dst_nx + til_y;

	src_x = dst_x - nm/2;
	src_y = dst_y - nm/2;

	if (src_x >= 0 && src_x < nx &&
	    src_y >= 0 && src_y < ny) {
		d_conc_tile[til_nx * til_y + til_x] = d_conc_old[nx * src_y + src_x];
	}

	/* tile data is shared: wait for all threads to finish copying */
	barrier(CLK_LOCAL_MEM_FENCE);

	/* compute the convolution */
	if (til_x < dst_ny && til_y < dst_nx) {
		for (j = 0; j < nm; j++) {
			for (i = 0; i < nm; i++) {
				value += d_mask[nm * j + i] * d_conc_tile[til_nx * (til_y+j) + til_x+i];
			}
		}
		/* record value */
		if (dst_y < ny && dst_x < nx) {
			d_conc_lap[nx * dst_y + dst_x] = value;
		}
	}

	/* wait for all threads to finish writing */
	barrier(CLK_GLOBAL_MEM_FENCE);
}
