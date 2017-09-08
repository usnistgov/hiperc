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
__kernel void convolution_kernel(__global fp_t* d_conc_old,
                                 __global fp_t* d_conc_lap,
                                 __constant fp_t** d_mask,
                                 int nx, int ny, int nm)
{
	int bx, by, i, j, tx, ty,
	int dst_row, dst_col, dst_tile_w, dst_tile_h;
	int src_row, src_col, src_tile_w, src_tile_h;
	fp_t value=0.;

	/* source tile width includes the halo cells */
	src_tile_w = get_local_size(0);
	src_tile_h = get_local_size(1);

	/* destination tile width excludes the halo cells */
	dst_tile_w = src_tile_w - nm + 1;
	dst_tile_h = src_tile_h - nm + 1;

	/* source block (working group) */
	bx = get_group_id(0);
	by = get_group_id(1);

	/* determine indices on which to operate */
	tx = get_local_id(0);
	ty = get_local_id(1);

	dst_col = bx * dst_tile_w + tx;
	dst_row = by * dst_tile_h + ty;

	src_col = dst_col - nm/2;
	src_row = dst_row - nm/2;

	/* copy tile: __local gives access to all threads working on this tile */
	__local fp_t d_conc_tile[TILE_H + MAX_MASK_H - 1][TILE_W + MAX_MASK_W - 1];

	if ((src_row >= 0) && (src_row < ny) &&
	    (src_col >= 0) && (src_col < nx)) {
		/* if src_row==0, then dst_row==nm/2: this is a halo row */
		d_conc_tile[ty][tx] = d_conc_old[src_row * nx + src_col];
	} else {
		/* points outside the halo should be switched off */
		d_conc_tile[ty][tx] = 0.;
	}

	/* tile data is shared: wait for all threads to finish copying */
	barrier(CLK_LOCAL_MEM_FENCE);

	/* compute the convolution */
	if (tx < dst_tile_w && ty < dst_tile_h) {
		for (j = 0; j < nm; j++) {
			for (i = 0; i < nm; i++) {
				value += d_mask[j * nm + i] * d_conc_tile[j+ty][i+tx];
			}
		}
		/* record value */
		if (dst_row < ny && dst_col < nx) {
			d_conc_lap[dst_row * nx + dst_col] = value;
		}
	}

	/* wait for all threads to finish writing */
	barrier(CLK_GLOBAL_MEM_FENCE);
}