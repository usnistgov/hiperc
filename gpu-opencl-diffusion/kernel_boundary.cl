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

#include "opencl_kernels.h"

/**
 \brief Boundary condition kernel for execution on the GPU
 \fn void boundary_kernel(fp_t* d_conc, fp_t d_bc[2][2], int nx, int ny, int nm)

 This function accesses 1D data rather than the 2D array representation of the
 scalar composition field
*/
__kernel void boundary_kernel(__global fp_t* d_conc,
                              __global fp_t d_bc[2][2],
                              int nx, int ny, int nm)
{
	int idx, col, row;
	int ihi, ilo, jhi, jlo, offset;

	/* determine indices on which to operate */

	col = get_global_id(0);
	row = get_global_id(1);
	idx = row * nx + col;

	/* apply fixed boundary values: sequence does not matter */

	if (row >= 0 && row < ny/2 && col >= 0 && col < 1+nm/2) {
		d_conc[idx] = d_bc[1][0]; /* left value */
	}

	if (row >= ny/2 && row < ny && col >= nx-1-nm/2 && col < nx) {
		d_conc[idx] = d_bc[1][1]; /* right value */
	}

	/* wait for all threads to finish writing */
	barrier(CLK_GLOBAL_MEM_FENCE);

	/* apply no-flux boundary conditions: inside to out, sequence matters */

	for (offset = 0; offset < nm/2; offset++) {
		ilo = nm/2 - offset;
		ihi = nx - 1 - nm/2 + offset;
		if (col == ilo-1 && row >= 0 && row < ny) {
			d_conc[idx] = d_conc[row * nx + ilo]; /* left condition */
		} else if (col == ihi+1 && row >= 0 && row < ny) {
			d_conc[idx] = d_conc[row * nx + ihi]; /* right condition */
		}
		barrier(CLK_GLOBAL_MEM_FENCE);
	}

	for (offset = 0; offset < nm/2; offset++) {
		jlo = nm/2 - offset;
		jhi = ny - 1 - nm/2 + offset;
		if (row == jlo-1 && col >= 0 && col < nx) {
			d_conc[idx] = d_conc[jlo * nx + col]; /* bottom condition */
		} else if (row == jhi+1 && col >= 0 && col < nx) {
			d_conc[idx] = d_conc[jhi * nx + col]; /* top condition */
		}
		barrier(CLK_GLOBAL_MEM_FENCE);
	}
}
