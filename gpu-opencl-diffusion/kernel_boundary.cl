/**********************************************************************************
 HiPerC: High Performance Computing Strategies for Boundary Value Problems
 Written by Trevor Keller and available from https://github.com/usnistgov/hiperc
 **********************************************************************************/

/**
 \brief Enable double-precision floats
*/
#if defined(cl_khr_fp64)  // Khronos extension available?
	#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)  // AMD extension available?
	#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif

#include "numerics.h"

/**
 \brief Boundary condition kernel for execution on the GPU
 \fn void boundary_kernel(fp_t* d_conc, const int nx, const int ny, const int nm)

 This function accesses 1D data rather than the 2D array representation of the
 scalar composition field
*/
__kernel void boundary_kernel(__global fp_t* d_conc,
                              const int nx,
                              const int ny,
                              const int nm)
{

	/* determine indices on which to operate */

	const int x = get_global_id(0);
	const int y = get_global_id(1);

	/* apply fixed boundary values: sequence does not matter */

	if (x < 1+nm/2 && y < ny/2) {
		d_conc[nx * y + x] = 1.; /* left value */
	}

	if (x >= nx-1-nm/2 && x < nx && y >= ny/2 && y < ny) {
		d_conc[nx * y + x] = 1.; /* right value */
	}

	/* wait for all threads to finish writing */
	barrier(CLK_GLOBAL_MEM_FENCE);

	/* apply no-flux boundary conditions: inside to out, sequence matters */

	for (int offset = 0; offset < nm/2; offset++) {
		const int ilo = nm/2 - offset;
		const int ihi = nx - 1 - nm/2 + offset;
		const int jlo = nm/2 - offset;
		const int jhi = ny - 1 - nm/2 + offset;

		if (ilo-1 == x && y < ny) {
			d_conc[nx * y + ilo-1] = d_conc[nx * y + ilo]; /* left condition */
		}
		if (ihi+1 == x && y < ny) {
			d_conc[nx * y + ihi+1] = d_conc[nx * y + ihi]; /* right condition */
		}
		if (jlo-1 == y && x < nx) {
			d_conc[nx * (jlo-1) + x] = d_conc[nx * jlo + x]; /* bottom condition */
		}
		if (jhi+1 == y && x < nx) {
			d_conc[nx * (jhi+1) + x] = d_conc[nx * jhi + x]; /* top condition */
		}

		barrier(CLK_GLOBAL_MEM_FENCE);
	}
}
