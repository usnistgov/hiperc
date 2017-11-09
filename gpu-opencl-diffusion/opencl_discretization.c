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
 \file  opencl_discretization.c
 \brief Implementation of boundary condition functions with OpenCL acceleration
*/

#include <stdio.h>
#include <math.h>
#include <omp.h>
#include "boundaries.h"
#include "discretization.h"
#include "mesh.h"
#include "numerics.h"
#include "timer.h"
#include "opencl_data.h"

void opencl_diffusion_solver(struct OpenCLData* dev, fp_t** conc_new,
                             const int bx, const int by,
                             const int nx, const int ny, const int nm,
                             const fp_t D, const fp_t dt, const int checks,
                             fp_t *elapsed, struct Stopwatch* sw)
{
	double start_time;
	int check=0;
	const int grid_size = nx * ny * sizeof(fp_t);
	/** Per <a href="https://software.intel.com/sites/landingpage/opencl/optimization-guide/Work-Group_Size_Considerations.htm">
	 Intel's OpenCL advice</a>, the ideal block size \f$ (bx \times by)\f$ is
	 within the range from 64 to 128 mesh points. The block size must be an even
	 power of two: 4, 8, 16, 32, etc. OpenCL will make a best-guess optimal
	 block size if you set size_t* tile_dim = NULL.
	*/
	const size_t tile_dim[2] = {bx, by};
	const size_t bloc_dim[2] = {ceil((float)(nx) / (tile_dim[0] - nm + 1)),
	                            ceil((float)(ny) / (tile_dim[1] - nm + 1))};
	const size_t grid_dim[2] = {bloc_dim[0] * tile_dim[0],
	                            bloc_dim[1] * tile_dim[1]};
	const size_t buf_size = (tile_dim[0] + nm) * (tile_dim[1] + nm) * sizeof(fp_t);

	cl_mem d_conc_old = dev->conc_old;
	cl_mem d_conc_new = dev->conc_new;

	cl_int status = CL_SUCCESS;

	/* set immutable kernel arguments */
	status |= clSetKernelArg(dev->boundary_kernel, 1, sizeof(cl_mem), (void *)&(dev->bc));
	status |= clSetKernelArg(dev->boundary_kernel, 2, sizeof(int), (void *)&nx);
	status |= clSetKernelArg(dev->boundary_kernel, 3, sizeof(int), (void *)&ny);
	status |= clSetKernelArg(dev->boundary_kernel, 4, sizeof(int), (void *)&nm);
	report_error(status, "constant boundary kernal args");

	status |= clSetKernelArg(dev->convolution_kernel, 1, sizeof(cl_mem), (void *)&(dev->conc_lap));
	status |= clSetKernelArg(dev->convolution_kernel, 2, sizeof(cl_mem), (void *)&(dev->mask));
	status |= clSetKernelArg(dev->convolution_kernel, 3, buf_size, NULL);
	status |= clSetKernelArg(dev->convolution_kernel, 4, sizeof(int), (void *)&nx);
	status |= clSetKernelArg(dev->convolution_kernel, 5, sizeof(int), (void *)&ny);
	status |= clSetKernelArg(dev->convolution_kernel, 6, sizeof(int), (void *)&nm);
	report_error(status, "constant convolution kernel args");

	status |= clSetKernelArg(dev->diffusion_kernel, 2, sizeof(cl_mem), (void *)&(dev->conc_lap));
	status |= clSetKernelArg(dev->diffusion_kernel, 3, sizeof(int), (void *)&nx);
	status |= clSetKernelArg(dev->diffusion_kernel, 4, sizeof(int), (void *)&ny);
	status |= clSetKernelArg(dev->diffusion_kernel, 5, sizeof(int), (void *)&nm);
	status |= clSetKernelArg(dev->diffusion_kernel, 6, sizeof(fp_t), (void *)&D);
	status |= clSetKernelArg(dev->diffusion_kernel, 7, sizeof(fp_t), (void *)&dt);
	report_error(status, "constant diffusion kernel args");

	/* OpenCL uses cl_mem, not fp_t*, so swap_pointers won't work.
     * We leave the pointers alone but call the kernel on the appropriate data location.
     */
	for (check = 0; check < checks; check++) {
		/* swap pointers on the device */
		if (check % 2 == 0) {
			d_conc_old = dev->conc_old;
			d_conc_new = dev->conc_new;
		} else {
			d_conc_old = dev->conc_new;
			d_conc_new = dev->conc_old;
		}

		/* set time-dependent kernel arguments */
		status = clSetKernelArg(dev->boundary_kernel, 0, sizeof(cl_mem), (void *)&d_conc_old);
		report_error(status, "mutable boundary kernel args");

		status = clSetKernelArg(dev->convolution_kernel, 0, sizeof(cl_mem), (void *)&d_conc_old);
		report_error(status, "mutable convolution kernel args");

		status |= clSetKernelArg(dev->diffusion_kernel, 0, sizeof(cl_mem), (void *)&d_conc_old);
		status |= clSetKernelArg(dev->diffusion_kernel, 1, sizeof(cl_mem), (void *)&d_conc_new);
		report_error(status, "mutable diffusion kernel args");

		/* apply boundary conditions */
		status |= clEnqueueNDRangeKernel(dev->commandQueue, dev->boundary_kernel,    2, NULL, grid_dim, tile_dim, 0, NULL, NULL);

		/* compute Laplacian */
		start_time = GetTimer();
		status |= clEnqueueNDRangeKernel(dev->commandQueue, dev->convolution_kernel, 2, NULL, grid_dim, tile_dim, 0, NULL, NULL);
		sw->conv += GetTimer() - start_time;

		/* compute result */
		start_time = GetTimer();
		status |= clEnqueueNDRangeKernel(dev->commandQueue, dev->diffusion_kernel,   2, NULL, grid_dim, tile_dim, 0, NULL, NULL);
		sw->step += GetTimer() - start_time;

		report_error(status, "enqueue kernels");
	}

	*elapsed += dt * checks;

	/* transfer from device out to host */
	start_time = GetTimer();
	status = clEnqueueReadBuffer(dev->commandQueue, d_conc_new, CL_TRUE, 0, grid_size, conc_new[0], 0, NULL, NULL);
	report_error(status, "retrieve result from GPU");
	sw->file += GetTimer() - start_time;
}

void check_solution(fp_t** conc_new, fp_t** conc_lap, const int nx, const int ny,
                    const fp_t dx, const fp_t dy, const int nm, const fp_t elapsed, const fp_t D,
                    fp_t bc[2][2], fp_t* rss)
{
	fp_t sum=0.;

	#pragma omp parallel reduction(+:sum)
	{
		fp_t r, cal, car;

		#pragma omp for collapse(2) private(cal,car,r)
		for (int j = nm/2; j < ny-nm/2; j++) {
			for (int i = nm/2; i < nx-nm/2; i++) {
				/* numerical solution */
				const fp_t cn = conc_new[j][i];

				/* shortest distance to left-wall source */
				r = distance_point_to_segment(dx * (nm/2), dy * (nm/2),
				                              dx * (nm/2), dy * (ny/2),
				                              dx * i, dy * j);
				analytical_value(r, elapsed, D, bc, &cal);

				/* shortest distance to right-wall source */
				r = distance_point_to_segment(dx * (nx-1-nm/2), dy * (ny/2),
				                              dx * (nx-1-nm/2), dy * (ny-1-nm/2),
				                              dx * i, dy * j);
				analytical_value(r, elapsed, D, bc, &car);

				/* superposition of analytical solutions */
				const fp_t ca = cal + car;

				/* residual sum of squares (RSS) */
				conc_lap[j][i] = (ca - cn) * (ca - cn) / (fp_t)((nx-1-nm/2) * (ny-1-nm/2));
			}
		}

		#pragma omp for collapse(2)
		for (int j = nm/2; j < ny-nm/2; j++) {
			for (int i = nm/2; i < nx-nm/2; i++) {
				sum += conc_lap[j][i];
			}
		}
	}

	*rss = sum;
}
