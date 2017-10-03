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
 \file  opencl_discretization.c
 \brief Implementation of boundary condition functions with OpenCL acceleration
*/

#include <stdio.h>
#include <math.h>
#include <omp.h>
#include "boundaries.h"
#include "discretization.h"
#include "mesh.h"
#include "timer.h"
#include "opencl_data.h"
#include "opencl_kernels.h"

void opencl_diffusion_solver(struct OpenCLData* dev, fp_t** conc_new,
                             int nx, int ny, int nm,
                             fp_t D, fp_t dt, int checks,
                             fp_t *elapsed, struct Stopwatch* sw)
{
	double start_time;
	int check=0;
	int grid_size = nx * ny * sizeof(fp_t);
	/** Per <a href="https://software.intel.com/sites/landingpage/opencl/optimization-guide/Work-Group_Size_Considerations.htm">
	 Intel's OpenCL advice</a>, the ideal block size \f$ (bx \times by)\f$ is
	 within the range from 64 to 128 mesh points. The block size must be an even
	 power of two: 4, 8, 16, 32, etc. OpenCL will make a best-guess optimal
	 block size if you set size_t* tile_dim = NULL.
	*/
	/*
	size_t grid_dim[2] = {(size_t)nx, (size_t)ny};
	size_t tile_dim[2] = {TILE_W, TILE_H};
	*/

	/* extremely precarious */
	size_t tile_dim[2] = {TILE_W - nm + 1, TILE_H - nm + 1};
	size_t bloc_dim[2] = {nm - 1 + floor((float)(nx)/tile_dim[0]), nm - 1 + floor((float)(ny)/tile_dim[1])};
	size_t grid_dim[2] = {bloc_dim[0]*tile_dim[0], bloc_dim[1]*tile_dim[1]};

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
	status |= clSetKernelArg(dev->convolution_kernel, 3, sizeof(int), (void *)&nx);
	status |= clSetKernelArg(dev->convolution_kernel, 4, sizeof(int), (void *)&ny);
	status |= clSetKernelArg(dev->convolution_kernel, 5, sizeof(int), (void *)&nm);
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
		status = clSetKernelArg(dev->boundary_kernel,    0, sizeof(cl_mem), (void *)&d_conc_old);
		report_error(status, "mutable boundary kernel args");

		status = clSetKernelArg(dev->convolution_kernel, 0, sizeof(cl_mem), (void *)&d_conc_old);
		report_error(status, "mutable convolution kernel args");

		status |= clSetKernelArg(dev->diffusion_kernel,   0, sizeof(cl_mem), (void *)&d_conc_old);
		status |= clSetKernelArg(dev->diffusion_kernel,   1, sizeof(cl_mem), (void *)&d_conc_new);
		report_error(status, "mutable diffusion kernel args");

		/* enqueue kernels */
		status |= clEnqueueNDRangeKernel(dev->commandQueue, dev->boundary_kernel,    2, NULL, grid_dim, tile_dim, 0, NULL, NULL);
		status |= clEnqueueNDRangeKernel(dev->commandQueue, dev->convolution_kernel, 2, NULL, grid_dim, tile_dim, 0, NULL, NULL);
		status |= clEnqueueNDRangeKernel(dev->commandQueue, dev->diffusion_kernel,   2, NULL, grid_dim, tile_dim, 0, NULL, NULL);
		report_error(status, "enqueue kernels");
	}

	*elapsed += dt * checks;

	/* transfer from device out to host */
	start_time = GetTimer();
	status = clEnqueueReadBuffer(dev->commandQueue, d_conc_new, CL_TRUE, 0, grid_size, conc_new[0], 0, NULL, NULL);
	report_error(status, "retrieve result from GPU");
	sw->file += GetTimer() - start_time;
}

void check_solution(fp_t** conc_new, fp_t** conc_lap, int nx, int ny,
                    fp_t dx, fp_t dy, int nm, fp_t elapsed, fp_t D,
                    fp_t bc[2][2], fp_t* rss)
{
	fp_t sum=0.;

	#pragma omp parallel reduction(+:sum)
	{
		int i, j;
		fp_t r, cal, car, ca, cn;

		#pragma omp for collapse(2) private(ca,cal,car,cn,i,j,r)
		for (j = nm/2; j < ny-nm/2; j++) {
			for (i = nm/2; i < nx-nm/2; i++) {
				/* numerical solution */
				cn = conc_new[j][i];

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
				ca = cal + car;

				/* residual sum of squares (RSS) */
				conc_lap[j][i] = (ca - cn) * (ca - cn) / (fp_t)((nx-1-nm/2) * (ny-1-nm/2));
			}
		}

		#pragma omp for collapse(2) private(i,j)
		for (j = nm/2; j < ny-nm/2; j++) {
			for (i = nm/2; i < nx-nm/2; i++) {
				sum += conc_lap[j][i];
			}
		}
	}

	*rss = sum;
}
