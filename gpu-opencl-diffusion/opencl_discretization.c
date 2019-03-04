/**********************************************************************************
 HiPerC: High Performance Computing Strategies for Boundary Value Problems
 Written by Trevor Keller and available from https://github.com/usnistgov/hiperc
 **********************************************************************************/

/**
 \file  opencl_discretization.c
 \brief Implementation of boundary condition functions with OpenCL acceleration
*/

#include <stdio.h>
#include <math.h>
#include <omp.h>
#include "boundaries.h"
#include "mesh.h"
#include "numerics.h"
#include "timer.h"
#include "opencl_data.h"

/* Per <a href="https://software.intel.com/sites/landingpage/opencl/optimization-guide/Work-Group_Size_Considerations.htm">
    Intel's OpenCL advice</a>, the ideal block size \f$ (bx \times by)\f$ is
    within the range from 64 to 128 mesh points. The block size must be an even
    power of two: 4, 8, 16, 32, etc. OpenCL will make a best-guess optimal
    block size if you set size_t* tile_dim = NULL.
*/

void device_boundaries(struct OpenCLData* dev, const int flip,
                       const int nx, const int ny, const int nm,
                       const int bx, const int by)
{
	/* OpenCL uses cl_mem, not fp_t*, so swap_pointers won't work.
	   We leave the pointers alone but call the kernel on the appropriate data location.
	 */
	const cl_mem d_conc_old = (flip == 0) ? dev->conc_old : dev->conc_new;

	const size_t tile_dim[2] = {bx, by};
	const size_t bloc_dim[2] = {ceil((float)(nx) / (tile_dim[0] - nm + 1)),
	                            ceil((float)(ny) / (tile_dim[1] - nm + 1))
	                           };
	const size_t grid_dim[2] = {bloc_dim[0]* tile_dim[0],
	                            bloc_dim[1]* tile_dim[1]
	                           };

	cl_int status = clSetKernelArg(dev->boundary_kernel, 0, sizeof(cl_mem), (void*)&d_conc_old);
	status |= clSetKernelArg(dev->boundary_kernel, 1, sizeof(int), (void*)&nx);
	status |= clSetKernelArg(dev->boundary_kernel, 2, sizeof(int), (void*)&ny);
	status |= clSetKernelArg(dev->boundary_kernel, 3, sizeof(int), (void*)&nm);
	report_error(status, "setting boundary kernel args");

	status |= clEnqueueNDRangeKernel(dev->commandQueue, dev->boundary_kernel, 2, NULL, grid_dim, tile_dim, 0, NULL, NULL);
	report_error(status, "enqueuing boundary kernel");
}

void device_convolution(struct OpenCLData* dev, const int flip,
                        const int nx, const int ny, const int nm,
                        const int bx, const int by)
{
	const cl_mem d_conc_old = (flip == 0) ? dev->conc_old : dev->conc_new;

	const size_t tile_dim[2] = {bx, by};
	const size_t bloc_dim[2] = {ceil((float)(nx) / (tile_dim[0] - nm + 1)),
	                            ceil((float)(ny) / (tile_dim[1] - nm + 1))
	                           };
	const size_t grid_dim[2] = {bloc_dim[0]* tile_dim[0],
	                            bloc_dim[1]* tile_dim[1]
	                           };
	const size_t buf_size = (tile_dim[0] + nm) * (tile_dim[1] + nm) * sizeof(fp_t);
	cl_int status = CL_SUCCESS;

	status |= clSetKernelArg(dev->convolution_kernel, 0, sizeof(cl_mem), (void*)&d_conc_old);
	status |= clSetKernelArg(dev->convolution_kernel, 1, sizeof(cl_mem), (void*)&(dev->conc_lap));
	status |= clSetKernelArg(dev->convolution_kernel, 2, sizeof(cl_mem), (void*)&(dev->mask));
	status |= clSetKernelArg(dev->convolution_kernel, 3, buf_size, NULL);
	status |= clSetKernelArg(dev->convolution_kernel, 4, sizeof(int), (void*)&nx);
	status |= clSetKernelArg(dev->convolution_kernel, 5, sizeof(int), (void*)&ny);
	status |= clSetKernelArg(dev->convolution_kernel, 6, sizeof(int), (void*)&nm);
	report_error(status, "setting convolution kernel args");

	status |= clEnqueueNDRangeKernel(dev->commandQueue, dev->convolution_kernel, 2, NULL, grid_dim, tile_dim, 0, NULL, NULL);
	report_error(status, "enqueuing convolution kernel");
}

void device_diffusion(struct OpenCLData* dev, const int flip,
                      const int nx, const int ny, const int nm,
                      const int bx, const int by,
                      const fp_t D, const fp_t dt)
{
	const cl_mem d_conc_old = (flip == 0) ? dev->conc_old : dev->conc_new;
	const cl_mem d_conc_new = (flip == 0) ? dev->conc_new : dev->conc_old;

	const size_t tile_dim[2] = {bx, by};
	const size_t bloc_dim[2] = {ceil((float)(nx) / (tile_dim[0] - nm + 1)),
	                            ceil((float)(ny) / (tile_dim[1] - nm + 1))
	                           };
	const size_t grid_dim[2] = {bloc_dim[0]* tile_dim[0],
	                            bloc_dim[1]* tile_dim[1]
	                           };
	cl_int status = CL_SUCCESS;

	status |= clSetKernelArg(dev->diffusion_kernel, 0, sizeof(cl_mem), (void*)&d_conc_old);
	status |= clSetKernelArg(dev->diffusion_kernel, 1, sizeof(cl_mem), (void*)&d_conc_new);
	status |= clSetKernelArg(dev->diffusion_kernel, 2, sizeof(cl_mem), (void*)&(dev->conc_lap));
	status |= clSetKernelArg(dev->diffusion_kernel, 3, sizeof(int), (void*)&nx);
	status |= clSetKernelArg(dev->diffusion_kernel, 4, sizeof(int), (void*)&ny);
	status |= clSetKernelArg(dev->diffusion_kernel, 5, sizeof(int), (void*)&nm);
	status |= clSetKernelArg(dev->diffusion_kernel, 6, sizeof(fp_t), (void*)&D);
	status |= clSetKernelArg(dev->diffusion_kernel, 7, sizeof(fp_t), (void*)&dt);
	report_error(status, "setting diffusion kernel args");

	status |= clEnqueueNDRangeKernel(dev->commandQueue, dev->diffusion_kernel, 2, NULL, grid_dim, tile_dim, 0, NULL, NULL);
	report_error(status, "enqueuing diffusion kernel");
}

void read_out_result(struct OpenCLData* dev, const int flip, fp_t** conc,
                     const int nx, const int ny)
{
	const cl_mem d_conc = (flip == 0) ? dev->conc_new : dev->conc_old;
	const int grid_size = nx * ny * sizeof(fp_t);

	cl_int status = clEnqueueReadBuffer(dev->commandQueue, d_conc, CL_TRUE, 0, grid_size, conc[0], 0, NULL, NULL);
	report_error(status, "retrieving result from GPU");
}
