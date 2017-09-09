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
 \file  opencl_data.c
 \brief Implementation of functions to create and destroy OpenCLData struct
*/

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "opencl_data.h"
#include "opencl_kernels.h"

void report_error(cl_int status, const char* message)
{
	if (status != CL_SUCCESS && message !=NULL)
		printf("Failure at %s. ", message);
	if (status < -999) {
		printf("OpenCL extension (driver) error: %i. ", status);
		printf("Refer to https://streamhpc.com/blog/2013-04-28/opencl-error-codes/.\n");
		exit(status);
	} else if (status < -29) {
		printf("OpenCL compilation error: %i. ", status);
		printf("Refer to https://streamhpc.com/blog/2013-04-28/opencl-error-codes/.\n");
		exit(status);
	} else if (status < 0) {
		printf("OpenCL runtime error: %i. ", status);
		printf("Refer to https://streamhpc.com/blog/2013-04-28/opencl-error-codes/.\n");
		exit(status);
	}
}

void init_opencl(fp_t** conc_old, fp_t** mask_lap, fp_t bc[2][2],
               int nx, int ny, int nm, struct OpenCLData* dev)
{
	int gridSize = nx * ny * sizeof(fp_t);
	int maskSize = nm * nm * sizeof(fp_t);
	int bcSize = 2 * 2 * sizeof(fp_t);

	cl_int status;
	cl_device_id gpu;
	cl_uint numPlatforms;
	size_t numDevices;
	cl_platform_id platforms[MAX_PLATFORMS];
	cl_device_id devices[MAX_DEVICES];

	status = clGetPlatformIDs(0, NULL, &numPlatforms);
	if (numPlatforms > MAX_PLATFORMS) {
		printf("Error: increase MAX_PLATFORMS beyond %i\n", numPlatforms);
		exit(-1);
	}

	status = clGetPlatformIDs(numPlatforms, platforms, NULL);
	report_error(status, "clGetPlatformIDs");

	cl_context_properties properties[] = {CL_CONTEXT_PLATFORM, (cl_context_properties)platforms[0], 0};
	dev->context = clCreateContextFromType(properties, CL_DEVICE_TYPE_ALL, 0, NULL, &status);
	report_error(status, "clCreateContextFromType");

	status = clGetContextInfo(dev->context, CL_CONTEXT_DEVICES, 0, NULL, &numDevices);
	report_error(status, "clGetContextInfo");
	status = clGetContextInfo(dev->context, CL_CONTEXT_DEVICES, numDevices, devices, NULL);
	report_error(status, "clGetContextInfo");
	if (numDevices > MAX_DEVICES) {
		printf("Error: increase MAX_DEVICES beyond %zu\n", numDevices);
		exit(-1);
	}

	gpu = devices[0];
	dev->commandQueue = clCreateCommandQueue(dev->context, gpu, 0, &status);
	report_error(status, "clCreateCommandQueue");

	/* allocate memory on device */
	dev->conc_old = clCreateBuffer(dev->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, gridSize, conc_old[0], &status);
	report_error(status, "create dev->conc_old");
	dev->conc_new = clCreateBuffer(dev->context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, gridSize, NULL, &status);
	report_error(status, "create dev->conc_new");
	dev->conc_lap = clCreateBuffer(dev->context, CL_MEM_READ_WRITE, gridSize, NULL, &status);
	report_error(status, "create dev->conc_lap");
	dev->mask = clCreateBuffer(dev->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, maskSize, mask_lap[0], &status);
	report_error(status, "create dev->mask");
	dev->bc = clCreateBuffer(dev->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bcSize, bc[0], &status);
	report_error(status, "create dev->bc");

	/* read programs from kernel files */
	build_program("kernel_boundary.cl", dev->context, gpu, dev->boundary_program, &status);
	report_error(status, "kernel_boundary.cl");
	build_program("kernel_convolution.cl", dev->context, gpu, dev->convolution_program, &status);
	report_error(status, "kernel_convolution.cl");
	build_program("kernel_diffusion.cl", dev->context, gpu, dev->diffusion_program, &status);
	report_error(status, "kernel_diffusion.cl");

	/* prepare kernels compatible with the just-in-time (JIT) compiler */
	dev->boundary_kernel = clCreateKernel(dev->boundary_program, "boundary_kernel", &status);
	report_error(status, "dev->boundary_kernel");
	dev->convolution_kernel = clCreateKernel(dev->convolution_program, "convolution_kernel", &status);
	report_error(status, "dev->convolution_kernel");
	dev->diffusion_kernel = clCreateKernel(dev->diffusion_program, "diffusion_kernel", &status);
	report_error(status, "dev->diffusion_kernel");

	/* That's a lot of source code required to simply prepare your accelerator to do work. */
}

void build_program(const char* filename,
                  cl_context context,
                  cl_device_id gpu,
                  cl_program program,
                  cl_int* status)
{
	FILE *fp;
	char *source_str;
	size_t source_len, program_size, read_size;

	fp = fopen(filename, "rb");
	if (!fp) {
	    printf("Failed to load kernel %s\n", filename);
	    exit(-1);
	}

	fseek(fp, 0, SEEK_END);
	program_size = ftell(fp);
	rewind(fp);

	source_str = (char*)malloc(program_size + 1);
	source_str[program_size] = '\0';

	read_size = fread(source_str, sizeof(char), program_size, fp);
	assert(read_size == program_size);
	fclose(fp);

	source_len = strlen(source_str);
	program = clCreateProgramWithSource(context, 1, (const char **)&source_str, &source_len, status);
	report_error(*status, filename);

	*status = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	report_error(*status, filename);

	free(source_str);
}

void free_opencl(struct OpenCLData* dev)
{
	/* free memory on device */
	clReleaseContext(dev->context);

	clReleaseKernel(dev->boundary_kernel);
	clReleaseKernel(dev->convolution_kernel);
	clReleaseKernel(dev->diffusion_kernel);

	clReleaseProgram(dev->boundary_program);
	clReleaseProgram(dev->convolution_program);
	clReleaseProgram(dev->diffusion_program);

	clReleaseCommandQueue(dev->commandQueue);
}
