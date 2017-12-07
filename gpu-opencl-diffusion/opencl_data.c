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
#include "numerics.h"
#include "opencl_data.h"

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

void init_opencl(fp_t** conc_old, fp_t** mask_lap,
               const int nx, const int ny, const int nm, struct OpenCLData* dev)
{
	/* Here's a lot of source code required to prepare your accelerator to do work. */

	const int gridSize = nx * ny * sizeof(fp_t);
	const int maskSize = nm * nm * sizeof(fp_t);

	cl_int status;
	cl_device_id gpu;
	cl_uint numPlatforms;
	size_t numDevices;
	cl_platform_id* platforms;
	cl_device_id* devices;

	/* Platform: The host plus a collection of devices managed by a specific
	   OpenCL implementation that allow an application to share resources and
	   execute kernels on devices in the platform. There will typically be one
	   platform, e.g. nvidia-opencl or amd-opencl. It is not uncommon to have
	   two or three cards from different manufacturers installed, with one
	   platform per vendor, so "typical" is not "only."
	 */
	status = clGetPlatformIDs(0, NULL, &numPlatforms);
	report_error(status, "clGetPlatformIDs");
	if (numPlatforms == 0) {
		printf("Error: No OpenCL framework found.\n");
		exit(-1);
	}
	platforms = (cl_platform_id*)malloc(numPlatforms);
	status = clGetPlatformIDs(numPlatforms, platforms, NULL);
	report_error(status, "clGetPlatformIDs");

	/* Context: shared memory space, allowing threads to collaborate.
	   Specific to one platform.
	 */
	cl_context_properties properties[] = {CL_CONTEXT_PLATFORM,
	                                      (cl_context_properties)platforms[0],
	                                      0};
	dev->context = clCreateContextFromType(properties, CL_DEVICE_TYPE_ALL, 0, NULL, &status);
	report_error(status, "clCreateContextFromType");

	/* Device: CPU, GPU, FPGA, or other threaded hardware available to do work.
	   Specific to one context. If you have more than one device, the following
	   definition of "gpu" may fail. Contact the developers if you'd like to
	   help us generalize the code to properly handle your setup.
	 */
	status = clGetContextInfo(dev->context, CL_CONTEXT_DEVICES, 0, NULL, &numDevices);
	report_error(status, "clGetContextInfo");
	devices = (cl_device_id*)malloc(numDevices);
	status = clGetContextInfo(dev->context, CL_CONTEXT_DEVICES, numDevices, devices, NULL);
	report_error(status, "clGetContextInfo");
	if (numDevices == 0) {
		printf("Error: No OpenCL-compatible devices found.\n");
		exit(-1);
	}
	gpu = devices[0];

	/* CommandQueue: coordinator of kernels to run in a given context */
	/* OpenCL v1 */
	dev->commandQueue = clCreateCommandQueue(dev->context, gpu, 0, &status);
	report_error(status, "clCreateCommandQueue");
	/* OpenCL v2 */
	/* dev->commandQueue = clCreateCommandQueueWithProperties(dev->context, gpu, (cl_queue_properties*)NULL, &status); */
	/* report_error(status, "clCreateCommandQueue"); */

	/* Program: set of one or more kernels to run in a given context,
	   read from kernel files *.cl
	 */
	build_program("kernel_boundary.cl", &(dev->context), &gpu, &(dev->boundary_program), &status);
	build_program("kernel_convolution.cl", &(dev->context), &gpu, &(dev->convolution_program), &status);
	build_program("kernel_diffusion.cl", &(dev->context), &gpu, &(dev->diffusion_program), &status);

	/* Kernel: code compatible with the just-in-time (JIT) compiler
	   Names (strings) must match the name of a function defined in the
	   cl_program produced by build_program().
	 */
	dev->boundary_kernel = clCreateKernel(dev->boundary_program, "boundary_kernel", &status);
	report_error(status, "dev->boundary_kernel");
	dev->convolution_kernel = clCreateKernel(dev->convolution_program, "convolution_kernel", &status);
	report_error(status, "dev->convolution_kernel");
	dev->diffusion_kernel = clCreateKernel(dev->diffusion_program, "diffusion_kernel", &status);
	report_error(status, "dev->diffusion_kernel");

	/* allocate memory on device */
	dev->conc_old = clCreateBuffer(dev->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, gridSize, conc_old[0], &status);
	report_error(status, "create dev->conc_old");
	dev->conc_new = clCreateBuffer(dev->context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, gridSize, NULL, &status);
	report_error(status, "create dev->conc_new");
	dev->conc_lap = clCreateBuffer(dev->context, CL_MEM_READ_WRITE, gridSize, NULL, &status);
	report_error(status, "create dev->conc_lap");

	dev->mask = clCreateBuffer(dev->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, maskSize, mask_lap[0], &status);
	report_error(status, "create dev->mask");

	/* clean up */
	free(platforms);
	free(devices);
}

void build_program(const char* filename,
                  cl_context* context,
                  cl_device_id* gpu,
                  cl_program* program,
                  cl_int* status)
{
	FILE *fp;
	char* source_str;
	char msg[1024];
	size_t source_len, program_size, read_size;
	char options[] = "-I../common-diffusion";

	fp = fopen(filename, "rb");
	if (!fp) {
	    printf("Failed to load kernel %s\n", filename);
	    exit(-1);
	}

	fseek(fp, 0, SEEK_END);
	program_size = ftell(fp);
	rewind(fp);

	source_str = (char*)malloc(program_size + sizeof(char));
	source_str[program_size] = '\0';

	read_size = fread(source_str, sizeof(char), program_size, fp);
	assert(read_size == program_size);
	fclose(fp);

	strcpy(msg, filename);
	strcat(msg, ": clCreateProgramWithSource");
	source_len = strlen(source_str);
	*program = clCreateProgramWithSource(*context, 1, (const char **)&source_str, &source_len, status);
	report_error(*status, msg);

	strcpy(msg, filename);
	strcat(msg, ": clBuildProgram");
	*status = clBuildProgram(*program, 0, NULL, (const char*)options, NULL, NULL);

	/* report_error is too granular: report specific build errors */
	if(*status != CL_SUCCESS) {
		/* Thanks to https://stackoverflow.com/a/29813956 */
		char *buff_erro;
		cl_int errcode;
		size_t build_log_len;
		errcode = clGetProgramBuildInfo(*program, *gpu, CL_PROGRAM_BUILD_LOG, 0, NULL, &build_log_len);
		if (errcode) {
			printf("clGetProgramBuildInfo failed at line %d\n", __LINE__);
			exit(-1);
		}

		buff_erro = (char*)malloc(build_log_len);
		if (!buff_erro) {
			printf("malloc failed at line %d\n", __LINE__);
			exit(-2);
		}

		errcode = clGetProgramBuildInfo(*program, *gpu, CL_PROGRAM_BUILD_LOG, build_log_len, buff_erro, NULL);
		if (errcode) {
			printf("clGetProgramBuildInfo failed at line %d\n", __LINE__);
			exit(-3);
		}

		fprintf(stderr,"Build log: \n%s\n", buff_erro);
		free(buff_erro);
		fprintf(stderr,"clBuildProgram failed\n");
	}

	report_error(*status, msg);

	/* clean up */
	free(source_str);
}

void free_opencl(struct OpenCLData* dev)
{
	/* clean up */
	free(dev->conc_old);
	free(dev->conc_new);
	free(dev->conc_lap);
	free(dev->mask);

	clReleaseContext(dev->context);

	clReleaseKernel(dev->boundary_kernel);
	clReleaseKernel(dev->convolution_kernel);
	clReleaseKernel(dev->diffusion_kernel);

	clReleaseProgram(dev->boundary_program);
	clReleaseProgram(dev->convolution_program);
	clReleaseProgram(dev->diffusion_program);

	clReleaseCommandQueue(dev->commandQueue);
}
