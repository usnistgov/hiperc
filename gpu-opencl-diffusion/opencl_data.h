/**********************************************************************************
 This file is part of Phase-field Accelerator Benchmarks, written by Trevor Keller
 and available from https://github.com/usnistgov/phasefield-accelerator-benchmarks.

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
 \file  opencl_data.h
 \brief Declaration of OpenCL data container
*/

/** \cond SuppressGuard */
#ifndef _OPENCL_DATA_H_
#define _OPENCL_DATA_H_
/** \endcond */

#include <CL/cl.h>

#include "type.h"

/**
 \brief Greatest number of expected platforms
*/
#define MAX_PLATFORMS 4

/**
 \brief Greatest number of expected devices
*/
#define MAX_DEVICES 32

/**
 \brief Container for GPU array pointers and parameters
*/
struct OpenCLData {
	/* data arrays on GPU */
	cl_mem conc_old;
	cl_mem conc_new;
	cl_mem conc_lap;

	cl_mem mask;
	cl_mem bc;

	/* kernel source code */
	cl_program boundary_program;
	cl_program convolution_program;
	cl_program diffusion_program;

	/* execution kernels */
	cl_kernel boundary_kernel;
	cl_kernel convolution_kernel;
	cl_kernel diffusion_kernel;

	/* OpenCL machinery */
	cl_context context;
	cl_command_queue commandQueue;
};

/**
 \brief Report error code when status is not \c CL_SUCCESS

 Refer to https://streamhpc.com/blog/2013-04-28/opencl-error-codes/
 for help interpreting error codes.
*/
void report_error(cl_int error, const char* message);

/**
 \brief Initialize OpenCL device memory before marching
*/
void init_opencl(fp_t** conc_old, fp_t** mask_lap, fp_t bc[2][2],
                 int nx, int ny, int nm, struct OpenCLData* dev);

/**
 \brief Specialization of solve_diffusion_equation() using OpenCL
*/
void opencl_diffusion_solver(struct OpenCLData* dev, fp_t** conc_new,
                             int nx, int ny, int nm, fp_t bc[2][2],
                             fp_t D, fp_t dt, int checks,
                             fp_t *elapsed, struct Stopwatch* sw);

/**
 \brief Free OpenCL device memory after marching
*/
void free_opencl(struct OpenCLData* dev);

/** \cond SuppressGuard */
#endif /* _OPENCL_DATA_H_ */
/** \endcond */
