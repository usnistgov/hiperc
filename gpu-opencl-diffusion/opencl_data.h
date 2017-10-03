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
 \brief Container for GPU array pointers and parameters

 From the <a href="https://www.khronos.org/registry/OpenCL/specs/opencl-1.2.pdf">OpenCL v1.2</a> spec:
 - A \a Context is the environment within which the kernels execute and the domain in which
   synchronization and memory management is defined. The context includes a set of devices, the
   memory accessible to those devices, the corresponding memory properties and one or more
   command-queues used to schedule execution of a kernel(s) or operations on memory objects.
 - A \a Program \a Object encapsulates the following information:
   - A reference to an associated context.
   - A program source or binary.
   - The latest successfully built program executable, the list of devices for which the program
     executable is built, the build options used and a build log.
   - The number of kernel objects currently attached.
 - A \a Kernel \a Object encapsulates a specific \c __kernel function declared in a
   program and the argument values to be used when executing this \c __kernel function.
*/
struct OpenCLData {
	/** OpenCL interface to the GPU, hardware and software */
	cl_context context;

	/** Copy of old composition field on the GPU */
	cl_mem conc_old;
	/** Copy of new composition field on the GPU */
	cl_mem conc_new;
	/** Copy of Laplacian field on the GPU */
	cl_mem conc_lap;

	/** Copy of Laplacian mask on the GPU */
	cl_mem mask;
	/** Copy of boundary values on the GPU */
	cl_mem bc;

	/** Boundary program source for JIT compilation on the GPU */
	cl_program boundary_program;
	/** Convolution program source for JIT compilation on the GPU */
	cl_program convolution_program;
	/** Timestepping program source for JIT compilation on the GPU */
	cl_program diffusion_program;

	/** Boundary program executable for the GPU */
	cl_kernel boundary_kernel;
	/** Convolution program executable for the GPU */
	cl_kernel convolution_kernel;
	/** Timestepping program executable for the GPU */
	cl_kernel diffusion_kernel;

	/** Queue for submitting OpenCL jobs to the GPU */
	cl_command_queue commandQueue;
};

/**
 \brief Report error code when status is not \c CL_SUCCESS

 Refer to https://streamhpc.com/blog/2013-04-28/opencl-error-codes/
 for help interpreting error codes.
*/
void report_error(cl_int error, const char* message);

/**
 \brief Build kernel program from text input

 Source follows the OpenCL Programming Book,
 https://www.fixstars.com/en/opencl/book/OpenCLProgrammingBook/calling-the-kernel/
*/
void build_program(const char* filename,
                  cl_context* context,
                  cl_device_id* gpu,
                  cl_program* program,
                  cl_int* status);

/**
 \brief Initialize OpenCL device memory before marching
*/
void init_opencl(fp_t** conc_old, fp_t** mask_lap, fp_t bc[2][2],
                 int nx, int ny, int nm, struct OpenCLData* dev);

/**
 \brief Specialization of solve_diffusion_equation() using OpenCL
*/
void opencl_diffusion_solver(struct OpenCLData* dev, fp_t** conc_new,
                             int nx, int ny, int nm,
                             fp_t D, fp_t dt, int checks,
                             fp_t *elapsed, struct Stopwatch* sw);

/**
 \brief Free OpenCL device memory after marching
*/
void free_opencl(struct OpenCLData* dev);

/** \cond SuppressGuard */
#endif /* _OPENCL_DATA_H_ */
/** \endcond */
