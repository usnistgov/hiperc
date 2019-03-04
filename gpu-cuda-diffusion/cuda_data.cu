/**********************************************************************************
 HiPerC: High Performance Computing Strategies for Boundary Value Problems
 Written by Trevor Keller and available from https://github.com/usnistgov/hiperc
 **********************************************************************************/

/**
 \file  cuda_data.cu
 \brief Implementation of functions to create and destroy CudaData struct
*/

extern "C" {
#include "cuda_data.h"
}

#include "cuda_kernels.cuh"

void init_cuda(fp_t** conc_old, fp_t** mask_lap,
               const int nx, const int ny, const int nm, struct CudaData* dev)
{
	/* allocate memory on device */
	cudaMalloc((void**) &(dev->conc_old), nx * ny * sizeof(fp_t));
	cudaMalloc((void**) &(dev->conc_lap), nx * ny * sizeof(fp_t));
	cudaMalloc((void**) &(dev->conc_new), nx * ny * sizeof(fp_t));

	/* transfer mask and boundary conditions to protected memory on GPU */
	cudaMemcpyToSymbol(d_mask, mask_lap[0], nm * nm * sizeof(fp_t));

	/* transfer data from host in to GPU */
	cudaMemcpy(dev->conc_old, conc_old[0], nx * ny * sizeof(fp_t),
	           cudaMemcpyHostToDevice);
}

void free_cuda(struct CudaData* dev)
{
	/* free memory on device */
	cudaFree(dev->conc_old);
	cudaFree(dev->conc_lap);
	cudaFree(dev->conc_new);
}
