/*
	File: mesh.c
	Role: implemenatation of mesh handling functions

	Questions/comments to trevor.keller@nist.gov
	Bugs/requests to https://github.com/tkphd/accelerator-testing
*/

#include <stdio.h>
#include <stdlib.h>

#include "diffusion.h"

void make_arrays(double*** A, double*** B, double*** C, double*** M, double** dataA, double** dataB, double** dataC, double** dataM, int nx, int ny)
{
	int j;

	if ((nx * sizeof(double)) % 32 != 0) {
		printf("Warning: domain width of %i (%lu B) produces misaligned DRAM reads. Consider a multiple of 32.\n", nx, nx * sizeof(double));
	}

	/* allocate 1D data arrays */
	(*dataA) = (double *)calloc(nx * ny, sizeof(double));
	(*dataB) = (double *)calloc(nx * ny, sizeof(double));
	(*dataC) = (double *)calloc(nx * ny, sizeof(double));
	(*dataM) = (double *)calloc(3 * 3, sizeof(double));

	/* map 2D arrays onto 1D data */
	(*A) = (double **)calloc(nx, sizeof(double *));
	(*B) = (double **)calloc(nx, sizeof(double *));
	(*C) = (double **)calloc(nx, sizeof(double *));
	(*M) = (double **)calloc(3, sizeof(double *));

	for (j = 0; j < ny; j++) {
		(*A)[j] = &((*dataA)[nx * j]);
		(*B)[j] = &((*dataB)[nx * j]);
		(*C)[j] = &((*dataC)[nx * j]);
	}

	for (j = 0; j < 3; j++) {
		(*M)[j] = &((*dataM)[3 * j]);
	}
}

void free_arrays(double** A, double** B, double** C, double** M, double* dataA, double* dataB, double* dataC, double* dataM)
{
	free(A);
	free(B);
	free(C);
	free(M);

	free(dataA);
	free(dataB);
	free(dataC);
	free(dataM);
}

void swap_pointers(double** dataA, double** dataB, double*** A, double*** B)
{
	double* dataC;
	double** C;

	#ifdef DEBUG
	printf("Ai=%li, Bi=%li", (*dataA), (*dataB));
	#endif

	dataC = (*dataA);
	(*dataA) = (*dataB);
	(*dataB) = dataC;

	C = (*A);
	(*A) = (*B);
	(*B) = C;

	#ifdef DEBUG
	printf(", Af=%li, Bf=%li\n", (*dataA), (*dataB));
	#endif

}
