/*
	File: mesh.c
	Role: implemenatation of mesh handling functions

	Questions/comments to trevor.keller@nist.gov
	Bugs/requests to https://github.com/tkphd/accelerator-testing
*/

#include <stdio.h>
#include <stdlib.h>

#include "diffusion.h"

void make_arrays(double*** A, double*** B, double*** C, double*** M,
                 double** dataA, double** dataB, double** dataC, double** dataM,
                 int nx, int ny, int nm)
{
	int j;

	/* allocate 1D data arrays */
	(*dataA) = (double *)calloc(nx * ny, sizeof(double));
	(*dataB) = (double *)calloc(nx * ny, sizeof(double));
	(*dataC) = (double *)calloc(nx * ny, sizeof(double));
	(*dataM) = (double *)calloc(nm * nm, sizeof(double));

	/* map 2D arrays onto 1D data */
	(*A) = (double **)calloc(nx, sizeof(double *));
	(*B) = (double **)calloc(nx, sizeof(double *));
	(*C) = (double **)calloc(nx, sizeof(double *));
	(*M) = (double **)calloc(nm, sizeof(double *));

	for (j = 0; j < ny; j++) {
		(*A)[j] = &((*dataA)[nx * j]);
		(*B)[j] = &((*dataB)[nx * j]);
		(*C)[j] = &((*dataC)[nx * j]);
	}

	for (j = 0; j < nm; j++) {
		(*M)[j] = &((*dataM)[nm * j]);
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

	dataC = (*dataA);
	(*dataA) = (*dataB);
	(*dataB) = dataC;

	C = (*A);
	(*A) = (*B);
	(*B) = C;
}
