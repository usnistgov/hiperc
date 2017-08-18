/*
	File: mesh.c
	Role: implemenatation of mesh handling functions

	Questions/comments to trevor.keller@nist.gov
	Bugs/requests to https://github.com/usnistgov/phasefield-accelerator-benchmarks
*/

#include <stdio.h>
#include <stdlib.h>

#include "diffusion.h"

void make_arrays(fp_t*** A, fp_t*** B, fp_t*** C, fp_t*** M,
                 fp_t** dataA, fp_t** dataB, fp_t** dataC, fp_t** dataM,
                 int nx, int ny, int nm)
{
	int j;

	/* allocate 1D data arrays */
	(*dataA) = (fp_t *)calloc(nx * ny, sizeof(fp_t));
	(*dataB) = (fp_t *)calloc(nx * ny, sizeof(fp_t));
	(*dataC) = (fp_t *)calloc(nx * ny, sizeof(fp_t));
	(*dataM) = (fp_t *)calloc(nm * nm, sizeof(fp_t));

	/* map 2D arrays onto 1D data */
	(*A) = (fp_t **)calloc(nx, sizeof(fp_t *));
	(*B) = (fp_t **)calloc(nx, sizeof(fp_t *));
	(*C) = (fp_t **)calloc(nx, sizeof(fp_t *));
	(*M) = (fp_t **)calloc(nm, sizeof(fp_t *));

	for (j = 0; j < ny; j++) {
		(*A)[j] = &((*dataA)[nx * j]);
		(*B)[j] = &((*dataB)[nx * j]);
		(*C)[j] = &((*dataC)[nx * j]);
	}

	for (j = 0; j < nm; j++) {
		(*M)[j] = &((*dataM)[nm * j]);
	}
}

void free_arrays(fp_t** A, fp_t** B, fp_t** C, fp_t** M, fp_t* dataA, fp_t* dataB, fp_t* dataC, fp_t* dataM)
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

void swap_pointers(fp_t** dataA, fp_t** dataB, fp_t*** A, fp_t*** B)
{
	fp_t* dataC;
	fp_t** C;

	dataC = (*dataA);
	(*dataA) = (*dataB);
	(*dataB) = dataC;

	C = (*A);
	(*A) = (*B);
	(*B) = C;
}
