/*
	File: mesh.c
	Role: implemenatation of mesh handling functions

	Questions/comments to trevor.keller@nist.gov
	Bugs/requests to https://github.com/usnistgov/phasefield-accelerator-benchmarks
*/

#include <stdio.h>
#include <stdlib.h>

#include "diffusion.h"

void make_arrays(fp_t*** conc_old, fp_t*** conc_new, fp_t*** conc_lap, fp_t*** mask_lap,
                 int nx, int ny, int nm)
{
	int j;

	if ((nx * sizeof(fp_t)) % 32 != 0) {
		printf("Warning: domain width of %i (%lu B) produces misaligned DRAM reads. Consider a multiple of 32.\n", nx, nx * sizeof(fp_t));
	}

	/* create 2D pointers */
	*conc_old = (fp_t **)calloc(nx, sizeof(fp_t *));
	*conc_new = (fp_t **)calloc(nx, sizeof(fp_t *));
	*conc_lap = (fp_t **)calloc(nx, sizeof(fp_t *));
	*mask_lap = (fp_t **)calloc(nm, sizeof(fp_t *));

	/* allocate 1D data arrays */
	(*conc_old)[0] = (fp_t *)calloc(nx * ny, sizeof(fp_t));
	(*conc_new)[0] = (fp_t *)calloc(nx * ny, sizeof(fp_t));
	(*conc_lap)[0] = (fp_t *)calloc(nx * ny, sizeof(fp_t));
	(*mask_lap)[0] = (fp_t *)calloc(nm * nm, sizeof(fp_t));

	/* map 2D pointers onto 1D arrays */
	for (j = 1; j < ny; j++) {
		(*conc_old)[j] = &(*conc_old[0])[nx * j];
		(*conc_new)[j] = &(*conc_new[0])[nx * j];
		(*conc_lap)[j] = &(*conc_lap[0])[nx * j];
	}

	for (j = 1; j < nm; j++) {
		(*mask_lap)[j] = &(*mask_lap[0])[nm * j];
	}
}

void free_arrays(fp_t** conc_old, fp_t** conc_new, fp_t** conc_lap, fp_t** mask_lap)
{
	free(conc_old[0]);
	free(conc_old);

	free(conc_new[0]);
	free(conc_new);

	free(conc_lap[0]);
	free(conc_lap);

	free(mask_lap[0]);
	free(mask_lap);
}

void swap_pointers(fp_t*** conc_old, fp_t*** conc_new)
{
	fp_t** temp;

	temp = (*conc_old);
	(*conc_old) = (*conc_new);
	(*conc_new) = temp;
}
