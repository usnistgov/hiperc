/**********************************************************************************
 HiPerC: High Performance Computing Strategies for Boundary Value Problems
 Written by Trevor Keller and available from https://github.com/usnistgov/hiperc
 **********************************************************************************/

/**
 \file  mesh.c
 \brief Implemenatation of mesh handling functions for diffusion benchmarks
*/

#include <stdio.h>
#include <stdlib.h>
#include "mesh.h"

void make_arrays(fp_t*** conc_old, fp_t*** conc_new,
				 fp_t*** conc_lap, fp_t*** conc_div,
				 fp_t*** mask_lap,
                 const int nx, const int ny, const int nm)
{
	int i;

	/* create 2D pointers */
	*conc_old = (fp_t **)calloc(nx, sizeof(fp_t *));
	*conc_new = (fp_t **)calloc(nx, sizeof(fp_t *));
	*conc_lap = (fp_t **)calloc(nx, sizeof(fp_t *));
	*conc_div = (fp_t **)calloc(nx, sizeof(fp_t *));
	*mask_lap = (fp_t **)calloc(nm, sizeof(fp_t *));

	/* allocate 1D data */
	(*conc_old)[0] = (fp_t *)calloc(nx * ny, sizeof(fp_t));
	(*conc_new)[0] = (fp_t *)calloc(nx * ny, sizeof(fp_t));
	(*conc_lap)[0] = (fp_t *)calloc(nx * ny, sizeof(fp_t));
	(*conc_div)[0] = (fp_t *)calloc(nx * ny, sizeof(fp_t));
	(*mask_lap)[0] = (fp_t *)calloc(nm * nm, sizeof(fp_t));

	/* map 2D pointers onto 1D data */
	for (i = 1; i < ny; i++) {
		(*conc_old)[i] = &(*conc_old[0])[nx * i];
		(*conc_new)[i] = &(*conc_new[0])[nx * i];
		(*conc_lap)[i] = &(*conc_lap[0])[nx * i];
		(*conc_div)[i] = &(*conc_div[0])[nx * i];
	}

	for (i = 1; i < nm; i++) {
		(*mask_lap)[i] = &(*mask_lap[0])[nm * i];
	}
}

void free_arrays(fp_t** conc_old, fp_t** conc_new,
				 fp_t** conc_lap, fp_t** conc_div,
				 fp_t** mask_lap)
{
	free(conc_old[0]);
	free(conc_old);

	free(conc_new[0]);
	free(conc_new);

	free(conc_lap[0]);
	free(conc_lap);

	free(conc_div[0]);
	free(conc_div);

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

void swap_pointers_1D(fp_t** conc_old, fp_t** conc_new)
{
	fp_t* temp;

	temp = (*conc_old);
	(*conc_old) = (*conc_new);
	(*conc_new) = temp;
}
