/**********************************************************************************
 HIPERC: High Performance Computing Strategies for Boundary Value Problems
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
 \file  mesh.c
 \brief Implemenatation of mesh handling functions for diffusion benchmarks
*/

#include <stdio.h>
#include <stdlib.h>
#include "mesh.h"

void make_arrays(fp_t*** conc_old, fp_t*** conc_new, fp_t*** conc_lap, fp_t*** mask_lap,
                 int nx, int ny, int nm)
{
	int j;

	/* create 2D pointers */
	*conc_old = (fp_t **)calloc(nx, sizeof(fp_t *));
	*conc_new = (fp_t **)calloc(nx, sizeof(fp_t *));
	*conc_lap = (fp_t **)calloc(nx, sizeof(fp_t *));
	*mask_lap = (fp_t **)calloc(nm, sizeof(fp_t *));

	/* allocate 1D data */
	(*conc_old)[0] = (fp_t *)calloc(nx * ny, sizeof(fp_t));
	(*conc_new)[0] = (fp_t *)calloc(nx * ny, sizeof(fp_t));
	(*conc_lap)[0] = (fp_t *)calloc(nx * ny, sizeof(fp_t));
	(*mask_lap)[0] = (fp_t *)calloc(nm * nm, sizeof(fp_t));

	/* map 2D pointers onto 1D data */
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

void swap_pointers_1D(fp_t** conc_old, fp_t** conc_new)
{
	fp_t* temp;

	temp = (*conc_old);
	(*conc_old) = (*conc_new);
	(*conc_new) = temp;
}
