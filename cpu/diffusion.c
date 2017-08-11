/*
	File: diffusion.c
	Role: implementation of semi-infinite diffusion equation

	Questions/comments to trevor.keller@nist.gov
	Bugs/requests to https://github.com/tkphd/accelerator-testing
*/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "diffusion.h"

int main(int argc, char* argv[])
{
	/* declare file handles */
	FILE * input, * error;

	/* declare mesh size */
	int nx, ny;

	/* declare mesh resolution */
	double dx, dy, h;

	/* declare mesh pointers */
	double **oldMesh, **newMesh, **conMesh, **mask;
	double *oldData, *newData, *conData, *maskData;
	double bc[2][2];

	/* declare time variables */
	double dt, elapsed;
	int step, steps, checks;

	/* declare materials and numerical parameters */
	double c0, D, linStab, sse;

	/* check for proper invocation */
	if (argc != 2) {
		printf("Error: improper arguments supplied.\nUsage: ./%s filename\n", argv[0]);
		exit(-1);
	}

	/* Read grid size and mesh resolution from file */
	input = fopen(argv[1], "r");
	if (input == NULL) {
		printf("Error: unable to open parameter file %s. Check permissions.\n", argv[1]);
		exit(-1);
	}

	/* read parameters */
	fscanf(input, "%i %i %lf %lf %i %i %lf", &nx, &ny, &dx, &dy, &steps, &checks, &D);
	#ifndef NDEBUG
	printf("Constructing %ix%i grid with resolution %fx%f. Taking %i steps with checkpoints every %i.\n", nx, ny, dx, dy, steps, checks);
	#endif
	fclose(input);

	elapsed = 0.0;
	h = (dx>dy) ? dy : dx;
	dt = (h * h) / (64.0 * D);
	linStab = (h * h) / (4.0 * dt * D);

	/* report runtime parameters */
	printf("Evolving %i steps with dt=%f. Using D=%f, linear stability is %f\n", steps, dt, D, linStab);

	/* initialize memory */
	make_arrays(&oldMesh, &newMesh, &conMesh, &mask, &oldData, &newData, &conData, &maskData, nx, ny);
	set_mask(mask);
	set_boundaries(&c0, bc);
	apply_initial_conditions(oldMesh, newMesh, conMesh, nx, ny, c0, bc);

	/* prepare to log errrors */
	error = fopen("error.csv", "w");

	/* do the work */
	for (step = 0; step < steps; step++) {
		apply_boundary_conditions(oldMesh, newMesh, nx, ny, bc);

		compute_convolution(oldMesh, newMesh, conMesh, mask, nx, ny, dx, dy);

		step_in_time(oldMesh, newMesh, conMesh, nx, ny, dt, &elapsed);

		check_solution(newMesh, nx, ny, dx, dy, elapsed, bc, &sse);
		fprintf(error, "%f,%f\n", elapsed, sse);

		write_csv(newMesh, nx, ny, dx, dy, step);

		swap_pointers(&oldData, &newData, &oldMesh, &newMesh);
	}

	/* clean up */
	fclose(error);

	free_arrays(oldMesh, newMesh, conMesh, mask, oldData, newData, conData, maskData, nx, ny);

	return 0;
}
