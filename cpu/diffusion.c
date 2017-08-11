/*
	File: diffusion.c
	Role: implementation of semi-infinite diffusion equation

	Questions/comments to trevor.keller@nist.gov
	Bugs/requests to https://github.com/tkphd/accelerator-testing
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "diffusion.h"

int main(int argc, char* argv[])
{
	/* declare file handles */
	FILE * input, * error;

	/* declare mesh size */
	int nx, ny, nm;

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
	printf("Constructing %i x %i grid with resolution %.2e x %.2e. Taking %i steps with checkpoints every %i.\n", nx, ny, dx, dy, steps, checks);
	#endif
	fclose(input);

	elapsed = 0.0;
	h = (dx>dy) ? dy : dx;
	dt = (h * h) / (64.0 * D);
	linStab = (h * h) / (4.0 * dt * D);

	/* report runtime parameters */
	printf("Evolving %i steps with dt=%.2e. Using D=%.2e, linear stability is 1/%.1f\n", steps, dt, D, linStab);

	/* initialize memory */
	make_arrays(&oldMesh, &newMesh, &conMesh, &mask, &oldData, &newData, &conData, &maskData, nx, ny);
	set_mask(dx, dy, &nm, mask);
	set_boundaries(&c0, bc);
	apply_initial_conditions(oldMesh, nx, ny, c0, bc);

	apply_boundary_conditions(oldMesh, nx, ny, bc);

	/* prepare to log errrors */
	error = fopen("error.csv", "w");

	/* write initial condition data */
	write_csv(newMesh, nx, ny, dx, dy, 0);
	write_png(newMesh, nx, ny, dx, dy, 0, bc);

	/* do the work */
	for (step = 1; step < steps+1; step++) {
		compute_convolution(oldMesh, conMesh, mask, nx, ny, nm);

		step_in_time(oldMesh, newMesh, conMesh, nx, ny, D, dt, &elapsed);

		apply_boundary_conditions(newMesh, nx, ny, bc);

		check_solution(newMesh, nx, ny, dx, dy, elapsed, D, bc, &sse);
		fprintf(error, "%f,%f\n", elapsed, sse);


		if (step % checks == 0) {
			write_csv(newMesh, nx, ny, dx, dy, step);
			write_png(newMesh, nx, ny, dx, dy, step, bc);
		}

		swap_pointers(&oldData, &newData, &oldMesh, &newMesh);
	}

	/* clean up */
	fclose(error);
	free_arrays(oldMesh, newMesh, conMesh, mask, oldData, newData, conData, maskData, nx, ny);

	return 0;
}
