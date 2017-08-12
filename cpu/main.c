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
	FILE * input, * output;

	/* declare mesh and mask sizes */
	int nx, ny, nm;

	/* declare mesh resolution */
	double dx, dy, h;

	/* declare mesh parameters */
	double **oldMesh, **newMesh, **conMesh, **mask;
	double *oldData, *newData, *conData, *maskData;
	int step=0, steps, checks;
	double bc[2][2];

	/* declare timers */
	double start_time, conv_time=0., step_time, file_time, soln_time=0.;

	/* declare materials and numerical parameters */
	double D, linStab=0.1, dt, elapsed=0., rss=0.;

	StartTimer();

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
	fclose(input);

	h = (dx > dy) ? dy : dx;
	dt = (linStab * h * h) / (4.0 * D);

	/* initialize memory */
	make_arrays(&oldMesh, &newMesh, &conMesh, &mask, &oldData, &newData, &conData, &maskData, nx, ny);
	set_mask(dx, dy, &nm, mask);
	set_boundaries(bc);

	start_time = GetTimer();
	apply_initial_conditions(oldMesh, nx, ny, bc);
	step_time = GetTimer() - start_time;

	/* write initial condition data */
	start_time = GetTimer();
	write_csv(oldMesh, nx, ny, dx, dy, 0);
	write_png(oldMesh, nx, ny, 0);
	file_time = GetTimer() - start_time;

	/* prepare to log comparison to analytical solution */
	output = fopen("runlog.csv", "w");
	if (output == NULL) {
		printf("Error: unable to %s for output. Check permissions.\n", "runlog.csv");
		exit(-1);
	}

	fprintf(output, "iter,sim_time,wrss,conv_time,step_time,IO_time,soln_time,run_time\n");
	fprintf(output, "%i,%f,%f,%f,%f,%f,%f,%f\n", step, elapsed, rss, conv_time, step_time, file_time, soln_time, GetTimer());

	/* do the work */
	for (step = 1; step < steps+1; step++) {
		apply_boundary_conditions(oldMesh, nx, ny, bc);

		start_time = GetTimer();
		compute_convolution(oldMesh, conMesh, mask, nx, ny, nm);
		conv_time += GetTimer() - start_time;

		start_time = GetTimer();
		step_in_time(oldMesh, newMesh, conMesh, nx, ny, D, dt, &elapsed);
		step_time += GetTimer() - start_time;

		swap_pointers(&oldData, &newData, &oldMesh, &newMesh);

		if (step % checks == 0) {
			start_time = GetTimer();
			write_csv(oldMesh, nx, ny, dx, dy, step);
			write_png(oldMesh, nx, ny, step);
			file_time += GetTimer() - start_time;
		}

		if (step % 100 == 0) {
			start_time = GetTimer();
			check_solution(oldMesh, nx, ny, dx, dy, elapsed, D, bc, &rss);
			soln_time += GetTimer() - start_time;

			fprintf(output, "%i,%f,%f,%f,%f,%f,%f,%f\n", step, elapsed, rss, conv_time, step_time, file_time, soln_time, GetTimer());
		}
	}

	/* clean up */
	fclose(output);
	free_arrays(oldMesh, newMesh, conMesh, mask, oldData, newData, conData, maskData);

	return 0;
}
