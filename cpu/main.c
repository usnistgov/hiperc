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

	/* declare parameter variables */
	char buffer[256];
	char* pch;
	int ith=0, inx=0, iny=0, idx=0, idy=0, ins=0, inc=0, idc=0, ico=0;

	/* declare mesh and mask sizes */
	int nx=512, ny=512, nm=3, nth=4;

	/* declare mesh resolution */
	double dx=0.5, dy=0.5, h=0.5;

	/* declare mesh parameters */
	double **oldMesh, **newMesh, **conMesh, **mask;
	double *oldData, *newData, *conData, *maskData;
	int step=0, steps=100000, checks=10000;
	double bc[2][2];

	/* declare timers */
	double start_time=0., conv_time=0., step_time=0., file_time=0., soln_time=0.;

	/* declare materials and numerical parameters */
	double D=0.00625, linStab=0.1, dt=1., elapsed=0., rss=0.;

	StartTimer();

	/* check for proper invocation */
	if (argc != 2) {
		printf("Error: improper arguments supplied.\nUsage: ./%s filename\n", argv[0]);
		exit(-1);
	}

	/* Read grid size and mesh resolution from file */
	input = fopen(argv[1], "r");
	if (input == NULL) {
		printf("Warning: unable to open parameter file %s. Marching with default values.\n", argv[1]);
	} else {
		/* read parameters */
		while ( !feof(input))
		{
			/* process key-value pairs line-by-line */
			if (fgets(buffer, 256, input) != NULL)
			{
				/* tokenize the key */
				pch = strtok(buffer, " ");

				if (strcmp(pch, "nt") == 0) {
					/* tokenize the value */
					pch = strtok(NULL, " ");
					/* set the value */
					nth = atof(pch);
					ith = 1;
				} else if (strcmp(pch, "nx") == 0) {
					pch = strtok(NULL, " ");
					nx = atoi(pch);
					inx = 1;
				} else if (strcmp(pch, "ny") == 0) {
					pch = strtok(NULL, " ");
					ny = atoi(pch);
					iny = 1;
				} else if (strcmp(pch, "dx") == 0) {
					pch = strtok(NULL, " ");
					dx = atof(pch);
					idx = 1;
				} else if (strcmp(pch, "dy") == 0) {
					pch = strtok(NULL, " ");
					dy = atof(pch);
					idy = 1;
				} else if (strcmp(pch, "ns") == 0) {
					pch = strtok(NULL, " ");
					steps = atoi(pch);
					ins = 1;
				} else if (strcmp(pch, "nc") == 0) {
					pch = strtok(NULL, " ");
					checks = atoi(pch);
					inc = 1;
				} else if (strcmp(pch, "dc") == 0) {
					pch = strtok(NULL, " ");
					D = atof(pch);
					idc = 1;
				} else if (strcmp(pch, "co") == 0) {
					pch = strtok(NULL, " ");
					linStab = atof(pch);
					ico = 1;
				} else {
					printf("Warning: unknown key %s. Ignoring value.\n", pch);
				}
			}
		}

		/* make sure we got everyone */
		if (! ith) {
			printf("Warning: parameter %s undefined. Using default value, %i.\n", "nt", nth);
		} else if (! inx) {
			printf("Warning: parameter %s undefined. Using default value, %i.\n", "nx", nx);
		} else if (! iny) {
			printf("Warning: parameter %s undefined. Using default value, %i.\n", "ny", ny);
		} else if (! idx) {
			printf("Warning: parameter %s undefined. Using default value, %f.\n", "dx", dx);
		} else if (! idy) {
			printf("Warning: parameter %s undefined. Using default value, %f.\n", "dy", dy);
		} else if (! ins) {
			printf("Warning: parameter %s undefined. Using default value, %i.\n", "ns", steps);
		} else if (! inc) {
			printf("Warning: parameter %s undefined. Using default value, %i.\n", "nc", checks);
		} else if (! idc) {
			printf("Warning: parameter %s undefined. Using default value, %f.\n", "dc", D);
		} else if (! ico) {
			printf("Warning: parameter %s undefined. Using default value, %f.\n", "co", linStab);
		}
	}

	/* set numerical parameters */
	set_threads(nth);
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
