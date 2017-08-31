/**********************************************************************************
 This file is part of Phase-field Accelerator Benchmarks, written by Trevor Keller
 and available from https://github.com/usnistgov/phasefield-accelerator-benchmarks.

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

/** \addtogroup DIFFUSION
 \{
*/

/**
 \file  common-diffusion/main.c
 \brief Implementation of semi-infinite diffusion equation
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "type.h"
#include "mesh.h"
#include "numerics.h"
#include "output.h"
#include "timer.h"
#include "boundaries.h"
#include "discretization.h"

/**
 \brief Read input parameters from file specified on the command line
*/
void param_parser(int argc, char* argv[], int* nx, int* ny, int* nm, int* code, fp_t* dx, fp_t* dy, fp_t* D, fp_t* linStab, int* steps, int* checks);

/**
 \brief Run simulation using input parameters specified on the command line

 Program will write a series of PNG image files to visualize scalar composition
 field, plus a final CSV raw data file and CSV runtime log tabulating the
 iteration counter (iter), elapsed simulation time (sim_time), system free
 energy (energy), error relative to analytical solution (wrss), time spent
 performing convolution (conv_time), time spent updating fields (step_time),
 time spent writing to disk (IO_time), time spent generating analytical values
 (soln_time), and total elapsed (run_time).
*/
int main(int argc, char* argv[])
{
	FILE * output;

	/* declare mesh size and resolution */
	fp_t **conc_old, **conc_new, **conc_lap, **mask_lap;
	int nx=512, ny=512, nm=3, code=53;
	fp_t dx=0.5, dy=0.5, h=0.5;
	fp_t bc[2][2];

	/* declare materials and numerical parameters */
	fp_t D=0.00625, linStab=0.1, dt=1., elapsed=0., rss=0.;
	int step=0, steps=100000, checks=10000;
	double start_time=0.;
	struct Stopwatch sw = {0., 0., 0., 0.};

	StartTimer();

	param_parser(argc, argv, &nx, &ny, &nm, &code, &dx, &dy, &D, &linStab, &steps, &checks);

	h = (dx > dy) ? dy : dx;
	dt = (linStab * h * h) / (4.0 * D);

	/* initialize memory */
	make_arrays(&conc_old, &conc_new, &conc_lap, &mask_lap, nx, ny, nm);
	set_mask(dx, dy, code, mask_lap, nm);
	set_boundaries(bc);

	start_time = GetTimer();
	apply_initial_conditions(conc_old, nx, ny, nm, bc);
	sw.step = GetTimer() - start_time;

	/* write initial condition data */
	start_time = GetTimer();
	write_png(conc_old, nx, ny, 0);

	/* prepare to log comparison to analytical solution */
	output = fopen("runlog.csv", "w");
	if (output == NULL) {
		printf("Error: unable to %s for output. Check permissions.\n", "runlog.csv");
		exit(-1);
	}
	sw.file = GetTimer() - start_time;

	fprintf(output, "iter,sim_time,wrss,conv_time,step_time,IO_time,soln_time,run_time\n");
	fprintf(output, "%i,%f,%f,%f,%f,%f,%f,%f\n", step, elapsed, rss, sw.conv, sw.step, sw.file, sw.soln, GetTimer());

	/* do the work */
	for (step = 1; step < steps+1; step++) {
		print_progress(step-1, steps);

		apply_boundary_conditions(conc_old, nx, ny, nm, bc);

		start_time = GetTimer();
		compute_convolution(conc_old, conc_lap, mask_lap, nx, ny, nm);
		sw.conv += GetTimer() - start_time;

		start_time = GetTimer();
		solve_diffusion_equation(conc_old, conc_new, conc_lap, nx, ny, nm, D, dt, &elapsed);
		sw.step += GetTimer() - start_time;

		if (step % checks == 0) {
			start_time = GetTimer();
			write_png(conc_new, nx, ny, step);
			sw.file += GetTimer() - start_time;
		}

		if (step % 100 == 0) {
			start_time = GetTimer();
			check_solution(conc_new, nx, ny, dx, dy, nm, elapsed, D, bc, &rss);
			sw.soln += GetTimer() - start_time;

			fprintf(output, "%i,%f,%f,%f,%f,%f,%f,%f\n", step, elapsed, rss, sw.conv, sw.step, sw.file, sw.soln, GetTimer());
		}

		swap_pointers(&conc_old, &conc_new);
	}

	write_csv(conc_old, nx, ny, dx, dy, steps);

	/* clean up */
	fclose(output);
	free_arrays(conc_old, conc_new, conc_lap, mask_lap);

	return 0;
}

void param_parser(int argc, char* argv[], int* nx, int* ny, int* nm, int* code, fp_t* dx, fp_t* dy, fp_t* D, fp_t* linStab, int* steps, int* checks)
{
	FILE * input;
	char buffer[256];
	char* pch;
	int inx=0, iny=0, idx=0, idy=0, ins=0, inc=0, idc=0, ico=0, isc=0;

	if (argc != 2) {
		printf("Error: improper arguments supplied.\nUsage: ./%s filename\n", argv[0]);
		exit(-1);
	}

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
				pch = strtok(buffer, " ");

				if (strcmp(pch, "nx") == 0) {
					pch = strtok(NULL, " ");
					*nx = atoi(pch);
					inx = 1;
				} else if (strcmp(pch, "ny") == 0) {
					pch = strtok(NULL, " ");
					*ny = atoi(pch);
					iny = 1;
				} else if (strcmp(pch, "dx") == 0) {
					pch = strtok(NULL, " ");
					*dx = atof(pch);
					idx = 1;
				} else if (strcmp(pch, "dy") == 0) {
					pch = strtok(NULL, " ");
					*dy = atof(pch);
					idy = 1;
				} else if (strcmp(pch, "ns") == 0) {
					pch = strtok(NULL, " ");
					*steps = atoi(pch);
					ins = 1;
				} else if (strcmp(pch, "nc") == 0) {
					pch = strtok(NULL, " ");
					*checks = atoi(pch);
					inc = 1;
				} else if (strcmp(pch, "dc") == 0) {
					pch = strtok(NULL, " ");
					*D = atof(pch);
					idc = 1;
				} else if (strcmp(pch, "co") == 0) {
					pch = strtok(NULL, " ");
					*linStab = atof(pch);
					ico = 1;
				} else if (strcmp(pch, "sc") == 0) {
					pch = strtok(NULL, " ");
					*nm = atoi(pch);
					pch = strtok(NULL, " ");
					*code = atoi(pch);
					isc = 1;
				} else {
					printf("Warning: unknown key %s. Ignoring value.\n", pch);
				}
			}
		}

		/* make sure we got everyone */
		if (! inx) {
			printf("Warning: parameter %s undefined. Using default value, %i.\n", "nx", *nx);
		} else if (! iny) {
			printf("Warning: parameter %s undefined. Using default value, %i.\n", "ny", *ny);
		} else if (! idx) {
			printf("Warning: parameter %s undefined. Using default value, %f.\n", "dx", *dx);
		} else if (! idy) {
			printf("Warning: parameter %s undefined. Using default value, %f.\n", "dy", *dy);
		} else if (! ins) {
			printf("Warning: parameter %s undefined. Using default value, %i.\n", "ns", *steps);
		} else if (! inc) {
			printf("Warning: parameter %s undefined. Using default value, %i.\n", "nc", *checks);
		} else if (! idc) {
			printf("Warning: parameter %s undefined. Using default value, %f.\n", "dc", *D);
		} else if (! ico) {
			printf("Warning: parameter %s undefined. Using default value, %f.\n", "co", *linStab);
		} else if (! isc) {
			printf("Warning: parameter %s undefined. Using default values, %i and %i.\n", "sc", *nm, *code);
		}
	}
}


/** \} */
