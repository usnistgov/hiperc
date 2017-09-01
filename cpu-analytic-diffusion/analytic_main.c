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

/**
 \file  analytic_main.c
 \brief Analytical solution to semi-infinite diffusion equation
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "type.h"
#include "mesh.h"
#include "numerics.h"
#include "output.h"
#include "timer.h"

/**
 \brief Update the scalar composition field using analytical solution
*/
void solve_diffusion_equation(fp_t** conc, int nx, int ny, int nm, fp_t dx, fp_t dy, fp_t D, fp_t dt, fp_t elapsed)
{
	int i, j;
	fp_t r, cal, car;
	fp_t bc[2][2] = {{1., 1.}, {1., 1.}};

	for (j = nm/2; j < ny-nm/2; j++) {
		for (i = nm/2; i < nx-nm/2; i++) {
			/* shortest distance to left-wall source */
			r = distance_point_to_segment(dx * (nm/2), dy * (nm/2),
			                              dx * (nm/2), dy * (ny/2),
			                              dx * i, dy * j);
			analytical_value(r, elapsed, D, bc, &cal);

			/* shortest distance to right-wall source */
			r = distance_point_to_segment(dx * (nx-1-nm/2), dy * (ny/2),
			                              dx * (nx-1-nm/2), dy * (ny-1-nm/2),
			                              dx * i, dy * j);
			analytical_value(r, elapsed, D, bc, &car);

			/* superposition of analytical solutions */
			conc[j][i] = cal + car;
		}
	}
}

/**
 \brief Run simulation using input parameters specified on the command line

 Program will write a series of PNG image files to visualize the  scalar
 composition field, plus \c delta.png showing the difference between the
 analytical result and the image stored in
 \c ../common-diffusion/diffusion.10000.png.
*/
int main(int argc, char* argv[])
{
	FILE * output;

	/* declare mesh size and resolution */
	fp_t **conc_old, **conc_new, **conc_lap, **mask_lap;
	int nx=512, ny=512, nm=3, code=53;
	fp_t dx=0.5, dy=0.5, h=0.5;

	/* declare materials and numerical parameters */
	fp_t D=0.00625, linStab=0.1, dt=1., elapsed=0., rss=0.;
	int step=0, steps=100000, checks=10000;
	double conv_time=0., file_time=0., soln_time=0., start_time=0., step_time=0.;

	StartTimer();

	param_parser(argc, argv, &nx, &ny, &nm, &code, &dx, &dy, &D, &linStab, &steps, &checks);

	h = (dx > dy) ? dy : dx;
	dt = (linStab * h * h) / (4.0 * D);

	/* initialize memory */
	make_arrays(&conc_old, &conc_new, &conc_lap, &mask_lap, nx, ny, nm);

	start_time = GetTimer();
	solve_diffusion_equation(conc_old, nx, ny, nm, dx, dy, D, dt, dt);
	step_time = GetTimer() - start_time;

	/* write initial condition data */
	start_time = GetTimer();
	write_png(conc_old, nx, ny, 0);
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
		print_progress(step-1, steps);

		if (step % checks == 0) {
			start_time = GetTimer();
			solve_diffusion_equation(conc_new, nx, ny, nm, dx, dy, D, dt, elapsed);
			step_time += GetTimer() - start_time;

			start_time = GetTimer();
			write_png(conc_new, nx, ny, step);
			file_time += GetTimer() - start_time;
		}

		elapsed += dt;
	}

	/* clean up */
	fclose(output);
	free_arrays(conc_old, conc_new, conc_lap, mask_lap);

	return 0;
}
