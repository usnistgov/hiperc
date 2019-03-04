/**********************************************************************************
 HiPerC: High Performance Computing Strategies for Boundary Value Problems
 Written by Trevor Keller and available from https://github.com/usnistgov/hiperc
 **********************************************************************************/

/**
 \file  phi_main.c
 \brief OpenMP implementation of semi-infinite diffusion equation
*/

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "boundaries.h"
#include "discretization.h"
#include "mesh.h"
#include "numerics.h"
#include "output.h"
#include "timer.h"

/**
 \brief Run simulation using input parameters specified on the command line
*/
int main(int argc, char* argv[])
{
	FILE * output;

	/* declare default mesh size and resolution */
	fp_t **conc_old, **conc_new, **conc_lap, **mask_lap;
	int bx=32, by=32, nx=512, ny=512, nm=3, code=53;
	fp_t dx=0.5, dy=0.5, h;
	fp_t bc[2][2];

	/* declare default materials and numerical parameters */
	fp_t D=0.00625, linStab=0.1, dt=1., elapsed=0., rss=0.;
	int step=0, steps=100000, checks=10000;
	double start_time=0.;
	struct Stopwatch sw = {0., 0., 0., 0.};

	StartTimer();

	param_parser(argc, argv, &bx, &by, &checks, &code, &D, &dx, &dy, &linStab, &nm, &nx, &ny, &steps);

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
	fflush(output);

	/* Note: block is equivalent to a typical
	  for (int step=1; step < steps+1; step++),
	  1-indexed so as not to overwrite the initial condition image,
	  but the loop-internals are farmed out to a coprocessor.
	  So we use a while loop instead. */

	/* do the work */
	step = 0;
	print_progress(step, steps);
	while (step < steps) {
		if (checks > steps - step)
			checks = steps - step;

		assert(step + checks <= steps);

		solve_diffusion_equation(conc_old, conc_new, conc_lap, mask_lap, nx, ny,
		                         nm, bc, D, dt, checks, &elapsed, &sw);
		/* returns after swapping pointers: new data is in conc_old */

		for (int i = 0; i < checks; i++) {
			step++;
			print_progress(step, steps);
		}

		start_time = GetTimer();
		write_png(conc_old, nx, ny, step);
		sw.file += GetTimer() - start_time;

		start_time = GetTimer();
		check_solution(conc_old, conc_lap, nx, ny, dx, dy, nm, elapsed, D, bc, &rss);
		sw.soln += GetTimer() - start_time;

		fprintf(output, "%i,%f,%f,%f,%f,%f,%f,%f\n", step, elapsed, rss, sw.conv, sw.step, sw.file, sw.soln, GetTimer());
		fflush(output);
	}

	write_csv(conc_old, nx, ny, dx, dy, step);

	/* clean up */
	fclose(output);
	free_arrays(conc_old, conc_new, conc_lap, mask_lap);

	return 0;
}
