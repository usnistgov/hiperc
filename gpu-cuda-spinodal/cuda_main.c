/**********************************************************************************
 HiPerC: High Performance Computing Strategies for Boundary Value Problems
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
 \file  cuda_main.c
 \brief CUDA implementation of semi-infinite diffusion equation
*/

/* system includes */
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* common includes */
#include "boundaries.h"
#include "mesh.h"
#include "numerics.h"
#include "output.h"
#include "timer.h"

/* specific includes */
#include "cuda_data.h"

/**
 \brief Run simulation using input parameters specified on the command line
*/
int main(int argc, char* argv[])
{
	FILE* output;

	/* declare default mesh size and resolution */
	fp_t** conc_old, **conc_new, **conc_lap, **conc_div, **mask_lap;
	int bx=32, by=32, nx=202, ny=202, nm=3, code=53;
	const fp_t dx=1.0, dy=1.0;

	/* declare default materials and numerical parameters */
	fp_t M=5.0, kappa=2.0, linStab=0.25, dt=1., elapsed=0., energy=0.;
	int step=0, steps=5000000, checks=100000;
	double start_time=0.;
	struct Stopwatch watch = {0., 0., 0., 0.};

	StartTimer();

	param_parser(argc, argv, &bx, &by, &checks, &code, &M, &kappa, &linStab, &nm, &nx, &ny, &steps);

	dt = linStab / (24.0 * M * kappa);

	/* initialize memory */
	make_arrays(&conc_old, &conc_new, &conc_lap, &conc_div, &mask_lap, nx, ny, nm);
	set_mask(dx, dy, code, mask_lap, nm);

	print_progress(step, steps);

	start_time = GetTimer();
	apply_initial_conditions(conc_old, nx, ny, nm);
	watch.step = GetTimer() - start_time;

	/* initialize GPU */
	struct CudaData dev;
	init_cuda(conc_old, mask_lap, nx, ny, nm, &dev);

	/* write initial condition data */
	start_time = GetTimer();
	write_png(conc_old, nx, ny, 0);

	/* prepare to log comparison to analytical solution */
	output = fopen("runlog.csv", "w");
	if (output == NULL) {
		printf("Error: unable to %s for output. Check permissions.\n", "runlog.csv");
		exit(-1);
	}
	watch.file = GetTimer() - start_time;

	fprintf(output, "iter,sim_time,energy,conv_time,step_time,IO_time,run_time\n");
	fprintf(output, "%i,%f,%f,%f,%f,%f,%f\n", step, elapsed, nx*dx * ny*dy * chem_energy(0.5),
	        watch.conv, watch.step, watch.file, GetTimer());
	fflush(output);

	/* do the work */
	for (step = 1; step < steps+1; step++) {
		print_progress(step, steps);

		/* === Start Architecture-Specific Kernel === */
		device_boundaries(dev.conc_old, nx, ny, nm, bx, by);

		start_time = GetTimer();
		device_laplacian(dev.conc_old, dev.conc_lap, kappa, nx, ny, nm, bx, by);
		watch.conv += GetTimer() - start_time;

		device_boundaries(dev.conc_lap, nx, ny, nm, bx, by);

		start_time = GetTimer();
		device_divergence(dev.conc_lap, dev.conc_div, nx, ny, nm, bx, by);
		watch.conv += GetTimer() - start_time;

		start_time = GetTimer();
		device_composition(dev.conc_old, dev.conc_new, dev.conc_div, nx, ny, nm, bx, by, M, dt);
		watch.conv += GetTimer() - start_time;

		swap_pointers_1D(&(dev.conc_old), &(dev.conc_new));
		/* === Finish Architecture-Specific Kernel === */

		elapsed += dt;

		if (step % checks == 0) {
			/* transfer result to host (conc_new) from device (dev.conc_old) */
			start_time = GetTimer();
			read_out_result(conc_new, dev.conc_old, nx, ny);
			watch.file += GetTimer() - start_time;

			free_energy(conc_new, conc_lap, dx, dy, nx, ny, nm, kappa, &energy);

			start_time = GetTimer();
			write_png(conc_new, nx, ny, dt * step);
			watch.file += GetTimer() - start_time;

			fprintf(output, "%i,%f,%f,%f,%f,%f,%f\n", step, elapsed, energy,
			        watch.conv, watch.step, watch.file, GetTimer());
			fflush(output);
		}
	}

	write_csv(conc_new, nx, ny, dx, dy, dt * steps);

	/* clean up */
	fclose(output);
	free_arrays(conc_old, conc_new, conc_lap, conc_div, mask_lap);
	free_cuda(&dev);

	return 0;
}
