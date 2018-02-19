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
 \brief CUDA implementation of PFHub Benchmark 7: Method of Manufactured Solutions
*/

/* system includes */
#include <assert.h>
#include <sys/stat.h>
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

int file_exist (char *filename)
{
	struct stat   buffer;
	return (stat (filename, &buffer) == 0);
}

/**
 \brief Run simulation using input parameters specified on the command line
*/
int main(int argc, char* argv[])
{
	FILE* output;
	int newfile = 1;

	/* declare default mesh size and resolution */
	fp_t** conc_old, **conc_new, **conc_lap, **mask_lap;
	int bx=32, by=32, nx=200, ny=100, nm=3, code=93;
	fp_t dx=0.01, dy=0.01, h;

	/* declare default materials and numerical parameters */
	fp_t kappa=0.0004, A1=0.0075, A2=0.03, B1=8.*M_PI, B2=22.*M_PI, C2=0.0625*M_PI;
	fp_t linStab=0.01, dt=2.0e-4, elapsed=0., L2=0.;
	int step=0, steps=512;
	double start_time=0.;
	struct Stopwatch watch = {0., 0., 0., 0.};

	const fp_t epsilon=1.0e-14, final=8.;

	StartTimer();

	param_parser(argc, argv, &bx, &by, &code, &steps, &dx, &dy, &linStab, &nx, &ny, &nm,
		         &A1, &A2, &B1, &B2, &C2, &kappa);

	if (linStab > 1.0 - epsilon) {
		printf("Error: CFL condition is %.2f. Fix your numerics.\n", linStab);
		return 1;
	}

	h = (dx > dy) ? dy : dx;
	dt = (linStab * h * h) / (4.0 * kappa);

	/* initialize memory */
	make_arrays(&conc_old, &conc_new, &conc_lap, &mask_lap, nx, ny, nm);
	set_mask(dx, dy, code, mask_lap, nm);

	print_progress(step, steps);

	start_time = GetTimer();
	apply_initial_conditions(conc_old, dx, dy, nx, ny, nm, A1, A2, B1, B2, C2, kappa);
	watch.step = GetTimer() - start_time;

	/* initialize GPU */
	struct CudaData dev;
	init_cuda(conc_old, mask_lap, nx, ny, nm, &dev);

	/* write initial condition data */
	start_time = GetTimer();
	write_png(conc_old, nx, ny, nm, 0);

	if (file_exist("runlog.csv") == 1) {
		/* file does not exist */
		newfile = 0;
	}

	/* prepare to log comparison to analytical solution */
	output = fopen("runlog.csv", "a");
	if (output == NULL) {
		printf("Error: unable to %s for output. Check permissions.\n", "runlog.csv");
		exit(-1);
	}
	watch.file = GetTimer() - start_time;

	if (newfile == 1)
		fprintf(output, "iter,sim_time,CFL,dt,dx,L2,conv_time,step_time,IO_time,soln_time,run_time\n");
	/*
	  fprintf(output, "%i,%f,%f,%f,%f,%f,%f,%f\n", step, elapsed, L2,
	          watch.conv, watch.step, watch.file, watch.soln, GetTimer());
	*/
	fflush(output);

	/* do the work */
	for (step = 1; step < steps+1; step++) {
		print_progress(step, steps);

		/* Make sure we hit the endpoint exactly */
		if (elapsed + dt > final + epsilon)
			dt = final - elapsed + epsilon;

		/* === Start Architecture-Specific Kernel === */
		device_boundaries(dev.conc_old,
						  bx, by,
						  nx, ny, nm);

		start_time = GetTimer();
		device_convolution(dev.conc_old, dev.conc_lap,
						   bx, by,
						   nx, ny, nm);
		watch.conv += GetTimer() - start_time;

		start_time = GetTimer();
		device_evolution(dev.conc_old, dev.conc_new, dev.conc_lap,
						 bx, by,
						 dx, dy, dt,
						 elapsed,
						 nx, ny, nm,
						 A1, A2, B1, B2, C2, kappa);
		watch.conv += GetTimer() - start_time;

		swap_pointers_1D(&(dev.conc_old), &(dev.conc_new));
		/* === Finish Architecture-Specific Kernel === */

		/* Note: There is no intermediate output, only the final answer */

		elapsed += dt;
	}

	/* transfer result to host (conc_new) from device (dev.conc_old) */
	start_time = GetTimer();
	read_out_result(conc_new, dev.conc_old, nx, ny);
	watch.file += GetTimer() - start_time;

	start_time = GetTimer();
	compute_L2_norm(conc_new, conc_lap,
					dx, dy,
					elapsed,
					nx, ny, nm,
					A1, A2,
					B1, B2,
					C2, kappa,
					&L2);
	watch.soln += GetTimer() - start_time;

	fprintf(output, "%i,%f,%f,%.12f,%.12f,%.12f,%f,%f,%f,%f,%f\n",
	                steps, elapsed, linStab,
			        dt, dx, L2,
	                watch.conv, watch.step, watch.file, watch.soln, GetTimer());

	write_csv(conc_new, dx, dy, elapsed, nx, ny, steps, A1, A2, B1, B2, C2, kappa);

	start_time = GetTimer();
	write_png(conc_new, nx, ny, nm, steps);
	watch.file += GetTimer() - start_time;

	/* clean up */
	fclose(output);
	free_arrays(conc_old, conc_new, conc_lap, mask_lap);
	free_cuda(&dev);

	return 0;
}
