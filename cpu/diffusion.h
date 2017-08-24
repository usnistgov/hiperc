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

/** \defgroup CPU Benchmarks using CPU hardware */
/** \defgroup serial Serial implementation */
/** \defgroup openmp OpenMP implementation */
/** \defgroup tbb Threading Building Blocks implementation */

/** \addtogroup CPU \{ */

/**
 \file  cpu/diffusion.h
 \brief Declaration of diffusion equation function prototypes for CPU benchmarks
*/

#ifndef _DIFFUSION_H_
#define _DIFFUSION_H_

/**
 Specify the basic data type to achieve the desired accuracy in floating-point
 arithmetic: float for single-precision, double for double-precision.
*/
typedef double fp_t;

/* Mesh handling: implemented in mesh.c */
void make_arrays(fp_t*** conc_old, fp_t*** conc_new, fp_t*** conc_lap, fp_t*** mask_lap,
                 int nx, int ny, int nm);
void free_arrays(fp_t** conc_old, fp_t** conc_new, fp_t** conc_lap, fp_t** mask_lap);
void swap_pointers(fp_t*** conc_old, fp_t*** conc_new);

/* Boundary conditions: implemented in boundaries.c */
void set_boundaries(fp_t bc[2][2]);
void apply_initial_conditions(fp_t** conc_old, int nx, int ny, int nm, fp_t bc[2][2]);
void apply_boundary_conditions(fp_t** conc_old, int nx, int ny, int nm, fp_t bc[2][2]);

/* Discretized mathematical operations: implemented in discretization.c[pp] */
void set_threads(int n);
void set_mask(fp_t dx, fp_t dy, int nm, fp_t** mask_lap);
void compute_convolution(fp_t** conc_old, fp_t** conc_lap, fp_t** mask_lap, int nx, int ny, int nm);
void solve_diffusion_equation(fp_t** conc_old, fp_t** conc_new, fp_t** conc_lap,
                              int nx, int ny, int nm,
                              fp_t D, fp_t dt, fp_t *elapsed);
void analytical_value(fp_t x, fp_t t, fp_t D, fp_t bc[2][2], fp_t* c);
void check_solution(fp_t** conc_new,
                    int nx, int ny, fp_t dx, fp_t dy, int nm,
                    fp_t elapsed, fp_t D, fp_t bc[2][2], fp_t* rss);

/* Output results: implemented in output.c */
void print_progress(const int step, const int steps);
void write_csv(fp_t** conc, int nx, int ny, fp_t dx, fp_t dy, int step);
void write_png(fp_t** conc, int nx, int ny, int step);

/* Time function calls: implemented in timer.c */
void StartTimer();
double GetTimer();

#endif /* _DIFFUSION_H_ */

/** \} */
