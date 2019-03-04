/**********************************************************************************
 HiPerC: High Performance Computing Strategies for Boundary Value Problems
 Written by Trevor Keller and available from https://github.com/usnistgov/hiperc
 **********************************************************************************/

/**
 \file  output.h
 \brief Declaration of output function prototypes for diffusion benchmarks
*/

/** \cond SuppressGuard */
#ifndef _OUTPUT_H_
#define _OUTPUT_H_
/** \endcond */

#include "type.h"

/**
 \brief Read parameters from file specified on the command line
*/
void param_parser(int argc, char* argv[], int* bx, int* by, int* checks, int* code,
                  fp_t* M, fp_t* kappa, fp_t* linStab, int* nm, int* nx, int* ny, int* steps);

/**
 \brief Prints timestamps and a 20-point progress bar to stdout

 Call inside the timestepping loop, near the top, e.g.
 \code
 for (int step=0; step<steps; step++) {
 	print_progress(step, steps);
 	take_a_step();
 	elapsed += dt;
 }
 \endcode
*/
void print_progress(const int step, const int steps);

/**
 \brief Writes scalar composition field to diffusion.???????.csv
*/
void write_csv(fp_t** conc, const int nx, const int ny, const fp_t dx, const fp_t dy, const int step);

/**
 \brief Writes scalar composition field to diffusion.???????.png
*/
void write_png(fp_t** conc, const int nx, const int ny, const int step);

/** \cond SuppressGuard */
#endif /* _OUTPUT_H_ */
/** \endcond */
