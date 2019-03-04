/**********************************************************************************
 HiPerC: High Performance Computing Strategies for Boundary Value Problems
 Written by Trevor Keller and available from https://github.com/usnistgov/hiperc
 **********************************************************************************/

/**
 \file  type.h
 \brief Definition of scalar data type and Doxygen diffusion group
*/

/** \cond SuppressGuard */
#ifndef _TYPE_H_
#define _TYPE_H_
/** \endcond */

/**
 Specify the basic data type to achieve the desired accuracy in floating-point
 arithmetic: float for single-precision, double for double-precision. This
 choice propagates throughout the code, and may significantly affect runtime
 on GPU hardware.
*/
typedef double fp_t;

/**
 Container for timing data
*/
struct Stopwatch {
	/**
	   Cumulative time executing compute_laplacian() and compute_divergence()
	*/
	fp_t conv;

	/**
	 Cumulative time executing solve_diffusion_equation()
	*/
	fp_t step;

	/**
	 Cumulative time executing write_csv() and write_png()
	*/
	fp_t file;

	/**
	 Cumulative time executing check_solution()
	*/
	fp_t soln;
};

/** \cond SuppressGuard */
#endif /* _TYPE_H_ */
/** \endcond */
