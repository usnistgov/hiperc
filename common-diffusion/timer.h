/**********************************************************************************
 HiPerC: High Performance Computing Strategies for Boundary Value Problems
 Written by Trevor Keller and available from https://github.com/usnistgov/hiperc
 **********************************************************************************/

/**
 \file  timer.h
 \brief Declaration of timer function prototypes for diffusion benchmarks
*/


/** \cond SuppressGuard */
#ifndef _TIMER_H_
#define _TIMER_H_
/** \endcond */

/**
 \brief Set CPU frequency and begin timing
*/
void StartTimer();

/**
 \brief Return elapsed time in seconds
*/
double GetTimer();

/** \cond SuppressGuard */
#endif /* _TIMER_H_ */
/** \endcond */
