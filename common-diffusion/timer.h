/**********************************************************************************
 HIPERC: High Performance Computing Strategies for Boundary Value Problems
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
