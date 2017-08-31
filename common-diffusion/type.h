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

/** \defgroup DIFFUSION Diffusion equation benchmark */

/** \addtogroup DIFFUSION
 \{
*/

/**
 \file  common-diffusion/type.h
 \brief Definition of scalar data type and Doxygen diffusion group
*/

#ifndef _TYPE_H_
#define _TYPE_H_

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
	double conv;
	double step;
	double file;
	double soln;
};

#endif /* _TYPE_H_ */

/** \} */
