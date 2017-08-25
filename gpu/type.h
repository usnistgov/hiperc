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

/** \defgroup GPU Benchmarks using GPU hardware */
/** \defgroup cuda CUDA implementation */
/** \defgroup openacc OpenAcc implementation */

/** \addtogroup GPU
 \{
*/

/**
 \file  gpu/type.h
 \brief Definition of scalar data type
*/

#ifndef _TYPE_H_
#define _TYPE_H_

/**
 Specify the basic data type to achieve the desired accuracy in floating-point
 arithmetic: float for single-precision, double for double-precision.
*/
typedef double fp_t;

#endif /* _TYPE_H_ */

/** \} */
