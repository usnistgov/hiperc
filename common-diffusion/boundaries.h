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
 \file  boundaries.h
 \brief Declaration of boundary condition function prototypes
*/

/** \cond SuppressGuard */
#ifndef _BOUNDARIES_H_
#define _BOUNDARIES_H_
/** \endcond */

#include "type.h"

/**
 \brief Set values to be used along the simulation domain boundaries

 Indexing is row-major, i.e. \f$ A[y][x] \f$, so
 \f$ bc = [[y_{lo},y_{hi}], [x_{lo},x_{hi}]] \f$.
*/
void set_boundaries(fp_t bc[2][2]);

/**
 \brief Initialize flat composition field with fixed boundary conditions

 The boundary conditions are fixed values of \f$ c_{hi} \f$ along the lower-left
 half and upper-right half walls, no flux everywhere else, with an initial
 values of \f$ c_{lo} \f$ everywhere. These conditions represent a carburizing
 process, with partial exposure (rather than the entire left and right walls)
 to produce an inhomogeneous workload and highlight numerical errors at the
 boundaries.
*/
void apply_initial_conditions(fp_t** conc_old, const int nx, const int ny, const int nm, fp_t bc[2][2]);

/**
 \brief Set fixed value \f$ (c_{hi}) \f$ along left and bottom, zero-flux elsewhere
*/
void apply_boundary_conditions(fp_t** conc_old, const int nx, const int ny, const int nm, fp_t bc[2][2]);

/** \cond SuppressGuard */
#endif /* _BOUNDARIES_H_ */
/** \endcond */
