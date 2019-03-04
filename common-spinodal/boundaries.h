/**********************************************************************************
 HiPerC: High Performance Computing Strategies for Boundary Value Problems
 Written by Trevor Keller and available from https://github.com/usnistgov/hiperc
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
 \brief Initialize flat composition field with fixed boundary conditions

 The boundary conditions are fixed values of \f$ c_{hi} \f$ along the lower-left
 half and upper-right half walls, no flux everywhere else, with an initial
 values of \f$ c_{lo} \f$ everywhere. These conditions represent a carburizing
 process, with partial exposure (rather than the entire left and right walls)
 to produce an inhomogeneous workload and highlight numerical errors at the
 boundaries.
*/
void apply_initial_conditions(fp_t** conc_old, const int nx, const int ny, const int nm);

/**
 \brief Set fixed value \f$ (c_{hi}) \f$ along left and bottom, zero-flux elsewhere
*/
void apply_boundary_conditions(fp_t** conc_old, const int nx, const int ny, const int nm);

/** \cond SuppressGuard */
#endif /* _BOUNDARIES_H_ */
/** \endcond */
