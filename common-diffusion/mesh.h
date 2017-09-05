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

/**
 \file  mesh.h
 \brief Declaration of mesh function prototypes for diffusion benchmarks
*/

/** \cond SuppressGuard */
#ifndef _MESH_H_
#define _MESH_H_
/** \endcond */

#include "type.h"

/**
 \brief Allocate 2D arrays to store scalar composition values

 Arrays are allocated as 1D arrays, then 2D pointer arrays are mapped over the
 top. This facilitates use of either 1D or 2D data access, depending on whether
 the task is spatially dependent or not.
*/
void make_arrays(fp_t*** conc_old, fp_t*** conc_new, fp_t*** conc_lap, fp_t*** mask_lap,
                 int nx, int ny, int nm);

/**
 \brief Free dynamically allocated memory
*/
void free_arrays(fp_t** conc_old, fp_t** conc_new, fp_t** conc_lap, fp_t** mask_lap);

/**
 \brief Swap pointers to 2D arrays

 Rather than copy data from \c fp_t** \a conc_old into \c fp_t** \a conc_new,
 an expensive operation, simply trade the top-most pointers. New becomes old,
 old becomes new, with no data lost and in almost no time.
*/
void swap_pointers(fp_t*** conc_old, fp_t*** conc_new);

/**
 \brief Swap pointers to data underlying 1D arrays

 Rather than copy data from \c fp_t* \a conc_old[0] into \c fp_t*
 \a conc_new[0], an expensive operation, simply trade the top-most pointers.
 New becomes old, old becomes new,  with no data lost and in almost no time.
*/
void swap_pointers_1D(fp_t** conc_old, fp_t** conc_new);

/** \cond SuppressGuard */
#endif /* _MESH_H_ */
/** \endcond */
