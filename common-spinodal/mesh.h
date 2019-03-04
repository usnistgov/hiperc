/**********************************************************************************
 HiPerC: High Performance Computing Strategies for Boundary Value Problems
 Written by Trevor Keller and available from https://github.com/usnistgov/hiperc
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
void make_arrays(fp_t*** conc_old, fp_t*** conc_new,
                 fp_t*** conc_lap, fp_t*** conc_div,
                 fp_t*** mask_lap,
                 const int nx, const int ny, const int nm);

/**
 \brief Free dynamically allocated memory
*/
void free_arrays(fp_t** conc_old, fp_t** conc_new,
                 fp_t** conc_lap, fp_t** conc_div,
                 fp_t** mask_lap);

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
