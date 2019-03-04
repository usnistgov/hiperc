/**********************************************************************************
 HiPerC: High Performance Computing Strategies for Boundary Value Problems
 Written by Trevor Keller and available from https://github.com/usnistgov/hiperc
 **********************************************************************************/

/**
 \file  numerics.h
 \brief Declaration of Laplacian operator and analytical solution functions
*/

/** \cond SuppressGuard */
#ifndef _NUMERICS_H_
#define _NUMERICS_H_
/** \endcond */

#include "type.h"

/**
 \brief Maximum width of the convolution mask (Laplacian stencil) array
*/
#define MAX_MASK_W 5

/**
 \brief Maximum height of the convolution mask (Laplacian stencil) array
*/
#define MAX_MASK_H 5

/**
 \brief Specify which stencil (mask) to use for the Laplacian (convolution)

 The mask corresponding to the numerical code will be applied. The suggested
 encoding is mask width as the ones digit and value count as the tens digit,
 \a e.g. 53 specifies five_point_Laplacian_stencil(), while
 93 specifies nine_point_Laplacian_stencil().

 To add your own mask (stencil), add a case to this function with your
 chosen numerical encoding, then specify that code in the input parameters file
 (params.txt by default). Note that, for a Laplacian stencil, the sum of the
 coefficients must equal zero and \a nm must be an odd integer.

 If your stencil is larger than \f$ 5\times 5\f$, you must increase the values
 defined by #MAX_MASK_W and #MAX_MASK_H.
*/
void set_mask(const fp_t dx, const fp_t dy, const int code, fp_t** mask_lap, const int nm);

/**
 \brief Write 5-point Laplacian stencil into convolution mask

 \f$3\times3\f$ mask, 5 values, truncation error \f$\mathcal{O}(\Delta x^2)\f$
*/
void five_point_Laplacian_stencil(const fp_t dx, const fp_t dy, fp_t** mask_lap, const int nm);

/**
 \brief Write 9-point Laplacian stencil into convolution mask

 \f$3\times3\f$ mask, 9 values, truncation error \f$\mathcal{O}(\Delta x^4)\f$
*/
void nine_point_Laplacian_stencil(const fp_t dx, const fp_t dy, fp_t** mask_lap, const int nm);

/**
 \brief Write 13-point biharmonic stencil into convolution mask

 \f$5\times5\f$ mask, 13 values, truncation error \f$\mathcal{O}(\Delta x^2)\f$
*/
void biharmonic_stencil(const fp_t dx, const fp_t dy, fp_t** mask_lap, const int nm);

/**
   \brief Compute interior Laplacian from old composition data
*/
void compute_laplacian(fp_t** const conc_old, fp_t** conc_lap, fp_t** const mask_lap,
                       const fp_t kappa, const int nx, const int ny, const int nm);

/**
 \brief Compute exterior Laplacian (divergence of gradient of Laplacian)
*/
void compute_divergence(fp_t** conc_lap, fp_t** conc_div, fp_t** const mask_lap,
                        const int nx, const int ny, const int nm);

/**
 \brief Update composition field using explicit Euler discretization (forward-time centered space)
*/
void update_composition(fp_t** conc_old, fp_t** conc_div, fp_t** conc_new,
                        const int nx, const int ny, const int nm,
                        const fp_t D, const fp_t dt);

/**
   \brief Compute gradient-squared, truncation error \f$\mathcal{O}(\Delta x^2)\f$
*/
fp_t grad_sq(fp_t** conc, const int x, const int y,
             const fp_t dx, const fp_t dy,
             const int nx, const int ny);

/**
   \brief Compute chemical free energy density
*/
fp_t chem_energy(const fp_t C);

/**
 \brief Compute total free energy
*/
void free_energy(fp_t** conc_new, fp_t** conc_lap,
                 const fp_t dx, const fp_t dy,
                 const int nx, const int ny, const int nm,
                 const fp_t kappa, fp_t* energy);

/** \cond SuppressGuard */
#endif /* _NUMERICS_H_ */
/** \endcond */
