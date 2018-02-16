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
 \brief Write 9-point Laplacian stencil into convolution mask

 \f$5\times5\f$ mask, 9 values, truncation error \f$\mathcal{O}(\Delta x^4)\f$

 Provided for testing and demonstration of scalability, only:
 as the name indicates, this 9-point stencil is computationally
 more expensive than the \f$3\times3\f$ version. If your code requires
 \f$\mathcal{O}(\Delta x^4)\f$ accuracy, please use nine_point_Laplacian_stencil().
*/
void slow_nine_point_Laplacian_stencil(const fp_t dx, const fp_t dy, fp_t** mask_lap, const int nm);

/**
   \brief Perform the convolution of the mask matrix with the composition matrix

   If the convolution mask is the Laplacian stencil, the convolution evaluates
   the discrete Laplacian of the composition field. Other masks are possible, for
   example the Sobel filters for edge detection. This function is general
   purpose: as long as the dimensions \a nx, \a ny, and \a nm are properly specified,
   the convolution will be correctly computed.
*/
void compute_convolution(fp_t** const conc_old, fp_t** conc_lap, fp_t** const mask_lap,
                         const int nx, const int ny, const int nm);

/**
   \brief Update composition field using explicit Euler discretization (forward-time centered space)
*/
void update_composition(fp_t** conc_old, fp_t** conc_lap, fp_t** conc_new,
				   const int nx, const int ny, const int nm,
				   const fp_t D, const fp_t dt);

/**
   \brief Manufactured shift, Equation 3
*/
void manufactured_shift(const fp_t x,  const fp_t t,
                        const fp_t A1, const fp_t A2,
                        const fp_t B1, const fp_t B2,
                        const fp_t C2, fp_t* alpha);

/**
   \brief Manufactured solution, Equation 2
*/
void manufactured_solution(const fp_t x,  const fp_t y,  const fp_t t,
                           const fp_t A1, const fp_t A2,
                           const fp_t B1, const fp_t B2,
                           const fp_t C2, const fp_t kappa,
                           fp_t* eta);

/**
   \brief Compare numerical and manufactured solutions of the Allen-Cahn equation
   \return L2 norm of the error

   Overwrites \a conc_lap, into which the point-wise error is written.
   L2 is then computed as the root of the sum of the point-wise values.
*/
void compute_L2_norm(fp_t** conc_new, fp_t** conc_lap,
                     const int nx,    const int ny,   const int nm,
                     const fp_t dx,   const fp_t dy,  const fp_t elapsed,
                     const fp_t A1,   const fp_t A2,
                     const fp_t B1,   const fp_t B2,
                     const fp_t C2,   const fp_t kappa,
                     fp_t* L2);

/** \cond SuppressGuard */
#endif /* _NUMERICS_H_ */
/** \endcond */
