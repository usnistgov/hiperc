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
 \brief Compute Euclidean distance between two points, \a a and \a b
*/
fp_t euclidean_distance(const fp_t ax, const fp_t ay,
                        const fp_t bx, const fp_t by);

/**
 \brief Compute Manhattan distance between two points, \a a and \a b
*/
fp_t manhattan_distance(const fp_t ax, const fp_t ay,
                        const fp_t bx, const fp_t by);

/**
 \brief Compute minimum distance from point \a p to a line segment bounded by points \a a and \a b

 This function computes the projection of \a p onto \a ab, limiting the
 projected range to [0, 1] to handle projections that fall outside of \a ab.
 Implemented after Grumdrig on Stackoverflow, https://stackoverflow.com/a/1501725.
*/
fp_t distance_point_to_segment(const fp_t ax, const fp_t ay,
                               const fp_t bx, const fp_t by,
                               const fp_t px, const fp_t py);

/**
 \brief Analytical solution of the diffusion equation for a carburizing process

 For 1D diffusion through a semi-infinite domain with initial and far-field
 composition \f$ c_{\infty} \f$ and boundary value \f$ c(x=0, t) = c_0 \f$
 with constant diffusivity \e D, the solution to Fick's second law is
 \f[ c(x,t) = c_0 - (c_0 - c_{\infty})\mathrm{erf}\left(\frac{x}{\sqrt{4Dt}}\right) \f]
 which reduces, when \f$ c_{\infty} = 0 \f$, to
 \f[ c(x,t) = c_0\left[1 - \mathrm{erf}\left(\frac{x}{\sqrt{4Dt}}\right)\right]. \f]
*/
void analytical_value(const fp_t x, const fp_t t, const fp_t D, fp_t* c);

/** \cond SuppressGuard */
#endif /* _NUMERICS_H_ */
/** \endcond */
