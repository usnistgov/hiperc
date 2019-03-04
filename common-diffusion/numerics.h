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

/**
   \brief Compare numerical and analytical solutions of the diffusion equation
   \return Residual sum of squares (RSS), normalized to the domain size.

   Overwrites \a conc_lap, into which the point-wise RSS is written.
   Normalized RSS is then computed as the sum of the point-wise values.
*/
void check_solution(fp_t** conc_new, fp_t** conc_lap, const int nx, const int ny,
                    const fp_t dx, const fp_t dy, const int nm,
                    const fp_t elapsed, const fp_t D, fp_t* rss);

/** \cond SuppressGuard */
#endif /* _NUMERICS_H_ */
/** \endcond */
