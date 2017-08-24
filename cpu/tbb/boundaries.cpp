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
 \file  cpu/tbb/boundaries.cpp
 \brief Implementation of boundary condition functions with TBB threading
*/

#include <math.h>
#include <tbb/tbb.h>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <tbb/blocked_range2d.h>

#include "diffusion.h"

/**
 \brief Set values to be used along the simulation domain boundaries

 Indexing is row-major, i.e. \f$A[y][x]\f$, so \f$\mathrm{bc} = [[y_{lo},y_{hi}], [x_{lo},x_{hi}]]\f$.
*/
void set_boundaries(fp_t bc[2][2])
{
	fp_t clo = 0.0, chi = 1.0;
	bc[0][0] = clo; /* bottom boundary */
	bc[0][1] = clo; /* top boundary */
	bc[1][0] = chi; /* left boundary */
	bc[1][1] = chi; /* right boundary */
}

/**
 \brief Initialize flat composition field with fixed boundary conditions

 The boundary conditions are fixed values of \f$c_{hi}\f$ along the lower-left half and
 upper-right half walls, no flux everywhere else, with an initial values of \f$c_{lo}\f$
 everywhere. These conditions represent a carburizing process, with partial
 exposure (rather than the entire left and right walls) to produce an
 inhomogeneous workload and highlight numerical errors at the boundaries.
*/
void apply_initial_conditions(fp_t** conc, int nx, int ny, int nm, fp_t bc[2][2])
{
	/* apply flat field values  (lambda function) */
	tbb::parallel_for(tbb::blocked_range2d<int>(0, nx, 0, ny),
		[=](const tbb::blocked_range2d<int>& r) {
			for (int j = r.cols().begin(); j != r.cols().end(); j++) {
				for (int i = r.rows().begin(); i != r.rows().end(); i++) {
					conc[j][i] = bc[0][0];
				}
			}
		}
	);

	/* apply left boundary values  (lambda function) */
	tbb::parallel_for(tbb::blocked_range2d<int>(0, 1+nm/2, 0, ny/2),
		[=](const tbb::blocked_range2d<int>& r) {
			for (int j = r.cols().begin(); j != r.cols().end(); j++) {
				for (int i = r.rows().begin(); i != r.rows().end(); i++) {
					conc[j][i] = bc[1][0];
				}
			}
		}
	);

	/* apply right boundary values  (lambda function) */
	tbb::parallel_for( tbb::blocked_range2d<int>(nx-1-nm/2, nx, ny/2, ny),
		[=](const tbb::blocked_range2d<int>& r) {
			for (int j = r.cols().begin(); j != r.cols().end(); j++) {
				for (int i = r.rows().begin(); i != r.rows().end(); i++) {
					conc[j][i] = bc[1][1];
				}
			}
		}
	);
}

/**
 \brief Set fixed value (\f$c_{hi}\f$) along left and bottom, zero-flux elsewhere
*/
void apply_boundary_conditions(fp_t** conc, int nx, int ny, int nm, fp_t bc[2][2])
{
	/* apply left boundary values  (lambda function) */
	tbb::parallel_for(tbb::blocked_range2d<int>(0, 1+nm/2, 0, ny/2),
		[=](const tbb::blocked_range2d<int>& r) {
			for (int j = r.cols().begin(); j != r.cols().end(); j++) {
				for (int i = r.rows().begin(); i != r.rows().end(); i++) {
					conc[j][i] = bc[1][0];
				}
			}
		}
	);

	/* apply right boundary values  (lambda function) */
	tbb::parallel_for( tbb::blocked_range2d<int>(nx-1-nm/2, nx, ny/2, ny),
		[=](const tbb::blocked_range2d<int>& r) {
			for (int j = r.cols().begin(); j != r.cols().end(); j++) {
				for (int i = r.rows().begin(); i != r.rows().end(); i++) {
					conc[j][i] = bc[1][1];
				}
			}
		}
	);

	/* apply no-flux boundary conditions  (serial) */
	for (int j = 0; j < ny; j++) {
		for (int i = nm/2; i > 0; i--)
			conc[j][i-1] = conc[j][i]; /* left condition */
		for (int i = nx-1-nm/2; i < nx-1; i++)
			conc[j][i+1] = conc[j][i]; /* right condition */
	}

	for (int i = 0; i < nx; i++) {
		for (int j = nm/2; j > 0; j--)
			conc[j-1][i] = conc[j][i]; /* bottom condition */
		for (int j = ny-1-nm/2; j < ny-1; j++)
			conc[j+1][i] = conc[j][i]; /* top condition */
	}
}
