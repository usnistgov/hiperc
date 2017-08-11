/*
	File: discretization.c
	Role: implementation of discretized mathematical operations

	Questions/comments to trevor.keller@nist.gov
	Bugs/requests to https://github.com/tkphd/accelerator-testing
*/

#include "diffusion.h"

void set_mask(double** M)
{
}

void compute_convolution(double** A, double** B, double** C, double** M, int nx, int ny, int dx, int dy)
{
}

void step_in_time(double** A, double** B, double** C, int nx, int ny, double dt, double* elapsed)
{
}

void check_solution(double** A, int nx, int ny, int dx, int dy, double t, double bc[2][2], double* sse)
{
}
