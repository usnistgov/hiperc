/*
	File: diffusion.h
	Role: declaration of diffusion equation function prototypes

	Questions/comments to trevor.keller@nist.gov
	Bugs/requests to https://github.com/tkphd/accelerator-testing
*/

#ifndef _DIFFUSION_H_
#define _DIFFUSION_H_

/* Mesh handling */
void make_arrays(double*** A, double*** B, double*** C, double*** M, double** dataA, double** dataB, double** dataC, double** dataM, int nx, int ny);
void free_arrays(double** A, double** B, double** C, double** M, double* dataA, double* dataB, double* dataC, double* dataM, int nx, int ny);
void swap_pointers(double** dataA, double** dataB, double*** A, double*** B);

/* Boundary conditions */
void set_boundaries(double* c0, double bc[2][2]);
void apply_initial_conditions(double** A, double** B, double** C, int nx, int ny, double c0, double bc[2][2]);
void apply_boundary_conditions(double** A, double** B, int nx, int ny, double bc[2][2]);

/* Discretized mathematical operations */
void set_mask(double** M);
void compute_convolution(double** A, double** B, double** C, double** M, int nx, int ny, int dx, int dy);
void step_in_time(double** A, double** B, double** C, int nx, int ny, double dt, double *elapsed);
void check_solution(double** A, int nx, int ny, int dx, int dy, double elapsed, double bc[2][2], double* sse);

/* Output results */
void write_csv(double** A, int nx, int ny, int dx, int dy, int step);
void write_png(double** A, int nx, int ny, int dx, int dy, int step);

#endif /* _DIFFUSION_H_ */
