/*
	File: diffusion.h
	Role: declaration of diffusion equation function prototypes

	Questions/comments to trevor.keller@nist.gov
	Bugs/requests to https://github.com/usnistgov/phasefield-accelerator-benchmarks
*/

#ifndef _DIFFUSION_H_
#define _DIFFUSION_H_

/* enable easy switching between single- and double-precision */
typedef double fp_t;

/* Mesh handling: implemented in mesh.c */
void make_arrays(fp_t*** A, fp_t*** B, fp_t*** C, fp_t*** M,
                 fp_t** dataA, fp_t** dataB, fp_t** dataC, fp_t** dataM,
                 int nx, int ny, int nm);
void free_arrays(fp_t** A, fp_t** B, fp_t** C, fp_t** M,
                 fp_t* dataA, fp_t* dataB, fp_t* dataC, fp_t* dataM);
void swap_pointers(fp_t** dataA, fp_t** dataB, fp_t*** A, fp_t*** B);

/* Boundary conditions: implemented in boundaries.c */
void set_boundaries(fp_t bc[2][2]);
void apply_initial_conditions(fp_t** A, int nx, int ny, int nm, fp_t bc[2][2]);
void apply_boundary_conditions(fp_t** A, int nx, int ny, int nm, fp_t bc[2][2]);

/* Discretized mathematical operations: implemented in discretization.c[pp] */
void set_threads(int n);
void set_mask(fp_t dx, fp_t dy, int nm, fp_t** M);
void compute_convolution(fp_t** A, fp_t** C, fp_t** M, int nx, int ny, int nm);
void solve_diffusion_equation(fp_t** A, fp_t** B, fp_t** C,
                              int nx, int ny, int nm,
                              fp_t D, fp_t dt, fp_t *elapsed);
void analytical_value(fp_t x, fp_t t, fp_t D, fp_t bc[2][2], fp_t* c);
void check_solution(fp_t** A,
                    int nx, int ny, fp_t dx, fp_t dy, int nm,
                    fp_t elapsed, fp_t D, fp_t bc[2][2], fp_t* rss);

/* Output results: implemented in output.c */
void print_progress(const int step, const int steps);
void write_csv(fp_t** A, int nx, int ny, fp_t dx, fp_t dy, int step);
void write_png(fp_t** A, int nx, int ny, int step);

/* Time function calls: implemented in timer.c */
void StartTimer();
double GetTimer();

#endif /* _DIFFUSION_H_ */
