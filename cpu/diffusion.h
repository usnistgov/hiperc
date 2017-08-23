/*
	File: diffusion.h
	Role: declaration of diffusion equation function prototypes for CPU benchmarks

	Questions/comments to trevor.keller@nist.gov
	Bugs/requests to https://github.com/usnistgov/phasefield-accelerator-benchmarks
*/

#ifndef _DIFFUSION_H_
#define _DIFFUSION_H_

/* enable easy switching between single- and double-precision */
typedef double fp_t;

/* Mesh handling: implemented in mesh.c */
void make_arrays(fp_t*** conc_old, fp_t*** conc_new, fp_t*** conc_lap, fp_t*** mask_lap,
                 int nx, int ny, int nm);
void free_arrays(fp_t** conc_old, fp_t** conc_new, fp_t** conc_lap, fp_t** mask_lap);
void swap_pointers(fp_t*** conc_old, fp_t*** conc_new);

/* Boundary conditions: implemented in boundaries.c */
void set_boundaries(fp_t bc[2][2]);
void apply_initial_conditions(fp_t** conc_old, int nx, int ny, int nm, fp_t bc[2][2]);
void apply_boundary_conditions(fp_t** conc_old, int nx, int ny, int nm, fp_t bc[2][2]);

/* Discretized mathematical operations: implemented in discretization.c[pp] */
void set_threads(int n);
void set_mask(fp_t dx, fp_t dy, int nm, fp_t** mask_lap);
void compute_convolution(fp_t** conc_old, fp_t** conc_lap, fp_t** mask_lap, int nx, int ny, int nm);
void solve_diffusion_equation(fp_t** conc_old, fp_t** conc_new, fp_t** conc_lap,
                              int nx, int ny, int nm,
                              fp_t D, fp_t dt, fp_t *elapsed);
void analytical_value(fp_t x, fp_t t, fp_t D, fp_t bc[2][2], fp_t* c);
void check_solution(fp_t** conc_new,
                    int nx, int ny, fp_t dx, fp_t dy, int nm,
                    fp_t elapsed, fp_t D, fp_t bc[2][2], fp_t* rss);

/* Output results: implemented in output.c */
void print_progress(const int step, const int steps);
void write_csv(fp_t** conc, int nx, int ny, fp_t dx, fp_t dy, int step);
void write_png(fp_t** conc, int nx, int ny, int step);

/* Time function calls: implemented in timer.c */
void StartTimer();
double GetTimer();

#endif /* _DIFFUSION_H_ */
