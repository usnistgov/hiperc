#include <iostream>
#include <hedgehog/hedgehog.h>

#include "data/GridPtrData.h"
#include "tasks/DiffOpTask.h"
#include "utils/type.h"
#include "utils/output.h"
#include "utils/mesh.h"
#include "utils/numerics.h"
#define USE_HTGS
void compute_convolution(fp_t** conc_old, fp_t** conc_lap, fp_t** mask_lap,
                         const int nx, const int ny, const int nm)
{
  int i;
  int j;
  for (j = nm/2; j < ny-nm/2; j++) {
    for (i = nm/2; i < nx-nm/2; i++) {
      fp_t value = 0.0;
      for (int mj = -nm/2; mj < nm/2+1; mj++) {
        for (int mi = -nm/2; mi < nm/2+1; mi++) {
          value += mask_lap[mj+nm/2][mi+nm/2] * conc_old[j+mj][i+mi];
        }
      }
      conc_lap[j][i] = value;
    }
  }
//  std::cout << "i = " << i << " j = " << j << std::endl;


}

void update_composition(fp_t** conc_old, fp_t** conc_lap, fp_t** conc_new,
                        const int nx, const int ny, const int nm,
                        const fp_t D, const fp_t dt)
{
  for (int j = nm/2; j < ny-nm/2; j++) {
    for (int i = nm/2; i < nx-nm/2; i++) {
      conc_new[j][i] = conc_old[j][i] + dt * D * conc_lap[j][i];
    }
  }
}


void apply_initial_conditions(fp_t** conc, const int nx, const int ny, const    int nm)
{
  for (int j = 0; j < ny; j++)
    for (int i = 0; i < nx; i++)
      conc[j][i] = 0.0;

  for (int j = 0; j < ny/2; j++)
    for (int i = 0; i < 1+nm/2; i++)
      conc[j][i] = 1.0; /* left half-wall */

  for (int j = ny/2; j < ny; j++)
    for (int i = nx-1-nm/2; i < nx; i++)
      conc[j][i] = 1.0; /* right half-wall */
}

void apply_boundary_conditions(fp_t** conc, const int nx, const int ny, const   int nm)
{
  /* apply fixed boundary values: sequence does not matter */

  for (int j = 0; j < ny/2; j++) {
    for (int i = 0; i < 1+nm/2; i++) {
      conc[j][i] = 1.0; /* left value */
    }
  }

  for (int j = ny/2; j < ny; j++) {
    for (int i = nx-1-nm/2; i < nx; i++) {
      conc[j][i] = 1.0; /* right value */
    }
  }

  /* apply no-flux boundary conditions: inside to out, sequence matters */

  for (int offset = 0; offset < nm/2; offset++) {
    const int ilo = nm/2 - offset;
    const int ihi = nx - 1 - nm/2 + offset;
    for (int j = 0; j < ny; j++) {
      conc[j][ilo-1] = conc[j][ilo]; /* left condition */
      conc[j][ihi+1] = conc[j][ihi]; /* right condition */
    }
  }

  for (int offset = 0; offset < nm/2; offset++) {
    const int jlo = nm/2 - offset;
    const int jhi = ny - 1 - nm/2 + offset;
    for (int i = 0; i < nx; i++) {
      conc[jlo-1][i] = conc[jlo][i]; /* bottom condition */
      conc[jhi+1][i] = conc[jhi][i]; /* top condition */
    }
  }
}

int main(int argc, char *argv[]) {
  auto begin = std::chrono::high_resolution_clock::now();
  // Initial setup of variables
  FILE * output;

  /* declare default mesh size and resolution */
  fp_t **conc_old, **conc_new, **conc_lap, **mask_lap;
  int bx=32, by=32, nx=512, ny=512, nm=3, code=53;


  fp_t dx=0.5, dy=0.5, h;

  /* declare default materials and numerical parameters */
  fp_t D=0.00625, linStab=0.1, dt=1., elapsed=0., rss=0.;
  int step=0, steps=100000, checks=10000;

  param_parser(argc, argv, &bx, &by, &checks, &code, &D, &dx, &dy, &linStab,  &nm, &nx, &ny, &steps);


  int nbx = nx / bx;
  int nby = ny / by;

  h = (dx > dy) ? dy : dx;
  dt = (linStab * h * h) / (4.0 * D);

  /* initialize memory */
  make_arrays(&conc_old, &conc_new, &conc_lap, &mask_lap, nx, ny, nm);
  set_mask(dx, dy, code, mask_lap, nm);

  print_progress(0, steps);

  apply_initial_conditions(conc_old, nx, ny, nm);


  output = fopen("runlog.csv", "w");
  if (output == NULL) {
    printf("Error: unable to %s for output. Check permissions.\n", "runlog. csv");
    exit(-1);
  }

//  fprintf(output, "iter,sim_time,wrss,conv_time,step_time,IO_time,soln_time,  run_time\n");
//  fprintf(output, "%i,%f,%f,%f,%f,%f,%f,%f\n", step, elapsed, rss,
//          watch.conv, watch.step, watch.file, watch.soln, GetTimer());
  fprintf(output, "iter,wrss,\n");
  fprintf(output, "%i,%f\n", step, rss);

  fflush(output);

  write_png(conc_old, nx, ny, 0);

#ifdef USE_HTGS
  size_t nThreadsDiff = 12;

  auto diffOpTask = std::make_shared<DiffOpTask>(nThreadsDiff, &conc_old, &conc_new, mask_lap, conc_lap, D, dt, nm, nbx, nby);

  auto taskGraph = hh::Graph<GridPtrData, GridPtrData>();
  taskGraph.input(diffOpTask);
  taskGraph.output(diffOpTask);

  taskGraph.executeGraph();
;
#endif
  uint64_t totTime2 = 0;

  for (step = 1; step < steps+1; step++)
  {
    auto begin1 = std::chrono::high_resolution_clock::now();
    print_progress(step, steps);



    apply_boundary_conditions(conc_old, nx, ny, nm);
    auto end1 = std::chrono::high_resolution_clock::now();

    totTime2 += std::chrono::duration_cast<std::chrono::microseconds>(end1 - begin1).count();
//    std::cout << "step " << step << " out of " << steps+1 << " Nby = " << nby << " Nbx = " << nbx << std::endl;
#ifndef USE_HTGS
    compute_convolution(conc_old, conc_lap, mask_lap, nx, ny, nm);
    update_composition(conc_old, conc_lap, conc_new, nx, ny, nm, D, dt);
#else
    for (int i = 0; i < nby; i++)
    {
      for (int j = 0; j < nbx; j++)
      {
        // Produce data block-by-block
        taskGraph.pushData(std::make_shared<GridPtrData>(j, i, bx, by));

      }
    }

    int count = 0;

    while (count < nby*nbx)
    {
      taskGraph.getBlockingResult();
      count++;
    }
#endif

    swap_pointers(&conc_old, &conc_new);

//    if ((step % 100) == 0)
//      write_png(conc_old, nx, ny, step);


  }

  write_png(conc_old, nx, ny, step);

#ifdef USE_HTGS
  taskGraph.finishPushingData();
  taskGraph.waitForTermination();
  taskGraph.createDotFile("post-exec.dot", hh::ColorScheme::EXECUTION);
#endif
  auto end = std::chrono::high_resolution_clock::now();

  auto totTime = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();


  std::cout << "Total time = " << totTime / 1000000.0 << " s, time outside of htgs = " << totTime2 / 1000000.0 << " s" << std::endl;

  return 0;
}
