//
// Created by Timothy Blattner on 12/27/17.
//

#ifndef HIPERC_HTGS_DIFFOPTASK_H
#define HIPERC_HTGS_DIFFOPTASK_H


#include <htgs/api/ITask.hpp>
#include "../data/GridPtrData.h"
#include "../utils/type.h"

class DiffOpTask : public htgs::ITask<GridPtrData, GridPtrData> {
public:
  DiffOpTask(size_t numThreads, fp_t ***conc_old, fp_t ***conc_new, fp_t **mask_lap, fp_t **conc_lap, fp_t D, fp_t dt, int nm, int nbx, int nby);

  void executeTask(std::shared_ptr<GridPtrData> data) override;

  void initialize() override;

  void shutdown() override;

  std::string getName() override;

  DiffOpTask *copy() override;

  fp_t **getMask_lap() const;

  fp_t **getConc_lap() const;

  fp_t getD() const;

  fp_t getDt() const;

  int getNm() const;

  fp_t ***getConc_old() const;

  fp_t ***getConc_new() const;

  int getNbx() const;

  int getNby() const;


private:

  int nm, nbx, nby;

  fp_t **mask_lap;
  fp_t** conc_lap;
  fp_t ***conc_old;
  fp_t ***conc_new;
  fp_t D;
  fp_t dt;

  void compute_convolution(fp_t** conc_old, fp_t** conc_lap, fp_t** mask_lap,
                           int startI, int startJ, const int nx, const int ny, const int nm)
  {

    for (int j = startJ; j < ny; j++) {
      for (int i = startI; i < nx; i++) {
        fp_t value = 0.0;
        for (int mj = -nm/2; mj < nm/2+1; mj++) {
          for (int mi = -nm/2; mi < nm/2+1; mi++) {
            value += mask_lap[mj+nm/2][mi+nm/2] * conc_old[j+mj][i+mi];
          }
        }
        conc_lap[j][i] = value;
      }
    }
  }

  void update_composition(fp_t** conc_old, fp_t** conc_lap, fp_t** conc_new,
                          int startI, int startJ, const int nx, const int ny, const int nm,
                          const fp_t D, const fp_t dt)
  {
    for (int j = startJ; j < ny; j++) {
      for (int i = startI; i < nx; i++) {
        conc_new[j][i] = conc_old[j][i] + dt * D * conc_lap[j][i];
      }
    }
  }




};


#endif //HIPERC_HTGS_DIFFOPTASK_H
