//
// Created by Timothy Blattner on 12/27/17.
//

#include "DiffOpTask.h"

DiffOpTask::DiffOpTask(size_t numThreads, fp_t ***conc_old, fp_t ***conc_new, fp_t **mask_lap, fp_t **conc_lap, fp_t D, fp_t dt, int nm, int nbx, int nby)
    : hh::AbstractTask<GridPtrData, GridPtrData>("DiffOpTask", numThreads), conc_old(conc_old), conc_new(conc_new), mask_lap(mask_lap), conc_lap(conc_lap), D(D), dt(dt), nm(nm), nbx(nbx), nby(nby) {}

void DiffOpTask::execute(std::shared_ptr<GridPtrData> data) {

  // compute starting location in block
  int blockIdx = data->getX();
  int blockIdy = data->getY();

  int ghostRegionSize = nm / 2;

  int nx = (data->getNX() * (blockIdx+1)) - (blockIdx == nbx-1 ? ghostRegionSize : 0) ;
  int ny = (data->getNY() * (blockIdy+1)) - (blockIdy == nby-1 ? ghostRegionSize : 0);

  int i = (blockIdx == 0 ? ghostRegionSize : 0) + (blockIdx * data->getNX());
  int j = (blockIdy == 0 ? ghostRegionSize : 0) + (blockIdy * data->getNY());


  // i and j should be locations inside the boundary
  // nx and ny should be the width and height of the block
  compute_convolution(*conc_old, conc_lap, mask_lap, i, j, nx, ny, nm);

  update_composition(*conc_old, conc_lap, *conc_new, i, j, nx, ny, nm, D, dt);
  addResult(data);
}

std::shared_ptr<hh::AbstractTask<GridPtrData, GridPtrData>> DiffOpTask::copy() {
  return std::make_shared<DiffOpTask>(this->numberThreads(), this->getConc_old(), this->getConc_new(), this->getMask_lap(), this->getConc_lap(), this->getD(), this->getDt(), this->getNm(), this->getNbx(), this->getNby());
}

int DiffOpTask::getNbx() const {
  return nbx;
}

int DiffOpTask::getNby() const {
  return nby;
}

fp_t **DiffOpTask::getMask_lap() const {
  return mask_lap;
}

fp_t **DiffOpTask::getConc_lap() const {
  return conc_lap;
}

fp_t DiffOpTask::getD() const {
  return D;
}

fp_t DiffOpTask::getDt() const {
  return dt;
}

int DiffOpTask::getNm() const {
  return nm;
}

fp_t ***DiffOpTask::getConc_old() const {
  return conc_old;
}

fp_t ***DiffOpTask::getConc_new() const {
  return conc_new;
}
