//
// Created by tjb3 on 12/27/17.
//

#include "GridPtrData.h"

GridPtrData::GridPtrData(int x, int y, int nx, int ny)
    : x(x), y(y), nx(nx), ny(ny) {}

int GridPtrData::getX() const {
  return x;
}

int GridPtrData::getY() const {
  return y;
}

int GridPtrData::getNX() const {
  return nx;
}

int GridPtrData::getNY() const {
  return ny;
}

