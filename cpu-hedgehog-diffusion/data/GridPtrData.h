//
// Created by tjb3 on 12/27/17.
//

#ifndef HIPERC_HEDGEHOG_GRIDPTRDATA_H
#define HIPERC_HEDGEHOG_GRIDPTRDATA_H


#include "../utils/type.h"

class GridPtrData  {

public:
  GridPtrData(int x, int y, int nx, int nyx);

  int getX() const;

  int getY() const;

  int getNX() const;

  int getNY() const;

private:
  int x;
  int y;
  int nx;
  int ny;

};


#endif //HIPERC_HEDGEHOG_GRIDPTRDATA_H
