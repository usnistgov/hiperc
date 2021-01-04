//
// Created by tjb3 on 12/27/17.
//

#ifndef HIPERC_HTGS_GRIDPTRDATA_H
#define HIPERC_HTGS_GRIDPTRDATA_H


#include <htgs/api/IData.hpp>
#include "../utils/type.h"

class GridPtrData : public htgs::IData {

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


#endif //HIPERC_HTGS_GRIDPTRDATA_H
