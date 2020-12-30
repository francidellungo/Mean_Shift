//
// Created by francesca on 08/10/20.
//

#ifndef MEAN_SHIFT_MEANSHIFTUTILS_H
#define MEAN_SHIFT_MEANSHIFTUTILS_H

#include <string>
#include "Point.h"

float computeKernel(float dist, float bandwidth, std::string type);

float computeDistance(const Point &p1,const Point &p2);

#endif //MEAN_SHIFT_MEANSHIFTUTILS_H
