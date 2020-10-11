//
// Created by francesca on 08/10/20.
//

#include <string>
#include <cmath>
#include "MeanShiftUtils.h"
#include "Point.h"

float kernel(const float distance, const float &bandwidth, const std::string type="gaussian"){
//    gaussian kernel
    float res;
    if (type == "gaussian")
        res = expf(- std::exp2(distance) / (2 * std::exp2f(bandwidth) ));
    return res;
}

float computeDistance(const Point &p1,const Point &p2){
    float distance = 0;
    for(int d=0; d<p1.getDim(); d++){
        distance+= exp2f(p1.getValues()[d] - p2.getValues()[d]);
    }
    distance = std::sqrt(distance);

    return distance;
}