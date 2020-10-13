//
// Created by francesca on 08/10/20.
//

#include <string>
#include <cmath>
#include "MeanShiftUtils.h"
#include "Point.h"


float computeDistance(const Point &p1,const Point &p2){
    float distance = 0;
    float sum;
    for(int d=0; d<p1.getDim(); d++){
        sum = p1.getValues()[d] - p2.getValues()[d];
        distance+= sum*sum;
    }
    distance = std::sqrt(distance);

    return distance;
}

float computeKernel(float dist, float bandwidth, const std::string type="gaussian"){
    float res;
//    std::string type="gaussian";
    if (type == "gaussian")
        res = expf(- dist*dist / (2 * bandwidth*bandwidth));
    return res;
}