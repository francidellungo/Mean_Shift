//
// Created by francesca on 06/10/20.
//

#ifndef MEAN_SHIFT_MEANSHIFT_H
#define MEAN_SHIFT_MEANSHIFT_H


#include <vector>
#include "Point.h"

class MeanShift{
public:
    MeanShift(float bandwidth, int maxIter) : bandwidth(bandwidth), max_iter(maxIter) {
        if(bandwidth == 0.0){
//            TODO estimate bandwidth
            this->bandwidth = 1.0;
        }
    }
    std::vector<Point> doMeanShift(const std::vector<Point> &points);
    Point updatePoint(Point &point, const std::vector<Point> &original_points, std::string kernel_type);

private:
    float bandwidth;
    int max_iter;

};


#endif //MEAN_SHIFT_MEANSHIFT_H
