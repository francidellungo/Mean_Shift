//
// Created by francesca on 05/10/20.
//

#ifndef MEAN_SHIFT_CLUSTER_H
#define MEAN_SHIFT_CLUSTER_H

#include <vector>
#include "Point.h"

class Cluster {

public:
    Cluster(const std::vector<Point> &originalPoints, Point center);

private:
    Point center;
    std::vector<Point> original_points;
    std::vector<Point> copied_points;
};


#endif //MEAN_SHIFT_CLUSTER_H
