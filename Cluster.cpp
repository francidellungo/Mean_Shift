//
// Created by francesca on 05/10/20.
//

#include "Cluster.h"


Cluster::Cluster(const std::vector<Point> &originalPoints, Point center)
        : original_points(originalPoints), center(center) {
    //this->center = Point(0,0,0); // initialize center as (0,0,0)
    //this->copied_points = this->original_points;

}
