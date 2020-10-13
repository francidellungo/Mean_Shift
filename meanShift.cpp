//
// Created by francesca on 06/10/20.
//

#include <vector>
#include <iostream>
#include "Point.h"
#include "MeanShift.h"
#include "MeanShiftUtils.h"

Point MeanShift::updatePoint(Point &point, const std::vector<Point> &original_points){
    this->bandwidth;
//    TODO to be finished
    float distance = computeDistance()
//            TODO ...

    return point;
};

std::vector<Point> MeanShift::doMeanShift(const std::vector<Point> &points){
    std::cout << "Mean Shift function" << '\n';
    std::vector<Point> copied_points = points;

    int n_iter = 100; //number of iterations to do

    for(int i=0; i<n_iter; i++){
//        for(auto it = copied_points.begin(); it!=copied_points.end(); ++it){

//      iterate over points
        for(auto& cp: copied_points){
            cp = updatePoint(cp, points);

        }

    }
    return points;
}
