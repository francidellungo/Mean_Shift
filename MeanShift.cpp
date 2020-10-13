//
// Created by francesca on 06/10/20.
//

#include <vector>
#include <iostream>
#include <functional>   // std::plus
#include <algorithm>
#include "Point.h"
#include "MeanShift.h"
#include "MeanShiftUtils.h"


Point MeanShift::updatePoint(Point &point, const std::vector<Point> &original_points, const std::string kernel_type="gaussian"){
//    this->bandwidth;
    std::vector<float> numerator;
//    initialize numerator
    numerator.reserve(point.getDim());
    for(int i=0; i<point.getDim(); i++){
        numerator.push_back(0);
    }

    float denominator = 0;
    for(auto& orig_point: original_points){
//        numerator
        float distance = computeDistance(point, orig_point);
        float w = computeKernel(distance, this->bandwidth, kernel_type);
//        update position of point to be shifted
        for(int d=0; d<point.getDim(); d++){ //for each dimension d do: num += coordinate_value * w
            numerator[d] += point.getValues()[d] * w;
        }
//        denominator
        denominator+= w;
    }
    std::transform(numerator.begin(), numerator.end(), numerator.begin(),
                   bind2nd(std::divides<float>(), denominator)); // numerator/ denominator
    point.setValues(numerator);
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
            updatePoint(cp, points);
        }
    }
    return copied_points;
}
