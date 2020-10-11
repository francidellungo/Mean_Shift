//
// Created by francesca on 05/10/20.
//

#include "Point.h"
#include <iostream>
using namespace std;


//Point::Point(const Point &obj) {
////    Copy constructor
//    Point::x = obj.x;
//    Point::y = obj.y;
//    Point::z = obj.z;
//
//
//}

Point::Point(const vector<float> &values) : values(values) {
    this->dim = this->values.size();
}

const vector<float> &Point::getValues() const {
    return values;
}

void Point::setValues(const vector<float> &values) {
    Point::values = values;
}

int Point::getDim() const {
    return dim;
}

void Point::setDim(int dim) {
    Point::dim = dim;
}
