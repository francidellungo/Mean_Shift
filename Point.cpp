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
Point::Point(int dim) : dim(dim) {
    vector<float> val;
    val.reserve(dim);
    for(int i=0; i<dim; i++)
        val.push_back(0);
    //set values to zero
    this->setValues(val);
}
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

Point &Point::operator+=(const Point &otherPoint){
    for (int i = 0; i < this->getDim(); ++i) {
        this->values[i] += otherPoint.getValues()[i];
    }
    return *this;
}

Point Point::operator/(float v) const {
    Point point(this->getValues());
    return point /= v;
}

Point &Point::operator/=(float d) {
    for (long i = 0; i < this->getDim(); ++i)
        this->values[i] /= d;
    return *this;
}

Point Point::operator*(const float d) const {
    Point point(this->values);
    return point *= d;
}


Point &Point::operator*=(const float d) {
    for (long i = 0; i < getDim(); ++i)
        this->values[i] *= d;
    return *this;
}