//
// Created by francesca on 05/10/20.
//

#include "Point.h"
#include <iostream>
using namespace std;

Point::Point(float x, float y, float z) : x(x), y(y), z(z) {}

float Point::getX() const {
    return x;
}

void Point::setX(float x) {
    Point::x = x;
}

Point::Point(const Point &obj) {
//    Copy constructor
    Point::x = obj.x;
    Point::y = obj.y;
    Point::z = obj.z;


}

float Point::getY() const {
    return y;
}

void Point::setY(float y) {
    Point::y = y;
}

float Point::getZ() const {
    return z;
}

void Point::setZ(float z) {
    Point::z = z;
}
