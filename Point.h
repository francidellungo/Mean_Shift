//
// Created by francesca on 05/10/20.
//

#ifndef MEAN_SHIFT_POINT_H
#define MEAN_SHIFT_POINT_H


#include <vector>


class Point {
public:
    explicit Point(const std::vector<float> &values);

    explicit Point(int dim);

////    define copy constructor to duplicate points
//    Point( const Point &obj);


public:
    const std::vector<float> &getValues() const;

    void setValues(const std::vector<float> &values);

    int getDim() const;

    void setDim(int dim);

    Point &operator+=(const Point &otherPoint);

    Point operator/(float v) const;

    Point &operator/=(float d);

    Point operator*(float d) const;

    Point &operator*=(float d);
private:
    std::vector<float> values;
    int dim;

};


#endif //MEAN_SHIFT_POINT_H
