//
// Created by francesca on 05/10/20.
//

#ifndef MEAN_SHIFT_POINT_H
#define MEAN_SHIFT_POINT_H


class Point {
public:
    Point(float x, float y, float z);
//    define copy constructor to duplicate points
    Point( const Point &obj);
    float getX() const;

    void setX(float x);

    float getY() const;

    void setY(float y);

    float getZ() const;

    void setZ(float z);

private:
    float x = 0;
    float y = 0;
    float z = 0;

};


#endif //MEAN_SHIFT_POINT_H
