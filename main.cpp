//
// Created by francesca on 05/10/20.
//

#include <iostream>
#include <vector>
#include "Point.h"
//#include "Cluster.h"

int main_(){
    //Point* p = new Point(2,2,3);
    auto* p1 = new Point(std::vector<float> { 1, 2, 3 });

    std::cout << p1->getX() << '\n';

    std::vector<int> v = {7, 5, 16, 8};
    std::vector<int> aa = v;

    for (int & it : aa)
        std::cout << ' ' << it;

    std::vector<Point> points;
    points.push_back(*p1);


    return 0;
}
