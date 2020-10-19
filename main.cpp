//
// Created by francesca on 05/10/20.
//

#include <iostream>
#include <vector>
#include <cmath>
#include "Point.h"
#include "MeanShift.h"
#include "Utils.h"
//#include "Cluster.h"
using namespace std;

string func(string &a){
    string b = a;
    a = "ciaone";
    cout << a << "\n";
    cout << b << "\n";
    b = "cicci";
    cout << a << "\n";
    cout << b << "\n";
    return a;
};

int main(){

//    std::vector<Point> pointss = readPointsFromCSV("dataset/100.csv");

//    std::cout << "number of pointss: " << pointss.size() << '\n';

//    read();

    std::string fileName = "../dataset/100.csv";
    std::vector<Point> pointss = getPointsFromCsv(fileName);
    std::cout << "number of pointss: " << pointss.size() << '\n';

    savePointsToCsv(pointss, "../dataset/ms/100.csv");

    std::cout << "initial points:";
    for (auto& it: pointss) {
        for(auto& coord: it.getValues())
            std::cout << ' ' << std::fixed << coord;
        std::cout << '\n';
    }

//    define points
    auto* p1 = new Point(vector<float> { 1, 2, 3 });
    auto* p11 = new Point(vector<float> { 1.01, 2, 3 });
    auto* p2 = new Point(vector<float> { 4, 5, 6 });
    auto* p3 = new Point(vector<float> { 0, 1, 2 });
    auto* p4 = new Point(vector<float> { 10, 11, 12 });
    auto* p5 = new Point(vector<float> { 10, 11, 13 });
    auto* p6 = new Point(vector<float> { 10.2, 11, 12.5 });
    auto* p7 = new Point(vector<float> { 10.2, 11, 12.501 });

//    create vector of points (to be done from csv file)
    std::vector<Point> points;
    points.push_back(*p1);
//    points.push_back(*p11);
    points.push_back(*p2);
    points.push_back(*p3);
//    points.push_back(*p4);
//    points.push_back(*p5);
//    points.push_back(*p6);
//    points.push_back(*p7);

    std::cout << "number of points: " << points.size() << '\n';

//    do mean shift
    MeanShift MS = MeanShift(10, 2);
    std::vector<Point> pp = MS.doMeanShift(pointss);

    std::cout << "final points:";
    for (auto& it: pp) {
        for(auto& coord: it.getValues())
            std::cout << ' ' << std::fixed << coord;
        std::cout << '\n';
    }

////    prove varie
//    string food = "Pizza";
//    string &meal = food;
////
//    cout << food << "\n";  // Outputs Pizza
//    cout << meal << "\n";  // Outputs Pizza
//
//    string copia = meal;
//    meal = "cambiato";
//
//    cout << copia << "\n";  // Outputs Pizza
//    cout << meal << "\n";  // Outputs cambiato
//
//    cout << func(meal) << "\n";  // Outputs ciaone
//    cout << meal << "\n";  // Outputs ciaone

//    cout << func(meal) << "\n";  // Outputs ciaone
//    func(meal);
//    cout << meal << "\n";
//    const float ci = 10;


    return 0;
}


//// transform algorithm example
//#include <iostream>     // std::cout
//#include <algorithm>    // std::transform
//#include <vector>       // std::vector
//#include <functional>   // std::plus
//
//int op_increase (int i) { return ++i; }
//float op_increasef (float i) { return ++i; }
//
//int main () {
//    std::vector<int> foo;
//    std::vector<int> bar;
//    std::vector<float> prova;
//
//    // set some values:
//    for (int i=1; i<6; i++){
//        foo.push_back (i*10);                         // foo: 10 20 30 40 50
//        prova.push_back(i);
//    }
//    bar.resize(foo.size());                         // allocate space
//    std::cout << "prova contains:";
//    for (std::vector<float>::iterator it=prova.begin(); it!=prova.end(); ++it)
//        std::cout << ' ' << *it;
//    std::cout << '\n';
//
//    std::transform (foo.begin(), foo.end(), bar.begin(), op_increase);
//    std::transform (prova.begin(), prova.end(), prova.begin(), op_increasef);
//    // bar: 11 21 31 41 51
//
//    // std::plus adds together its two arguments:
//    std::transform (foo.begin(), foo.end(), bar.begin(), foo.begin(), std::plus<int>());
//    // foo: 21 41 61 81 101
//    std::transform(prova.begin(), prova.end(), prova.begin(),
//                   bind2nd(std::plus<float>(), 1.5));
//
//    std::cout << "prova contains:";
//    for (std::vector<float>::iterator it=prova.begin(); it!=prova.end(); ++it)
//        std::cout << ' ' << *it;
//    std::cout << '\n';
//
//    return 0;
//}
