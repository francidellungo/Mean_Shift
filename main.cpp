//
// Created by francesca on 05/10/20.
//

#include <iostream>
#include <vector>
#include <cmath>
#include <omp.h>
#include "Point.h"
#include "MeanShift.h"
#include "Utils.h"
#include "Tests.h"

//#include "Cluster.h"
using namespace std;

//string func(string &a){
//    string b = a;
//    a = "ciaone";
//    cout << a << "\n";
//    cout << b << "\n";
//    b = "cicci";
//    cout << a << "\n";
//    cout << b << "\n";
//    return a;
//};

int main(){

//    std::vector<Point> pointss = readPointsFromCSV("dataset/100.csv");

//    std::cout << "number of pointss: " << pointss.size() << '\n';

//    read();

//    std::string fileName = "../dataset/1000.csv";
//    std::vector<Point> pointss = getPointsFromCsv(fileName);
//    std::cout << "number of pointss: " << pointss.size() << '\n';
//
//    savePointsToCsv(pointss, "../dataset/ms/100.csv");
//
//    std::cout << "initial points:";
////    for (auto& it: pointss) {
////        for(auto& coord: it.getValues())
////            std::cout << ' ' << std::fixed << coord;
////        std::cout << '\n';
////    }
//
////    define points
//    auto* p1 = new Point(vector<float> { 1, 2, 3 });
//    auto* p11 = new Point(vector<float> { 1.01, 2, 3 });
//    auto* p2 = new Point(vector<float> { 4, 5, 6 });
//    auto* p3 = new Point(vector<float> { 0, 1, 2 });
//    auto* p4 = new Point(vector<float> { 10, 11, 12 });
//    auto* p5 = new Point(vector<float> { 10, 11, 13 });
//    auto* p6 = new Point(vector<float> { 10.2, 11, 12.5 });
//    auto* p7 = new Point(vector<float> { 10.2, 11, 12.501 });
//    Point *p = new Point(vector<float>{1,2,3});
//
////    create vector of points (to be done from csv file)
//    std::vector<Point> points;
//    points.push_back(*p1);
////    points.push_back(*p11);
//    points.push_back(*p2);
//    points.push_back(*p3);
////    points.push_back(*p4);
////    points.push_back(*p5);
////    points.push_back(*p6);
////    points.push_back(*p7);
//
//    std::cout << "number of points: " << points.size() << '\n';
//
////    do mean shift
//    int n_threads = omp_get_max_threads();
//    std::cout << "number of threads: " <<  n_threads << '\n';
//
//    MeanShift MS = MeanShift(2, 2);
//    std::vector<Point> pp = MS.doMeanShift(pointss, n_threads);
//
//    savePointsToCsv(pp, "../dataset/ms/100.csv");
//
//    std::cout << "final points:";
//    for (auto& it: pp) {
//        for(auto& coord: it.getValues())
//            std::cout << ' ' << std::fixed << coord;
//        std::cout << '\n';
//    }

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
//
//    cout << func(meal) << "\n";  // Outputs ciaone
//    func(meal);
//    cout << meal << "\n";
//    const float ci = 10;

    char path[] = __FILE__;
    std::string p = std::string(path);
//    std::cout << "Current path is: " << p << '\n';
    bool on_server = false;
    if (p == "main.cpp"){
        on_server = true;
        std::cout << "on server "  << '\n';
    }
    else
        std::cout << "local "  << '\n';

    int n_points = 1000;
    int n_iterations = 10;

    std::string filename;
    std::string dataset_dir;
    std::string output_filename;
    std::string times_output_dir;
    std::string times_output_filename;


    if (on_server){
        //  server paths
        filename = "dataset/3d/" + std::to_string(n_points) +".csv";
        dataset_dir = "dataset/3d/";
        output_filename = "experiments/ms";
        times_output_dir = "experiments/times/";
    } else{
//    local paths
        filename = "../dataset/3d/" + std::to_string(n_points) +".csv";
        dataset_dir = "../dataset/3d/";
        output_filename = "../experiments/ms";
        times_output_dir = "../experiments/times/";

    }


//    Initialize vector to store time results
    std::vector<Result> results_time;

//    Iterate over different dimensions
    int dimensions [6] = {100, 500, 1000, 10000, 100000, 1000000};
//    int dimensions [6] = {100, 500, 1000, 10000, 100000, 500000}; //, 1000000};
//    int dimensions [3] = {100000, 500000, 1000000};

    for(auto dim : dimensions){
        filename = dataset_dir + std::to_string(dim) +".csv";
        times_output_filename = times_output_dir + "seq_openMP_" + std::to_string(dim) + "_" + std::to_string(N_runs)+ ".csv";
        std::cout << "Final times filename: " << times_output_filename << std::endl;
        test(2, filename, n_iterations, output_filename, results_time, 1);
        saveResultsToCsv(results_time, times_output_filename);
    }


//    Save results at the end
//    saveResultsToCsv(results_time, times_output_filename);




    return 0;
}

