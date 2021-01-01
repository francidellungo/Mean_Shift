//
// Created by francesca on 10/12/20.
//

#ifndef MEAN_SHIFT_CUDAUTILS_H
#define MEAN_SHIFT_CUDAUTILS_H


#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

#include <cstdarg>

// summation
inline __host__ __device__ float3 operator+(float3 a, float3 b){
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

inline __host__ __device__ void operator+=(float3 &a, float3 b){
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
}

// subtraction
inline __host__ __device__ float3 operator-(float3 a, float3 b){
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

// multiplication by scalar value
inline __host__ __device__ float3 operator*(float3 a, float w){
    return make_float3(a.x * w, a.y * w, a.z * w);
}

// /=
inline __host__ __device__ void operator/=(float3 &a, float w){
    a.x /= w;
    a.y /= w;
    a.z /= w;
}

// dot product between float3
inline __host__ __device__ float dot(float3 a, float3 b){
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

// Read points from CSV file

std::vector<float> getPointsFromCsv(std::string& file_name){
//    Read points from CSV and store into vectors
//    define x,y,z vectors with all values
    std::vector<float> x;
    std::vector<float> y;
    std::vector<float> z;
    std::ifstream data(file_name);
    std::string line;
//    printf("line res: %s", std::getline(data, line));
    while (std::getline(data, line)){
//        printf("line ! \n");
        std::stringstream lineStream(line);
        std::string cell;
        std::vector<float> point;
        while (getline(lineStream, cell, ','))
            point.push_back(stod(cell));
        x.push_back(point[0]);
        y.push_back(point[1]);
        z.push_back(point[2]);
    }
//  all points in one only vector [x1,...xn, y1, ... yn, z1, ... zn]
//    printf("num points: %zu \n", x.size());
    std::vector<float> all_points = x;
    all_points.insert(all_points.end(), y.begin(), y.end());
    all_points.insert(all_points.end(), z.begin(), z.end());
    return all_points;
}


// Write points to CSV file

void savePointsToCsv(const std::vector<float> points, std::string filename, const int num_points){
    std::cout << "Save final points to csv" << std::endl;
    std::ofstream myFile(filename);

    // Send data to the stream
    for(int i=0; i < num_points; i++){
        myFile << points[i] << ",";
        myFile << points[i + num_points] << ",";
        myFile << points[i + 2 * num_points] << "\n";
    }
    // Close the file
    myFile.close();
}

struct Result{
    bool tiling;  // 0: naive cuda, 1: tiling cuda
    int num_points;
    float bandwidth;
    int ms_iterations;
    int runs;
    float time;
};

void saveResultsToCsv(std::vector<Result> results_time, std::string filename){
//    Save time results for experiments to file
    std::ofstream myFile(filename);

    // Send data to the stream
    for(auto experiment : results_time){
        myFile << experiment.tiling << ",";
        myFile << experiment.num_points << ",";
        myFile << experiment.bandwidth << ",";
        myFile << experiment.ms_iterations << ",";
        myFile << experiment.runs << ",";
        myFile << experiment.time << "\n";
    }
    // Close the file
    myFile.close();

}

//void saveToCsv(std::string filename){
//    std::ofstream myFile(filename);
//    myFile << "ciao" << ",";
//    myFile << "nini" << "\n";
//    myFile.close();
//}


std::string joinPath(std::vector<std::string> paths_vec){
    std::string final_path ="";
    std::string sep = "/";
    for(auto wp : paths_vec){
        final_path = final_path.append(wp + sep);
    }
    return final_path;
}

#endif //MEAN_SHIFT_CUDAUTILS_H