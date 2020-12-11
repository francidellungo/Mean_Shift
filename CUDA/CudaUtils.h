//
// Created by francesca on 10/12/20.
//

#ifndef MEAN_SHIFT_CUDAUTILS_H
#define MEAN_SHIFT_CUDAUTILS_H


#include <iostream>
#include <fstream>
#include <sstream>

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
    printf("num points: %d \n", x.size());
    std::vector<float> all_points = x;
    all_points.insert(all_points.end(), y.begin(), y.end());
    all_points.insert(all_points.end(), z.begin(), z.end());
    return all_points;
}



#endif //MEAN_SHIFT_CUDAUTILS_H