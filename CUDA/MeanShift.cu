//
// Created by francesca on 30/11/20.
//

#include "MeanShift.cuh"
#include <iostream>
#include <string>
#include <sstream>


#include <vector>
#include "CudaUtils.h"


__global__ void MeanShift(float* shifted_points, const float* orig_points, const unsigned num_points, float bandwidth){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
//    TODO check points dims (to generalize to different number of features)
    float3 newPosition = make_float3(0.0, 0.0, 0.0);
    float totalWeight = 0.0;

    if(idx < num_points){
        float x = shifted_points[idx];
        float y = shifted_points[idx + num_points];
        float z = shifted_points[idx + 2 * num_points];
        float3 shiftedPoint = make_float3(x, y, z);

        for(int i = 0; i < num_points; i++){
            x = orig_points[i];
            y = orig_points[i + num_points];
            z = orig_points[i + 2 * num_points];
            float3 originalPoint = make_float3(x, y, z);
            printf("x:  %d \n", originalPoint.x);
            printf("x:  %d \n", shiftedPoint.x);

            float3 difference = shiftedPoint - originalPoint;
            printf("x:  %d \n", difference.x);

            float squaredDistance = dot(difference, difference);
            float weight = std::exp((-squaredDistance) / (2 * powf(bandwidth, 2)));
            newPosition += originalPoint * weight;
            totalWeight += weight;
        }
        newPosition /= totalWeight;
        shifted_points[idx] = newPosition.x;
        shifted_points[idx + num_points] = newPosition.y;
        shifted_points[idx + 2 * num_points] = newPosition.z;

    }

}

__global__ void HelloWorld(){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    printf("Hello! from thread: %d \n", idx);
}

int main(){
//    HelloWorld<<<2,5>>>();

    char path[] = __FILE__;
//    stringstream ss;
//    std::string path = std::to_string(array) ;
    std::string p = std::string(path);
    std::cout << "Current path is: " << p << '\n';

//   TODO: split path
    // Project directory path (Mean_shift dir)
    std::string project_dir = p.substr(0, (p.substr(0, p.find_last_of("/")).find_last_of("/")));
    std::cout << "Project dir is: " << project_dir << '\n';

//  Dataset dir path
    std::string dataset_dir_path = project_dir + "/dataset";
    std::cout << "Dataset dir is: " << dataset_dir_path << '\n';

//  Path to csv file in datset dir
    std::string fileName = "dataset/variable_size/1000.csv";
    std::string filen = project_dir + "/" + fileName;
    std::vector<float> inputPoints = getPointsFromCsv(filen);

//    cudaDeviceSynchronize();


    return 0;
}