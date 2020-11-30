//
// Created by francesca on 30/11/20.
//

#include "MeanShift.cuh"
#include <iostream>

__global__ void MeanShift(float* shifted_points, const float* orig_points, const unsigned num_points){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
//    TODO check points dims (to generalize to different number of features)
    float3 newPosition = make_float3(0.0, 0.0, 0.0);
    float totalWeight = 0.0;

    if(idx < num_points){
        float x = shiftedPoints[idx];
        float y = shiftedPoints[idx + numPoints];
        float z = shiftedPoints[idx + 2 * numPoints];
        float3 shiftedPoint = make_float3(x, y, z);

    }

}

__global__ void HelloWorld(){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    printf("Hello! from thread: %d \n", idx);
}

int main(){
    HelloWorld<<<2,5>>>();
    cudaDeviceSynchronize();
    return 0;
}