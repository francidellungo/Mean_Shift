//
// Created by francesca on 30/11/20.
//

#include "MeanShift.cuh"
#include <iostream>
#include <string>
#include <sstream>
#include <chrono>


#include <vector>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "CudaUtils.h"
#define N_runs 5 // same in Test.h for sequential and openMP implementations
#define BLOCK_DIM 64


__global__ void MeanShift_3D(float* shifted_points, const float* orig_points, const unsigned num_points, float bandwidth){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
//    TODO  generalize to different number of features (now only 3d points)
    float3 new_position = make_float3(0.0, 0.0, 0.0);
    float tot_weight = 0.0;

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
            float3 difference = shiftedPoint - originalPoint;

            float squared_dist = dot(difference, difference);
            float weight = std::exp((-squared_dist) / (2 * powf(bandwidth, 2)));
            new_position += originalPoint * weight;
            tot_weight += weight;
        }
        new_position /= tot_weight;
        shifted_points[idx] = new_position.x;
        shifted_points[idx + num_points] = new_position.y;
        shifted_points[idx + 2 * num_points] = new_position.z;

    }

}




int main(){

//  Paths
    char path[] = __FILE__;
    std::string p = std::string(path);
    std::cout << "Current path is: " << p << '\n';

// Project directory path (Mean_shift dir)
    std::string project_dir = p.substr(0, (p.substr(0, p.find_last_of("/")).find_last_of("/")));
    std::cout << "Project dir is: " << project_dir << '\n';

//  Dataset dir path
    std::string dataset_dir_path = project_dir + "/dataset";
    std::cout << "Dataset dir is: " << dataset_dir_path << '\n';

//  Path to csv file in datset dir
    std::string fileName = "dataset/3d/1000.csv";
    std::string complete_fileName = project_dir + "/" + fileName;
    std::vector<float> points = getPointsFromCsv(complete_fileName);

//  Values for mean shift algorithm
    float bandwidth = 2.;
    int num_features = 3;
    int num_points = points.size() / num_features;

//    Experiments time
    float elapsed_time = 0;
    int n_blocks = 1;
    int n_threads = 5;
    int n_iterations = 20;

//   Path to save final shifted points
    std::string output_filename = "dataset/ms/cuda/";
    std::string cuda_results_dir = project_dir + "/" + output_filename + std::to_string(num_points) + ".csv";

//  Copy points to device
    thrust::device_vector<float> orig_points = points;
    thrust::device_vector<float> shifted_points = points;


//    Set of experiments
    for(int run_idx = 0; run_idx < N_runs; run_idx ++){
        orig_points = points;
        shifted_points = points;

        float* orig_pointer = thrust::raw_pointer_cast(&orig_points[0]);
        float* shifted_pointer = thrust::raw_pointer_cast(&shifted_points[0]);

//        define number of threads and number of blocks
        dim3 num_blocks = dim3(ceil((float) num_points / BLOCK_DIM));
        dim3 num_threads = dim3(BLOCK_DIM);

//        run mean shift
        std::chrono::high_resolution_clock::time_point start_time = std::chrono::high_resolution_clock::now();

        for(int i=0; i < n_iterations; i++){
            MeanShift_3D<<<num_blocks, num_threads>>>(shifted_pointer, orig_pointer, num_points, bandwidth);
            cudaDeviceSynchronize();
        }
        std::chrono::high_resolution_clock::time_point end_time = std::chrono::high_resolution_clock::now();

        elapsed_time += std::chrono::duration_cast<std::chrono::duration<float>>(end_time - start_time).count();

    }

    float cuda_time = elapsed_time / N_runs;

    printf("Cuda elapsed time on %d iterations, with %d points: %f \n", n_iterations, num_points, cuda_time);

//  Copy final points from device to host
    thrust::host_vector<float> host_shifted_points = shifted_points;
    std::vector<float> outputPoints;
    outputPoints.resize(host_shifted_points.size());
    thrust::copy(host_shifted_points.begin(), host_shifted_points.end(), outputPoints.begin());

    //    Save shifted points to file
//    std::cout << "output filename is: " << cuda_results_dir << '\n';
    savePointsToCsv(outputPoints, cuda_results_dir, num_points);

// TODO assign every point to a cluster
//  see https://github.com/LorenzoAgnolucci/MeanShiftClustering/blob/d729a1b00d52e3b13a9b186f4bf41462298077a6/CUDA/MeanShiftClustering.cu#L189

    return 0;
}