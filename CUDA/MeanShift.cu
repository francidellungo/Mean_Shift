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


#define N_runs 15 // same in Test.h for sequential and openMP implementations


//tiling version
#define TILE_WIDTH 128
//int tile_widths [7] = {8, 16, 32, 64, 128, 256, 512, 1024};

//#define TILE_WIDTH 64
#define BLOCK_DIM TILE_WIDTH

__global__ void MeanShift_3D(float* shifted_points, const float* orig_points, const unsigned num_points, float bandwidth){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
//    TODO  generalize to different number of features (now only 3d points)
    float3 new_position = make_float3(0.0, 0.0, 0.0);
    float tot_weight = 0.0;

    if(idx < num_points){ // Ensure that threads do not attempt illegal memory access (this can happen because there could be more threads than elements in an array)
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


__global__ void TilingMeanShift(float* shiftedPoints, const float*  originalPoints, const unsigned numPoints, float bandwidth) {
    __shared__ float tile[TILE_WIDTH][3];

    int tx = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tx;

    float3 new_pos = make_float3(0.0, 0.0, 0.0);
    float tot_weight = 0.0;

    for (int tile_i = 0; tile_i < (numPoints - 1) / TILE_WIDTH + 1; ++tile_i) {

//      Load data from global memory into shared memory
        int tile_idx = tile_i * TILE_WIDTH + tx;

        if(tile_idx < numPoints){
            tile[tx][0] = originalPoints[tile_idx];
            tile[tx][1] = originalPoints[tile_idx + numPoints];
            tile[tx][2] = originalPoints[tile_idx + 2 * numPoints];
        }else{
            tile[tx][0] = 0.0;
            tile[tx][1] = 0.0;
            tile[tx][2] = 0.0;
        }

//      Assures that all threads have loaded the data into shared memory (with syncthreads)
        __syncthreads();

        if(idx < numPoints){
//            float x = shiftedPoints[idx];
//            float y = shiftedPoints[idx + numPoints];
//            float z = shiftedPoints[idx + 2 * numPoints];
            float3 shiftedPoint = make_float3(shiftedPoints[idx], shiftedPoints[idx + numPoints], shiftedPoints[idx + 2 * numPoints]);

//      Process data (of current tile)
            for(int i = 0; i < TILE_WIDTH; i++){
                float3 originalPoint = make_float3(tile[i][0], tile[i][1], tile[i][2]); // from shared mem
                float3 difference = shiftedPoint - originalPoint;
                float squared_dist = dot(difference, difference);
//                if(sqrt(squared_dist) <= bandwidth){
                float weight = std::exp((-squared_dist) / (2 * powf(bandwidth, 2)));
                new_pos += originalPoint * weight;
                tot_weight += weight;
//                }
            }
        }
        __syncthreads();
    }
    if(idx < numPoints){
        new_pos /= tot_weight;
        shiftedPoints[idx] = new_pos.x;
        shiftedPoints[idx + numPoints] = new_pos.y;
        shiftedPoints[idx + 2 * numPoints] = new_pos.z;
    }

}


__global__ void TilingMeanShift_v2(float* shiftedPoints, const float*  originalPoints, const unsigned numPoints, float bandwidth, const int pt_dim) {
    __shared__ float tile[TILE_WIDTH][3];

    int tx = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tx;

    float3 new_pos = make_float3(0.0, 0.0, 0.0);
    float tot_weight = 0.0;

    for (int tile_i = 0; tile_i < (numPoints - 1) / TILE_WIDTH + 1; ++tile_i) {

//      Load data from global memory into shared memory
        int tile_idx = tile_i * TILE_WIDTH + tx;

		int index = tile_idx * pt_dim;

        if(tile_idx < numPoints){
            tile[tx][0] = originalPoints[index];
            tile[tx][1] = originalPoints[index + 1];
            tile[tx][2] = originalPoints[index + 2 ];
        }else{
            tile[tx][0] = 0.0;
            tile[tx][1] = 0.0;
            tile[tx][2] = 0.0;
        }

//      Assures that all threads have loaded the data into shared memory (with syncthreads)
        __syncthreads();

        if(idx < numPoints){ // for threads inside bounds do following processing
            float3 shiftedPoint = make_float3(shiftedPoints[idx], shiftedPoints[idx + 1], shiftedPoints[idx + 2]);

//      Process data (of current tile)
            for(int j = 0; j < TILE_WIDTH; j++){
                float3 originalPoint = make_float3(tile[j][0], tile[j][1], tile[j][2]); // from shared mem
                float3 difference = shiftedPoint - originalPoint;
                float squared_dist = dot(difference, difference);
//                if(sqrt(squared_dist) <= bandwidth){
                float weight = std::exp((-squared_dist) / (2 * powf(bandwidth, 2)));
                new_pos += originalPoint * weight;
                tot_weight += weight;
//                }
            }
        }
        __syncthreads();
//       end of processing for tile t_ij
    }
    if(idx < numPoints){
//        Store final value
        new_pos /= tot_weight;
        shiftedPoints[idx] = new_pos.x;
        shiftedPoints[idx + 1] = new_pos.y;
        shiftedPoints[idx + 2] = new_pos.z;
    }

}



//// Tests
Result test(bool tiling, const float bandwidth, std::string &points_filename, const int n_iterations, std::string &output_filename,
                     int verbose, bool save_output) {

// prova versione (x1,y1,z1) dei punti
//    std::vector<float> points = getPointsFromCsv_diffOrder(points_filename);
    std::vector<float> points = getPointsFromCsv(points_filename);
    const int features = 3;
    int num_points = points.size() / features;

    float elapsed_time = 0;
//    Result initialization variables
    Result r{tiling, num_points, bandwidth, n_iterations, N_runs };
    if(verbose > 0)
        printf("Test setting: cuda %s,  #points: %d, iterations: %d, mean on %d runs \n", (tiling ? "tiling" : "naive (NO tiling)"), r.num_points, n_iterations, N_runs);

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
        if (run_idx == 0)
            printf("Grid : %d blocks. Blocks : %d threads.\n", num_blocks.x, num_threads.x);

//        Run mean shift - naive version (no tiling)
        if (not tiling){
            std::chrono::high_resolution_clock::time_point start_time = std::chrono::high_resolution_clock::now();

            for(int i=0; i < n_iterations; i++){

                MeanShift_3D<<<num_blocks, num_threads>>>(shifted_pointer, orig_pointer, num_points, bandwidth);
                cudaDeviceSynchronize();
            }
            std::chrono::high_resolution_clock::time_point end_time = std::chrono::high_resolution_clock::now();

            elapsed_time += std::chrono::duration_cast<std::chrono::duration<float>>(end_time - start_time).count();
//            printf("Time: \n");
//            std::cout << std::chrono::duration_cast<std::chrono::duration<float>>(end_time - start_time).count() << "\n";
//            std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count() << "\n";
        }

//        Run mean shift - shared memory version (tiling)
        else{
            std::chrono::high_resolution_clock::time_point start_time = std::chrono::high_resolution_clock::now();

            for(int i=0; i < n_iterations; i++){

//                TilingMeanShift_v2<<<num_blocks, num_threads>>>(shifted_pointer, orig_pointer, num_points, bandwidth, features);
                TilingMeanShift<<<num_blocks, num_threads>>>(shifted_pointer, orig_pointer, num_points, bandwidth);
                cudaDeviceSynchronize();

            }
            std::chrono::high_resolution_clock::time_point end_time = std::chrono::high_resolution_clock::now();

            elapsed_time += std::chrono::duration_cast<std::chrono::duration<float>>(end_time - start_time).count();
        }


    }

    float cuda_time = elapsed_time / N_runs;
    r.time = cuda_time;

    printf("Cuda elapsed time on %d iterations, with %d points: %f \n", n_iterations, num_points, cuda_time);

    //  Copy final points from device to host
    thrust::host_vector<float> host_shifted_points = shifted_points;
    std::vector<float> outputPoints;
    outputPoints.resize(host_shifted_points.size());
    thrust::copy(host_shifted_points.begin(), host_shifted_points.end(), outputPoints.begin());

//    Save final points
    if (save_output){
//        std::cout << "output filename is: " << output_filename << '\n';
        savePointsToCsv(outputPoints, output_filename, num_points);
    }

    return r;
}

int main(){

//  Paths
    char path[] = __FILE__;
    std::string p = std::string(path);
//    std::cout << "Current path is: " << p << '\n';

    bool on_server = false;
    if (p == "CUDA/MeanShift.cu"){
        on_server = true;
        std::cout << "on server "  << '\n';
    }
    else
        std::cout << "local "  << '\n';

// Project directory path (Mean_shift dir)
    std::string project_dir = p.substr(0, (p.substr(0, p.find_last_of("/")).find_last_of("/")));
    if (on_server)
        project_dir = "";
//    std::cout << "Project dir is: " << project_dir << '\n';

//  Dataset dir path
    std::string dataset_dir_path;
    if (on_server)
        dataset_dir_path = project_dir + "dataset/3d";
    else
        dataset_dir_path = project_dir + "/dataset/3d";


//  Values for mean shift algorithm
    float bandwidth = 2.;
    int num_features = 3;
//    bool tiling_cuda = true;
//    Experiments time
    int n_iterations = 10;

//   Path to save final shifted points
    std::string output_filename = "experiments/ms/cuda/";
//   File to store time results
    std::string times_dir = "experiments/times";

    std::string results_time_filename;

//    Cuda naive and cuda tiling tests (128 tile width)
    bool tiling_experiments[2] = {false, true};
    for (auto tiling_cuda : tiling_experiments){
        if (not tiling_cuda)
            if (on_server)
                results_time_filename = times_dir + "/cuda_naive"  + ".csv";
            else
                results_time_filename = project_dir + "/" + times_dir + "/cuda_naive"  + ".csv";
        else
        if (on_server)
            results_time_filename =  times_dir + "/cuda_tiling_" + std::to_string(TILE_WIDTH)  + ".csv";
        else
            results_time_filename = project_dir + "/" + times_dir + "/cuda_tiling_" + std::to_string(TILE_WIDTH) + ".csv";


        //    Initialize vector to store time results
        std::vector<Result> results_time;

        ////   Iterate over different dataset dimensions
//        int dimensions [4] = {100, 500, 1000, 10000};
        int dimensions [7] = {100, 500, 1000, 10000, 100000, 500000, 1000000};
        //    int dimensions [8] = {100, 1000, 10000, 20000, 50000, 100000, 250000, 500000};

        for (auto d : dimensions){
            printf("Test MS with #points: %d \n", d);
            std::string complete_fileName = dataset_dir_path + "/" + std::to_string(d) + ".csv";
            //        std::cout << "Dataset_dir_path: " << dataset_dir_path << '\n';
            std::vector<float> points = getPointsFromCsv(complete_fileName);
            int num_points = points.size() / num_features;

            //        File to store mean shift results
            std::string cuda_results_dir = project_dir + "/" + output_filename + std::to_string(num_points) + ".csv";
            std::cout << "results_time_filename: " << results_time_filename << '\n';

            //       Experiments
            Result r = test(tiling_cuda, bandwidth, complete_fileName, n_iterations, cuda_results_dir, 1, true);
            results_time.push_back(r);


        }

        //    Save results times

        saveResultsToCsv(results_time, results_time_filename);
    }



////  Iterate over different tile width dimensions for tiling cuda
//    int tile_widths [7] = {16, 32, 64, 128, 256, 512, 1024};
//    tiling_cuda = true;
//    int dataset_dim = 10000; // fixed
//    std::string complete_fileName = dataset_dir_path + "/" + std::to_string(dataset_dim) + ".csv";
//
//    //    Initialize vector to store time results
//    std::vector<Result> tile_w_res_times;
//
////    for (auto tile_w : tile_widths){
//    printf("Test MS tiling with tile width: %d \n", TILE_WIDTH);
//    Result r = test(tiling_cuda, bandwidth, complete_fileName, n_iterations, complete_fileName, 1, false);
//    tile_w_res_times.push_back(r);
////    }


////  To join paths
//    std::vector<std::string> paths {dataset_dir_path, dataset_dir_path};
//    std::string a = joinPath(paths);
//    std::cout << "join result: " << a << '\n';






    return 0;
}