//
// Created by francesca on 27/12/20.
//

#include "CudaTests.h"


void CudaTests::test(const float bandwidth, std::string &points_filename, const int n_iterations, std::string &output_filename,
                int verbose) {

    std::vector<float> points = getPointsFromCsv(points_filename);
//  Copy points to device
    thrust::device_vector<float> orig_points = points;
    thrust::device_vector<float> shifted_points = points;

//    Set of experiments - naive cuda implementation
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
            TilingMeanShift<<<num_blocks, num_threads>>>(shifted_pointer, orig_pointer, num_points, bandwidth);
//            MeanShift_3D<<<num_blocks, num_threads>>>(shifted_pointer, orig_pointer, num_points, bandwidth);
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
}
