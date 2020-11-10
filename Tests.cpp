//
// Created by francesca on 08/11/20.
//

#include <iostream>
#include "Tests.h"


void testSequential(const float bandwidth, std::string& points_filename, const int iterations, int verbose){

}

void test(const float bandwidth, std::string& points_filename, const int iterations, int verbose) {
//    test algorithm execution time with a different number of threads (for openMP parallel version)
    std::vector<Point> points = getPointsFromCsv(points_filename);
    std::cout << "#points: " << points.size() << ", bandwidth: " << bandwidth << ", n iterations: " << iterations << '\n';

//    sequential version
    float elapsed_time = 0;
    MeanShift MS = MeanShift(bandwidth, iterations);

    for(int run_idx=0; run_idx < N_runs; run_idx++){
        std::chrono::high_resolution_clock::time_point start_time = std::chrono::high_resolution_clock::now();
        std::vector<Point> pp = MS.doSeqMeanShift(points);
        std::chrono::high_resolution_clock::time_point end_time = std::chrono::high_resolution_clock::now();

        elapsed_time += std::chrono::duration_cast<std::chrono::duration<float>>(end_time - start_time).count();
//            TODO save final points to file
    }
    float seq_time = elapsed_time / N_runs;
    if(verbose > 0)
        std::cout  << " seq version -> elapsed time: " << seq_time << '\n';

//    parallel version (OpenMP)
    for(int n_threads=1; n_threads <= omp_get_max_threads(); n_threads++){
        float elapsed_time = 0;

//        mean time on N_runs runs
        for(int run_idx=0; run_idx < N_runs; run_idx++){
            std::chrono::high_resolution_clock::time_point start_time = std::chrono::high_resolution_clock::now();
            std::vector<Point> pp = MS.doMeanShift(points, n_threads);
            std::chrono::high_resolution_clock::time_point end_time = std::chrono::high_resolution_clock::now();

            elapsed_time += std::chrono::duration_cast<std::chrono::duration<float>>(end_time - start_time).count();
//            TODO save final points to file
        }
        elapsed_time = elapsed_time / N_runs;
        if(verbose > 0){
            std::cout << "#threads: " << n_threads << " -> elapsed time: " << elapsed_time << '\n';
            std::cout << "speed-up: " << n_threads << " : " << seq_time/elapsed_time << '\n';

        }
    }
}