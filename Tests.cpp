//
// Created by francesca on 08/11/20.
//

#include <iostream>
#include "Tests.h"
#include "Utils.h"

void testSequential(const float bandwidth, std::string& points_filename, const int iterations, int verbose){

}

void test(const float bandwidth, std::string& points_filename, const int iterations, std::string& output_filename, std::vector<Result>& time_res_vec, int verbose) {
//    test algorithm execution time with a different number of threads (for openMP parallel version)

    std::vector<Point> points = getPointsFromCsv(points_filename);
    std::vector<std::string> path_tokens = getPathTokens(points_filename, "/");
    std::string csv_filename = path_tokens.back();
    std::cout << "#points: " << points.size() << ", bandwidth: " << bandwidth << ", n iterations: " << iterations << '\n';



//    Sequential version
    float elapsed_time = 0;
    MeanShift MS = MeanShift(bandwidth, iterations);

    //    Initialize variables for time results (seq version)
    Result time_res;
    time_res.opeMP_par = false;
    time_res.num_points = points.size();
    time_res.bandwidth = bandwidth;
    time_res.ms_iterations = iterations;
    time_res.runs = N_runs;

    for(int run_idx=0; run_idx < N_runs; run_idx++){
        std::chrono::high_resolution_clock::time_point start_time = std::chrono::high_resolution_clock::now();
        std::vector<Point> pp = MS.doSeqMeanShift(points);
        std::chrono::high_resolution_clock::time_point end_time = std::chrono::high_resolution_clock::now();

        elapsed_time += std::chrono::duration_cast<std::chrono::duration<float>>(end_time - start_time).count();

        if (run_idx == N_runs - 1){
//            Save final points to file
//            std::string final_out_path = output_filename + "/seq/" + csv_filename;

            savePointsToCsv(pp, output_filename + "/seq/" + csv_filename, verbose);
        }

    }
    float seq_time = elapsed_time / N_runs;
    time_res.time = seq_time;
    time_res_vec.push_back(time_res);

    if(verbose > 0)
        std::cout  << " seq version -> elapsed time: " << seq_time << '\n';

//    Parallel version (OpenMP)
    time_res.opeMP_par = true;
    time_res.omp_static = true; // it does NOT change automatically!
    for(int n_threads=1; n_threads <= omp_get_max_threads(); n_threads++){
        float elapsed_time = 0;

//        mean time on N_runs runs
        for(int run_idx=0; run_idx < N_runs; run_idx++){
            std::chrono::high_resolution_clock::time_point start_time = std::chrono::high_resolution_clock::now();
            std::vector<Point> pp = MS.doMeanShift(points, n_threads);
            std::chrono::high_resolution_clock::time_point end_time = std::chrono::high_resolution_clock::now();

            elapsed_time += std::chrono::duration_cast<std::chrono::duration<float>>(end_time - start_time).count();

            if (run_idx == N_runs - 1){
//            Save final points to file
//                std::string final_out_path = output_filename + "/openmp/" + csv_filename;

                savePointsToCsv(pp, output_filename + "/openmp/" + csv_filename, verbose);
            }
        }
        float openMP_time = elapsed_time / N_runs;
        time_res.n_threads = n_threads;
        time_res.time = openMP_time;
        time_res_vec.push_back(time_res);

        if(verbose > 0){
            std::cout << "#threads: " << n_threads << " -> elapsed time: " << openMP_time << '\n';
            std::cout << "speed-up: " << n_threads << " : " << seq_time/openMP_time << '\n';

        }
    }
}
