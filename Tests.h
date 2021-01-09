//
// Created by francesca on 08/11/20.
//

#ifndef MEAN_SHIFT_TESTS_H
#define MEAN_SHIFT_TESTS_H

#include <chrono>
#include <omp.h>
//#include <string>

#include "Utils.h"
#include "MeanShift.h"

void test(float bandwidth, std::string& points_filename, int iterations, std::string& output_filename, std::vector<Result>& time_res_vec, int verbose=0);

#define N_runs 15

#endif //MEAN_SHIFT_TESTS_H
