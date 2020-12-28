//
// Created by francesca on 27/12/20.
//

#ifndef MEAN_SHIFT_CUDATESTS_H
#define MEAN_SHIFT_CUDATESTS_H


class CudaTests {
    void test(float bandwidth, std::string& points_filename, int iterations, std::string& output_filename, int verbose);

};


#endif //MEAN_SHIFT_CUDATESTS_H
