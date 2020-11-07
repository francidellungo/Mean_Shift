cmake_minimum_required(VERSION 3.15)
project(Mean_Shift)

set(CMAKE_CXX_STANDARD 14)

# added -fopenmp
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -fopenmp")

add_executable(Mean_Shift genPoints.cpp Point.cpp Point.h main.cpp Cluster.cpp Cluster.h Utils.cpp Utils.h MeanShift.cpp MeanShift.h MeanShiftUtils.cpp MeanShiftUtils.h)

find_package(OpenMP REQUIRED)
if(NOT TARGET OpenMP::OpenMP_CXX)
    add_library(OpenMP_TARGET INTERFACE)
    add_library(OpenMP::OpenMP_CXX ALIAS OpenMP_TARGET)
    target_compile_options(OpenMP_TARGET INTERFACE ${OpenMP_CXX_FLAGS})
    find_package(Threads REQUIRED)
    target_link_libraries(OpenMP_TARGET INTERFACE Threads::Threads)
    target_link_libraries(OpenMP_TARGET INTERFACE ${OpenMP_CXX_FLAGS})
endif()

target_link_libraries(Mean_Shift PUBLIC OpenMP::OpenMP_CXX)