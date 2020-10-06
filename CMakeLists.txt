cmake_minimum_required(VERSION 3.15)
project(Mean_Shift)

set(CMAKE_CXX_STANDARD 14)

add_executable(Mean_Shift genPoints.cpp Point.cpp Point.h main.cpp Cluster.cpp Cluster.h Utils.cpp Utils.h)