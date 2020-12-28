//
// Created by francesca on 05/10/20.
//

#ifndef MEAN_SHIFT_UTILS_H
#define MEAN_SHIFT_UTILS_H


#include <vector>
#include <string>

#include "Point.h"


std::vector<Point> CSVReader(const std::string &filename);

//std::vector<Point> readPointsFromCSV(const std::string& fileName);
std::vector<Point> getPointsFromCsv(std::string& fileName);

void savePointsToCsv(std::vector<Point> points, std::string filename);

void read();

void CSVWriter(const std::string &filename, std::vector<Point> points);

std::vector<std::string> getPathTokens(std::string s, std::string delimiter);

#endif //MEAN_SHIFT_UTILS_H
