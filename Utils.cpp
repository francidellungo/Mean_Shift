//
// Created by francesca on 05/10/20.
//

//TODO adjust all

//#include <fstream>
//#include <iostream>
//#include <sstream>
//#include "Utils.h"
//using  namespace std;
//
//std::vector<Point> CSVReader(const std::string &filename) {
//
//    fstream fin; // File pointer
//    fin.open(filename, ios::in); // Open an the csv file
//
//    int rollnum, roll2, count = 0;
//    // Read the Data from the file
//    // as String Vector
//    vector<string> row;
//    string line, word, temp;
//
//    while (fin >> temp) {
//
//        row.clear();
//
//        // read an entire row and
//        // store it in a string variable 'line'
//        getline(fin, line);
//
//        // used for breaking words
//        stringstream s(line);
//
//        // read every column data of a row and
//        // store it in a string variable, 'word'
//        while (getline(s, word, ', ')) {
//
//            // add all the column data
//            // of a row to a vector
//            row.push_back(word);
//        }
//
//        // convert string to integer for comparision
//        roll2 = stoi(row[0]);
//
//        // Compare the roll number
//        if (roll2 == rollnum) {
//
//            // Print the found data
//            count = 1;
//            cout << "Details of Roll " << row[0] << " : \n";
//            cout << "Name: " << row[1] << "\n";
//            cout << "Maths: " << row[2] << "\n";
//            cout << "Physics: " << row[3] << "\n";
//            cout << "Chemistry: " << row[4] << "\n";
//            cout << "Biology: " << row[5] << "\n";
//            break;
//        }
//    }
//
//
//    return std::vector<Point>();
//}
//
//


#include <string>
#include <fstream>
#include <iostream>
#include <cstdlib>
#include <vector>
#include <utility> // std::pair
#include <stdexcept> // std::runtime_error
#include <sstream> // std::stringstream

#include "Point.h"
#include "Utils.h"

std::vector<Point> read_csv(std::string filename){
    // Reads a CSV file into a vector of <string, vector<int>> pairs where
    // each pair represents <column name, column values>

    // Create a vector of <string, int vector> pairs to store the result
    std::vector<Point> result;

    // Create an input filestream
    std::ifstream myFile(filename);

    // Make sure the file is open
    if(!myFile.is_open()) throw std::runtime_error("Could not open file");

    // Helper vars
    std::string line, colname;
    int val;

//    // Read the column names
//    if(myFile.good())
//    {
//        // Extract the first line in the file
//        std::getline(myFile, line);
//
//        // Create a stringstream from line
//        std::stringstream ss(line);
//
//        // Extract each column name
//        while(std::getline(ss, colname, ',')){
//
//            // Initialize and add <colname, int vector> pairs to result
//            result.push_back({colname, std::vector<int> {}});
//        }
//    }

    // Read data, line by line
    while(std::getline(myFile, line))
    {
        // Create a stringstream of the current line
        std::stringstream ss(line);

        // Keep track of the current column index
        int colIdx = 0;

        // Extract each integer
        while(ss >> val){

            // Add the current integer to the 'colIdx' column's values vector
//            result.push_back(val);

            // If the next token is a comma, ignore it and move on
            if(ss.peek() == ',') ss.ignore();

            // Increment the column index
            colIdx++;
        }
    }

    // Close file
    myFile.close();

    return result;
}

std::vector<Point> getPointsFromCsv(std::string& fileName)
{
    std::vector<Point> points;
    std::ifstream data(fileName);
    std::string line;
    while (getline(data, line)) {
        std::stringstream lineStream(line);
        std::string cell;
        std::vector<float> point;
        while (std::getline(lineStream, cell, ','))
            point.push_back(stod(cell));
        points.emplace_back(Point(point));
    }

    return points;

}

void savePointsToCsv(const std::vector<Point> points, std::string filename, int verbose){
    if (verbose > 0)
        std::cout << "savePointsToCsv" << std::endl;

    std::ofstream myFile(filename);

    // Send data to the stream
    for(auto & point : points)
    {
        for(int coord_idx = 0; coord_idx < point.getValues().size(); coord_idx++)
        {
            myFile << point.getValues()[coord_idx];
            if(coord_idx != point.getValues().size() - 1)
                myFile << ",";
        }
        myFile << "\n";
    }

    // Close the file
    myFile.close();
}

std::vector<std::string> getPathTokens(std::string s, const std::string delimiter){
    std::vector<std::string> tokens;
    size_t pos = 0;
    std::string token;
    while ((pos = s.find(delimiter)) != std::string::npos) {
        token = s.substr(0, pos);
        tokens.push_back(token);
        s.erase(0, pos + delimiter.length());
    }
    tokens.push_back(s);

//    for(auto & t : tokens){
//        std::cout << t << std::endl;
//    }
    return tokens;
}



void saveResultsToCsv(std::vector<Result> results_time, std::string filename){
//    Save time results for experiments to file
    std::ofstream myFile(filename);

    // Send data to the stream
    for(auto experiment : results_time){
        myFile << experiment.opeMP_par << ",";
        myFile << experiment.omp_static << ",";
        myFile << experiment.n_threads << ",";
        myFile << experiment.num_points << ",";
        myFile << experiment.bandwidth << ",";
        myFile << experiment.ms_iterations << ",";
        myFile << experiment.runs << ",";
        myFile << experiment.time << "\n";
    }
    // Close the file
    myFile.close();

}
