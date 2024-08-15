#ifndef DATASET_H
#define DATASET_H

#include <iostream>
#include <fstream>
#include <vector>

std::vector<std::vector<double>> readImages(const std::string& file_name);
std::vector<double> readLabels(const std::string& file_name);

#endif //DATASET_H
