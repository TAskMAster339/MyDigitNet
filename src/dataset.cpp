#include "dataset.h"

// 28x28
std::vector<std::vector<double>> readImages(const std::string& file_name){
    std::ifstream file(file_name, std::ios::binary);
    
    if (!file){
        std::cerr << "Failed to open the file" << std::endl;
        return {};
    }
    char magicNumber[4];
    char numOfImages[4];
    char numOfRows[4];
    char numOfCols[4];

    file.read(magicNumber, 4);
    file.read(numOfImages, 4);
    file.read(numOfRows, 4);
    file.read(numOfCols, 4);

    int numImages = (static_cast<unsigned char>(numOfImages[0]) << 24) | (static_cast<unsigned char>(numOfImages[1]) << 16) | (static_cast<unsigned char>(numOfImages[2]) << 8) | (static_cast<unsigned char>(numOfImages[3]));
    int numRows = (static_cast<unsigned char>(numOfRows[0]) << 24) | (static_cast<unsigned char>(numOfRows[1]) << 16) | (static_cast<unsigned char>(numOfRows[2]) << 8) | (static_cast<unsigned char>(numOfRows[3]));
    int numCols = (static_cast<unsigned char>(numOfCols[0]) << 24) | (static_cast<unsigned char>(numOfCols[1]) << 16) | (static_cast<unsigned char>(numOfCols[2]) << 8) | (static_cast<unsigned char>(numOfCols[3]));

    std::vector<std::vector<double>> result;
    result.reserve(numImages);

    for (int i = 0; i < numImages; ++i){
        std::vector<unsigned char> tmp(numRows*numCols);
        file.read((char*)(tmp.data()), numRows*numCols);
        std::vector<double> img(784);
        for (size_t i = 0; i < tmp.size(); ++i){
            img[i] = (int)tmp[i] / 255.0;
        }
        result.push_back(img);
    }
    file.close();
    return result;
}
std::vector<double> readLabels(const std::string& file_name){
    std::ifstream file(file_name, std::ios::binary);
    
    if (!file){
        std::cerr << "Failed to open the file" << std::endl;
        return {};
    }
    char magicNumber[4];
    char numOfLabels[4];

    file.read(magicNumber, 4);
    file.read(numOfLabels, 4);

    int numLabels = (static_cast<unsigned char>(numOfLabels[0]) << 24) | (static_cast<unsigned char>(numOfLabels[1]) << 16) | (static_cast<unsigned char>(numOfLabels[2]) << 8) | (static_cast<unsigned char>(numOfLabels[3]));

    std::vector<double> result;
    result.reserve(numLabels);

    for (int i = 0; i < numLabels; ++i){
        std::vector<unsigned char> tmp(1);
        file.read((char*)(tmp.data()), 1);
        result.push_back((double)tmp[0]);
    }
    file.close();
    return result;
}
