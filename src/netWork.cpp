#include "netWork.h"
#include <fstream>
#include <cstdlib>

void NetWork::Init(data_NetWork data){
    actFunc.set();
    srand(time(NULL));
    L = data.L;
    size = new int[L];
    for (int i = 0; i < L; i++){
        size[i] = data.size[i];
    }
    weights = new Matrix[L - 1];
    bios = new double* [L - 1];
    for (int i = 0; i < L - 1; i++){
        weights[i].Init(size[i+1], size[i]);
        bios[i] = new double[size[i+1]];
        weights[i].Rand();
        for (int j = 0; j < size[i+1]; j++){
            bios[i][j] = ((rand() % 50)) * 0.06 / (size[i] + 15);
        }
    }
    neuron_val = new double* [L];
    neurons_err = new double* [L];
    for (int i = 0; i < L; i++){
        neuron_val[i] = new double[size[i]];
        neurons_err[i] = new double[size[i]];
    }
    neuron_bios_val = new double[L - 1];
    for (int i = 0; i < L - 1; i++){
        neuron_bios_val[i] = 1;
    }
}
void NetWork::PrintConfig(){
    std::cout << "**************************************************\n";
    std::cout << "Network has " << L << " layers\nSIZE[]: ";
    for (int i = 0; i < L; i++){
        std::cout << size[i] << " ";
    }
    std::cout << "\n**************************************************\n\n";
}
void NetWork::SetInput(double* values){
    for (int i = 0; i < size[0]; i++){
        neuron_val[0][i] = values[i];
    }
}
double NetWork::ForwardFeed(){
    for (int k = 1; k < L; ++k){
        Matrix::Multi(weights[k-1], neuron_val[k-1], size[k-1], neuron_val[k]);
        Matrix::SumVector(neuron_val[k], bios[k-1], size[k]);
        actFunc.use(neuron_val[k], size[k]);
    }
    int pred = NetWork::SearchMaxIndex(neuron_val[L-1]);
    return pred;
}
std::vector<double> NetWork::MakePredict(double* values){
    for (int i = 0; i < size[0]; i++){
        neuron_val[0][i] = values[i];
    }
    for (int k = 1; k < L; ++k){
        Matrix::Multi(weights[k-1], neuron_val[k-1], size[k-1], neuron_val[k]);
        Matrix::SumVector(neuron_val[k], bios[k-1], size[k]);
        actFunc.use(neuron_val[k], size[k]);
    }
    std::vector<double> result(size[L-1]);
    for (int i = 0; i < size[L-1]; ++i){
        result[i] = neuron_val[L-1][i];
    }
    return result;
}
void NetWork::BackPropagation(double expect){
    for (int i = 0; i < size[L - 1]; i++){
        if (i != int(expect))
            neurons_err[L - 1][i] = -neuron_val[L - 1][i] * actFunc.useDer(neuron_val[L - 1][i]);
        else
            neurons_err[L - 1][i] = (1.0 - neuron_val[L - 1][i]) * actFunc.useDer(neuron_val[L - 1][i]);
    }
    for (int k = L - 2; k > 0; k--){
        Matrix::Multi_T(weights[k], neurons_err[k + 1], size[k + 1], neurons_err[k]);
        for (int j = 0; j < size[k]; j++){
            neurons_err[k][j] *= actFunc.useDer(neuron_val[k][j]);
        }
    }
}
void NetWork::WeightsUpdater(double lr){
    for (int i = 0; i < L - 1; ++i){
        for (int j = 0; j < size[i+1]; ++j){
            for (int k = 0; k < size[i]; ++k){
                weights[i](j, k) += neuron_val[i][k] * neurons_err[i+1][j] * lr;
            }
        }
    }
}
int NetWork::SearchMaxIndex(double* value){
    double max = value[0];
    int prediction = 0;
    double tmp;
    for (int j = 1; j < size[L - 1]; j++){
        tmp = value[j];
        if (tmp > max){
            prediction = j;
            max = tmp;
        }
    }
    return prediction;
}
void NetWork::PrintValues(int L){
    for (int j = 0; j < size[L]; j++){
        std::cout << j << " " << neurons_err[L][j] << std::endl;
    }
}
void NetWork::SaveWeights(){
    std::ofstream fout;
    fout.open("Weights.txt");
    if (!fout.is_open()){
        std::cout << "Error reading the file";
        system("pause");
    }
    for (int i = 0; i < L - 1; ++i){
        fout << weights[i] << " ";
    }
    for (int i = 0; i < L - 1; ++i){
        for (int j = 0; j < size[i + 1]; ++j){
            fout << bios[i][j] << " ";
        }
    }
    std::cout << "Weights saved \n";
    fout.close();
}
void NetWork::ReadWeights(){
    std::ifstream fin;
    fin.open(RESOURCE_PATH "Weights.txt");
    if (!fin.is_open()){
        std::cout << "Error reading the file";
        return;
    }
    for (int i = 0; i < L - 1; ++i){
        fin >> weights[i];
    }
    for (int i = 0; i < L - 1; ++i){
        for (int j = 0; j < size[i + 1]; ++j){
            fin >> bios[i][j];
        }
    }
    std::cout << "OK. Weights has been readed.\n";
    fin.close();
}
