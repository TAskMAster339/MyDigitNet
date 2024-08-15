#ifndef NETWORK_H
#define NETWORK_H

#include "activateFunction.h"
#include "Matrix.h"
#include <fstream>
#include <vector>

struct data_NetWork {
    int L;
    int* size;
};
class NetWork{
        int L;
        int* size;
        ActivateFunction actFunc;
        Matrix* weights;
        double** bios;
        double** neuron_val, ** neurons_err;
        double* neuron_bios_val;

    public:
        void Init(data_NetWork data);
        void PrintConfig();
        void SetInput(double* values);
        double ForwardFeed();
        std::vector<double> MakePredict(double* values);
        void BackPropagation(double expect);
        void WeightsUpdater(double lr);

        void SaveWeights(); 
        void ReadWeights();
        int SearchMaxIndex(double* value);
        void PrintValues(int L);
};

#endif //NETWORK_H
