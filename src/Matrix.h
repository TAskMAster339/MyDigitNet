#pragma once
#include <iostream>

class Matrix
{
   
    double** matrix;
    int row, col;

    public:
        void Init(int row, int col);
        void Rand();
        static void Multi(const Matrix& m, const double* b, int n, double* c);
        static void Multi_T(const Matrix& m, const double* b, int n, double* c);
        static void SumVector(double* a, const double* b, int n);
        double& operator () (int i, int j);
        friend std::ostream& operator << (std::ostream& os, const Matrix& m);
        friend std::istream& operator >> (std::istream& is, Matrix& m);
};
