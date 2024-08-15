#ifndef ACTIVATION_FUNCTION_H
#define ACTIVATION_FUNCTION_H

#include <iostream>
enum activateFunc {sigmoid = 1, ReLU, thx };

class ActivateFunction{
    private:
        activateFunc actFunc;
    public:
        void set();
        void use(double* value, int n);
        void useDer(double* value, int n);
        double useDer(double value);
};

#endif //ACTIVATION_FUNCTION_H
