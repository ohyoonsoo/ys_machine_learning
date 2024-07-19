#ifndef __NEURON_HPP
#define __NEURON_HPP

#include <cmath>
#include <vector>

class Neuron {
    public:
        double output;
        double delta;
        std::vector<double> weights;
        Neuron(int, int);
        ~Neuron();
        void initializeWeights(int);
    // std::vector<double> weights;
    // double preActivation;
    // double activatedOutput;
    // double outputDerivitive;
    // double error;
    // double alpha; // used in activation funcitons

    // public:
    //     Neuron(int, int);
    //     ~Neuron();

    //     void initializeWeights(int previousLayerSize, int currentLayerSize);
    //     void setError(double);
    //     void setWeight(double, int);
    //     double calculatePreActivation(std::vector<double>);
    //     double activate();
    //     double calculateOutputDerivative();
    //     double sigmoid();
    //     double relu();
    //     double leakyRelu();
    //     double inverseSqrtRelu();
    //     double getOutput();
    //     double getOutputDerivative();
    //     double getError();
    //     std::vector<double> getWeights();
};

#endif