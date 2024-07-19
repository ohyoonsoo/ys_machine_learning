#ifndef __NETWORK_HPP
#define __NETWORK_HPP

#include "../data_handler/data.hpp"
#include "./neuron/neuron.hpp"
#include "./layer/layer.hpp"
#include "../common_data/common_data.hpp"

class Network : public common_data {
    public:
        std::vector<Layer *> layers;
        double learningRate;
        double testPerformance;
        Network(std::vector<int> spec, int, int, double);   // spec is the list of number of neurons for the layers except last layer.
        ~Network();
        std::vector<double> fprop(data *data); // Calculate the output layer with input layer from data
        double activate(std::vector<double>, std::vector<double>); // dot product of wait vector and input vector
        double transfer(double);    // use sigmoid function for this project
        double transferDerivative(double); // used for backprop. Derivatice of sigmoid function.
        void bprop(data *);
        void updateWeights(data *);
        int predict(data *); // return the index of maximum value in the output array.
        void train(int); // num iterations
        double test();
        void validate();

    // private:
    //     InputLayer *inputLayer;
    //     OutputLayer *outputLayer;
    //     std::vector<HiddenLayer *> *hiddenLayers;
    //     double eta; //learning rate

    // public:
    //     Network(std::vector<int> hiddenLayerSpec, int, int);
    //     ~Network();

    //     void fprop(data *data);
    //     void bprop(data *data);
    //     void updateWeight();
    //     void train();
    //     void test();
    //     void validate();
};
#endif