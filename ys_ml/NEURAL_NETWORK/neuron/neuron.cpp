#include "neuron.hpp"
#include <random>

double generateRandomNumber(double min, double max) {
    double random = static_cast<double>(rand()) / RAND_MAX;
    return min + random * (max - min);
}

Neuron::Neuron(int previousLayerSize, int currentLayerSize) {
    initializeWeights(previousLayerSize);

}
Neuron::~Neuron() {

}
void Neuron::initializeWeights(int previousLayerSize) {
    std::default_random_engine generator;
    std::normal_distribution<double> distribution(0.0, 1.0);
    for (int i = 0; i < previousLayerSize + 1; i++) { // add 1 for the bias
        weights.push_back(generateRandomNumber(-1.0, 1.0));
    }
}