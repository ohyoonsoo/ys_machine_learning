#include "network.hpp"
#include <numeric>

Network::Network(std::vector<int> spec, int inputSize, int numClasses, double learningRate) {
    for (int i = 0; i < spec.size(); i++) {
        if (i == 0) {
            layers.push_back(new Layer(inputSize, spec.at(i))); // first hiden layer
        }
        else {
            layers.push_back(new Layer(layers.at(i - 1)->neurons.size(), spec.at(i))); // other hidden layers
        }
    }
    layers.push_back(new Layer(layers.at(layers.size() - 1)->neurons.size(), numClasses)); // output layer
    this->learningRate = learningRate;
}

Network::~Network() {

}

// Calculate the output layer.
std::vector<double> Network::fprop(data *data) {
    std::vector<double> inputs = *data->get_normalized_feature_vector();
    for (int i = 0; i < layers.size(); i++) {
        Layer *layer = layers.at(i);
        std::vector<double> newInputs;
        for (Neuron *n : layer->neurons) {
            double activation = this->activate(n->weights, inputs);
            n->output = this->transfer(activation); // sigmoid function
            newInputs.push_back(n->output);
        }
        inputs = newInputs;
    }
    return inputs; // output layer outputs
}

double Network::activate(std::vector<double> weights, std::vector<double> inputs) { // dot product (wait vector, input vector)
    double activation = weights.back(); // bias term
    for(int i = 0; i < weights.size() - 1; i++) {
        activation += weights[i] * inputs[i];
    }
    return activation;
}

double Network::transfer(double activation) {
    return 1.0 / (1.0 + exp(0 - activation)); // sigmoid function
}

double Network::transferDerivative(double output) { // used for backprop
    return output * (1 - output); // derivative of sigmoid function is sigmoid(x) * (1 - sigmoid(x))
}

void Network::bprop(data *data) {
    for (int i = layers.size() - 1; i >= 0; i--) {
        Layer *layer = layers.at(i);
        std::vector<double> errors;
        if (i != layers.size() - 1) {
            for (int j = 0; j < layer->neurons.size(); j++) {
                double error = 0.0;
                for (Neuron *n : layers.at(i + 1)->neurons) {
                    error += (n->weights.at(j) * n->delta);
                }
                errors.push_back(error);
            }
        }
        else {
            for (int j = 0; j < layer->neurons.size(); j++) {
                Neuron *n = layer->neurons.at(j);
                errors.push_back((n->output - static_cast<double>(data->get_class_vector()->at(j)))); // the sign is oposite with video.
            }
        }
        for (int j = 0; j < layer->neurons.size(); j++) {
            Neuron *n = layer->neurons.at(j);
            n->delta = errors.at(j) * this->transferDerivative(n->output); // gradient / dericative part of back prop.
        }
    }
}

void Network::updateWeights(data *data) {
    std::vector<double> inputs;     // = *data->get_normalized_feature_vector();
    for (int i = 0; i < layers.size(); i++) {
        if (i == 0) {
            inputs = *data->get_normalized_feature_vector();
        }
        else {
            for (Neuron *n : layers.at(i - 1)->neurons) {
                inputs.push_back(n->output);
            }
        }
        for(Neuron *n : layers.at(i)->neurons) {
            // update weights
            for(int j = 0; j < inputs.size(); j++) {
                n->weights.at(j) -= this->learningRate * n->delta * inputs.at(j); // in video it adds, but i'll subtract
            }
            // update bias
            n->weights.back() -= this->learningRate * n->delta;
        }
        inputs.clear();
    }
}

int Network::predict(data *data) { // return the index of maximum value in the output array.
    std::vector<double> outputs = fprop(data);
    return std::distance(outputs.begin(), std::max_element(outputs.begin(), outputs.end()));
}

void Network::train(int numEpochs) { // num iterations
    for (int i = 0; i < numEpochs; i++) {
        double sumError = 0.0;
        for (data *data: *this->training_data) {
            std::vector<double> outputs = fprop(data);
            std::vector<int> expected = *data->get_class_vector();
            double tempErrorSum = 0.0;
            for (int j = 0; j < outputs.size(); j++) {
                tempErrorSum += pow(static_cast<double>(expected.at(j)) - outputs.at(j), 2);
            }
            sumError += tempErrorSum;
            bprop(data);
            updateWeights(data);
        }
        std::cout << "Iteration : " << i << ", Error = " << sumError << std::endl;
    }
}

double Network::test() {
    double numCorrect = 0.0;
    double count = 0.0;
    for (data * data : *this->test_data) {
        count += 1;
        int index = predict(data);
        if(data->get_class_vector()->at(index) == 1) {
            numCorrect += 1;
        }
    }
    testPerformance = (numCorrect / count);
    return testPerformance;
}

void Network::validate() {
    double numCorrect = 0.0;
    double count = 0.0;
    for (data * data : *this->validation_data) {
        count += 1;
        int index = predict(data);
        if(data->get_class_vector()->at(index) == 1) {
            numCorrect += 1;
        }
    }
    std::cout << "Valiation Performance : " << numCorrect / count << std::endl;
    
}