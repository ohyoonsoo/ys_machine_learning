#ifndef __LAYER_HPP
#define __LAYER_HPP

#include "../neuron/neuron.hpp"
#include <vector>

class Layer {
    public:
        int currentLayerSize;
        std::vector<Neuron *> neurons;
        std::vector<double> layerOutputs;
        Layer(int, int);
        ~Layer();
};

#endif