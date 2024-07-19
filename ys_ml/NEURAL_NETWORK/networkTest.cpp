#include "network.hpp"
#include "../data_handler/data_handler.hpp"

int main() {
    try {
    // Get data from files and split them.
    data_handler *dh = new data_handler();
    dh->read_csv("../iris_dataset/Iris.csv", ",");
    dh->normalize();
    dh->split_data();
    dh->count_classes();
    dh->dh_set_class_vectors();
    // Apply neural network algorithm.
    std::vector<int> hiddenLayers = {10, 10};
    Network *net = new Network(
        hiddenLayers,
        dh->get_training_data()->at(0)->get_normalized_feature_vector()->size(),
        dh->get_class_counts(),
        0.25);
    net->set_training_data(dh->get_training_data());
    net->set_test_data(dh->get_test_data());
    net->set_validation_data(dh->get_validation_data());
    net->train(15);
    net->validate();
    std::cout << "Test Performance : " << net->test() << std::endl;

    } catch (YSException& exception) {
        std::cout << exception.what() << std::endl;
    }
}