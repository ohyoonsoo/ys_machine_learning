#include "data_handler/data_handler.hpp"

int main() {
    try {
    data_handler *dh = new data_handler();
    // dh->read_feature_vector("./mnist_database/train-images.idx3-ubyte"); // For MNIST dataset
    // dh->read_feature_labels("./mnist_database/train-labels.idx1-ubyte"); // For MNIST dataset
    dh->read_csv("./iris_dataset/Iris.csv", ",");
    dh->normalize();
    dh->split_data();
    dh->count_classes();
    } catch (YSException& exception) {
        std::cout << exception.what() << std::endl;
    }
}