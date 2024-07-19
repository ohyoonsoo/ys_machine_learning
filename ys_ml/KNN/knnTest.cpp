#include "knn.hpp"

int main() {
    try {
    // Get data from files and split them.
    data_handler *dh = new data_handler();
    dh->read_feature_vector("../mnist_database/train-images.idx3-ubyte");
    dh->read_feature_labels("../mnist_database/train-labels.idx1-ubyte");
    dh->split_data();
    dh->count_classes();

    // Apply knn algorithm.
    knn *knearest = new knn();
    knearest->set_test_data(dh->get_test_data());
    knearest->set_training_data(dh->get_training_data());
    knearest->set_validation_data(dh->get_validation_data());

    // Test KNN validation for the case that K = 4
    int num_k = 4;
    double performance = 0;
    double best_performance = 0;
    int best_k = 0;
    for (int i = 1; i < num_k; i++) {
        knearest->set_k(i);
        performance = knearest->validate_performance();
        if (performance > best_performance) {
            best_performance = performance;
            best_k = i;
        }
    }

    // Test KNN
    knearest->set_k(best_k);
    knearest->test_performace();


    } catch (YSException& exception) {
        std::cout << exception.what() << std::endl;
    }
}