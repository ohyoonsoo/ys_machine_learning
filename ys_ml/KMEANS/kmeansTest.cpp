#include "kmeans.hpp"

int main() {
    try {
        // Get data from files and split them.
        data_handler *dh = new data_handler();
        dh->read_feature_vector("../mnist_database/train-images.idx3-ubyte");
        dh->read_feature_labels("../mnist_database/train-labels.idx1-ubyte");
        dh->split_data();
        dh->count_classes();

        // Apply knn algorithm.
        double performance = 0.0;
        double best_performance = 0.0;
        int best_k = 1;
        // find the best k
        for (int k = dh->get_class_counts(); k < dh->get_training_data()->size() * 0.1; k++) {
            kmeans *km = new kmeans(k);
            km->set_test_data(dh->get_test_data());
            km->set_training_data(dh->get_training_data());
            km->set_validation_data(dh->get_validation_data());
            km->init_clusters();
            km->train();
            performance = km->validate();
            std::cout << "Current Performance of K = "<< k << " : " << performance << "%." << std::endl;
            if(performance > best_performance){
                best_performance = performance;
                best_k = k;
            }
        }
        // Implement for the test case.
        kmeans *km = new kmeans(best_k);
        km->set_test_data(dh->get_test_data());
        km->set_training_data(dh->get_training_data());
        km->set_validation_data(dh->get_validation_data());
        km->init_clusters();
        km->train();
        performance = km->validate();
        std::cout << "Tested Performance of K = "<< best_k << " : " << performance << "%." << std::endl;

    } catch (YSException& exception) {
        std::cout << exception.what() << std::endl;
    }


}