#include "knn.hpp"


knn::knn(int val) {
    k = val;
}

knn::~knn() {
    //Dynamically delete memory.
}

// if K ~ N : O(N^2)
// if K is small(for example, 2) : O(~N) 
void knn::find_knearest(data *querry_point) {
    neighbors = new std::vector<data*>(); // create neighbor vector here. It'll be deleted after the prediction.
    double min = std::numeric_limits<double>::max();
    double previous_min = min;
    int index = 0;

    for (int i = 0; i < k; i++) {
        if (i == 0) {
            for (int j = 0; j < training_data->size(); j++) {
                double distance = calculate_distance(querry_point, training_data->at(j));
                training_data->at(j)->set_distance(distance);
                if (distance < min) {
                    min = distance;
                    index = j;
                }
            }
            neighbors->push_back(training_data->at(index));
            previous_min = min;
            min = std::numeric_limits<double>::max();
        }
        else {
            for(int j = 0; j < training_data->size(); j++) {
                double distance = calculate_distance(querry_point, training_data->at(j));
                training_data->at(j)->set_distance(distance);
                // double distance = training_data->at(j)->get_distance();
                if(distance > previous_min && distance < min) {
                    min = distance;
                    index = j;
                }
            }
            neighbors->push_back(training_data->at(index));
            previous_min = min;
            min = std::numeric_limits<double>::max();
        }
    }
}

void knn::set_k(int val) {
    k = val;
}

int knn::predict() {
    std::map<uint8_t, int> class_freq;

    for (int i = 0; i < neighbors->size(); i++) {
        if(class_freq.find(neighbors->at(i)->get_label()) == class_freq.end()) {
            class_freq[neighbors->at(i)->get_label()] = 1;
        }
        else {
            class_freq[neighbors->at(i)->get_label()]++;
        }
    }
    int best = 0;
    int max = 0;
    for (auto kv : class_freq) {
        if (kv.second > max) {
            max = kv.second;
            best = kv.first;
        }
    }
    // Delete neighbor vector
    delete neighbors;

    return best;
}
double knn::calculate_distance(data * querry_point, data * input) {
    double distance = 0.0;
    if (querry_point->get_feature_vector_size() != input->get_feature_vector_size()) {
        throw YSException("Error Vector size mismatch.");
    }
    for (unsigned i = 0; i < querry_point->get_feature_vector_size(); i++) {
        
        distance += pow(querry_point->get_feature_vector()->at(i)
                            - input->get_feature_vector()->at(i), 2);
    }
    distance = sqrt(distance);
    return distance;
}



double knn::validate_performance() {
    double current_performance = 0;
    int count = 0;
    int data_index = 0;

    for (data *query_point : *validation_data) {
        find_knearest(query_point);
        int prediction = predict();
        if(prediction == query_point->get_label()) {
            count++;
        }
        data_index++;
        std::cout << "Current Performance: "
            << (static_cast<double>(count) / static_cast<double>(data_index)) * 100
            << " %" << std::endl;
    }
    current_performance = (static_cast<double>(count) / static_cast<double>(validation_data->size())) * 100;
    std::cout << "Validation Performance: " << current_performance << " %" 
    << " for K = " << k << "." << std::endl;
    return current_performance;
}

double knn::test_performace() {
    double current_performance = 0;
    int count = 0;
    for (data *query_point : *test_data) {
        find_knearest(query_point);
        int prediction = predict();
        if(prediction == query_point->get_label()) {
            count++;
        }
    }
    current_performance = (static_cast<double>(count) / static_cast<double>(test_data->size())) * 100;
    std::cout << "Tested Performance: " << current_performance << " %" << std::endl;
    return current_performance;
}