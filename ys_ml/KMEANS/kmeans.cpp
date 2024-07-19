#include "kmeans.hpp"

    kmeans::kmeans(int k) {
        num_clusters = k;
        clusters = new std::vector<cluster_t *>;
        used_indexes = new std::unordered_set<int>();
    }

    void kmeans::init_clusters() {
        // Pick num_clusters points for the centroid.
        for (int i = 0; i < num_clusters; i++) {
            int index;
            do {
                index = rand() % training_data->size();
            }
            while (used_indexes->find(index) != used_indexes->end());
        
        clusters->push_back(new cluster(training_data->at(index)));
        used_indexes->insert(index);
        }
    }

    void kmeans::init_clusters_for_each_class() {
        std::unordered_set<int> classes_used;
        for (int i = 0; i < training_data->size(); i++) {
            if (classes_used.find(training_data->at(i)->get_label()) == classes_used.end()) {
                clusters->push_back(new cluster_t(training_data->at(i)));
                classes_used.insert(training_data->at(i)->get_label());
                used_indexes->insert(i);
            }
        }
    }

    void kmeans::train() {
        while (used_indexes->size() < training_data->size()) {
            // pick one random data
            int index = rand() % training_data->size();
            double min_dist = std::numeric_limits<double>::max();
            int best_cluster = 0;
            // select the nearest cluseter and add to it.
            for (int j = 0; j < clusters->size(); j++) {
                double current_dist = euclidian_distance(clusters->at(j)->centroid, training_data->at(index));
                if (current_dist < min_dist) {
                    min_dist = current_dist;
                    best_cluster = j;
                }
            }
            clusters->at(best_cluster)->add_to_cluster(training_data->at(index));
            used_indexes->insert(index);
        }
    }

    double kmeans::euclidian_distance(std::vector<double> *centroid, data *point) {
        double dist = 0.0;
        for (int i = 0; i < centroid->size(); i++) {
            dist += pow(centroid->at(i) - static_cast<double>(point->get_feature_vector()->at(i)), 2);
        }
        return sqrt(dist);
    }

    double kmeans::validate() {
        double num_correct = 0.0;
        for (auto query_point : *validation_data) {
            double min_dist = std::numeric_limits<double>::max();
            int best_cluster = 0;
            // select the nearest cluseter and add to it.
            for (int j = 0; j < clusters->size(); j++) {
                double current_dist = euclidian_distance(clusters->at(j)->centroid, query_point);
                if (current_dist < min_dist) {
                    min_dist = current_dist;
                    best_cluster = j;
                }
            }
            if (clusters->at(best_cluster)->most_frequent_class == query_point->get_label()) {
                num_correct += 1;
            }
        }
        return num_correct / static_cast<double>(validation_data->size()) * 100.0;
    }

    double kmeans::test() {
        double num_correct = 0.0;
        for (auto query_point : *test_data) {
            double min_dist = std::numeric_limits<double>::max();
            int best_cluster = 0;
            // select the nearest cluseter and add to it.
            for (int j = 0; j < clusters->size(); j++) {
                double current_dist = euclidian_distance(clusters->at(j)->centroid, query_point);
                if (current_dist < min_dist) {
                    min_dist = current_dist;
                    best_cluster = j;
                }
            }
            if (clusters->at(best_cluster)->most_frequent_class == query_point->get_label()) {
                num_correct += 1;
            }
        }
        return num_correct / static_cast<double>(test_data->size()) * 100.0;
    }