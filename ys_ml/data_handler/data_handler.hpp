#ifndef __DATA_HANDLER_H
#define __DATA_HANDLER_H

#include <fstream>
#include <vector>
#include <string>
#include <cstdint>
#include <map>
#include <unordered_set>
#include "data.hpp"
#include "../exception/ysException.hpp"

class data_handler {
    private:
        std::vector<data *> *data_array;
        std::vector<data *> *training_data;
        std::vector<data *> *test_data;
        std::vector<data *> *validation_data;

        int num_classes;
        int feature_vector_size;
        std::map<uint8_t, int> class_map;
        std::map<std::string, int> classMap;                        // For iris dataset

        const double TRAIN_SET_PERCENT = 0.75;
        const double TEST_SET_PERCENT = 0.20;
        const double VALIDATION_PERCENT = 0.05;

    public:
        data_handler();
        ~data_handler();

        void read_csv(std::string path, std::string delimiter);     // For iris dataset
        void normalize(); // For iris dataset, normalize the data. Find max and min value and normalize them between 0, 1.
        void read_feature_vector(std::string path);
        void read_feature_labels(std::string path);
        void split_data();
        void count_classes();
        void dh_set_class_vectors();

        uint32_t convert_to_litte_endian(const unsigned char* bytes);

        std::vector<data *> * get_training_data();
        std::vector<data *> * get_test_data();
        std::vector<data *> * get_validation_data();

        int get_class_counts();

};

#endif
