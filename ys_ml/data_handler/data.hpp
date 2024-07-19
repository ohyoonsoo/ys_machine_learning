#ifndef __DATA_H
#define __DATA_H

#include <vector>
#include <cstdint>
#include <iostream>

class data {
    private:
        std::vector<uint8_t> * feature_vector;
        std::vector<double> * normalized_feature_vector;    // For iris dataset
        std::vector<int> * class_vector;                // For iris dataset
        uint8_t label;
        int enum_label;
        double distance;

    public:
        data();
        ~data();

        void set_feature_vector(std::vector<uint8_t> *);
        void append_to_feature_vector(uint8_t);

        void set_normalized_feature_vector(std::vector<double> *); // For iris dataset
        void append_to_normalized_feature_vector(double);          // For iris dataset
        void set_class_vector(int count);      // For iris dataset

        void set_label(uint8_t);
        void set_enumerated_label(int);
        void set_distance(double);

        int get_feature_vector_size();
        uint8_t get_label();
        uint8_t get_enumerated_label();
        double get_distance();

        std::vector<uint8_t> * get_feature_vector();
        std::vector<double> * get_normalized_feature_vector();
        std::vector<int> * get_class_vector();

};

#endif