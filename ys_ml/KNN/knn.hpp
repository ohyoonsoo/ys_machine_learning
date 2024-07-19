#ifndef __KNN_H
#define __KNN_H

#include <cmath>
#include <limits>
#include <map>
#include <cstdint>
#include "../data_handler/data_handler.hpp"
#include "../common_data/common_data.hpp"

class knn : public common_data{
    private:
        int k;
        std::vector<data *> *neighbors;

    public:
        knn() = default;
        knn(int);
        ~knn();

        void find_knearest(data *querry_point);
        void set_k(int val);

        int predict();
        double calculate_distance(data * querry_point, data * input);
        double validate_performance();
        double test_performace();

};

#endif