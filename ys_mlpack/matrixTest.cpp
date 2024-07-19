#include <iostream>
#include "mlpack.hpp"
#include "csv.h"

int main() {
    const uint32_t columns_num = 4;
    io::CSVReader<columns_num> csv_reader("data.csv");
    std::vector<std::string> categorical_column;
    std::vector<double> values;

    using RowType = std::tuple<double, double, double, double, std::string>;
    RowType row;
    
     
}