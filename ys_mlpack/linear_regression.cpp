#include <iostream>
#include <cmath>
#include "mlpack.hpp"

int main() {
    arma::mat m;
    arma::field<std::string> header;
    arma::mat train_data; // I'll use 90% of datas to train.
    arma::rowvec train_label;
    arma::mat test_data; // I'll use 10% of datats as test date.
    arma::rowvec test_label;
    double error = 0.0;
    m.load(arma::csv_name("./datasets/linear_regression/Admission_Predict.csv", header, arma::csv_opts::trans));
    m.shed_row(0); // remove serial number
    train_data = m.submat(0, 0, m.n_rows - 1, static_cast<size_t>(0.9 * m.n_cols));
    train_label = train_data.row(train_data.n_rows - 1);
    train_data.shed_row(train_data.n_rows - 1);
    test_data = m.submat(0, static_cast<size_t>(0.9 * m.n_cols) + 1, m.n_rows - 1, m.n_cols - 1);
    test_label = test_data.row(train_data.n_rows - 1);
    test_data.shed_row(test_data.n_rows - 1);
    
    //Linear regression train and test
    mlpack::LinearRegression lr;
    lr.Train(train_data, train_label);
    arma::rowvec prediction;
    lr.Predict(test_data, prediction);
    for (size_t i = 0; i < test_label.n_cols - 1; i++) {
        error += abs(test_label(i) - prediction(i));
    }
    std::cout << "Average Error rate of linear regression : " << error / test_label.n_cols << std::endl;
}