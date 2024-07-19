#include "data_handler.hpp"

data_handler::data_handler() {
    data_array = new std::vector<data *>;
    training_data = new std::vector<data *>;
    test_data = new std::vector<data *>;
    validation_data = new std::vector<data *>;
}
data_handler::~data_handler() {
    // // Deleting data_array
    // for (auto data : *data_array) {
    //     delete data;
    // }
    // std::vector<data *>().swap(*data_array);
    // data_array = nullptr;

    // // Deleting training_data
    // for (auto data : *training_data) {
    //     delete data;
    // }
    // std::vector<data *>().swap(*training_data);
    // training_data = nullptr;

    // // Deleting test_data
    // for (auto data : *test_data) {
    //     delete data;
    // }
    // std::vector<data *>().swap(*test_data);
    // test_data = nullptr;

    // // Deleting validation_data
    // for (auto data : *validation_data) {
    //     delete data;
    // }
    // std::vector<data *>().swap(*validation_data);
    // validation_data = nullptr;
}

void data_handler::read_csv(std::string path, std::string delimiter) {
    num_classes = 0;
    std::ifstream data_file(path.c_str());
    std::string line; // holds each line
    // First line only includes the data information. Ignore it.
    std::getline(data_file, line);
    while (std::getline(data_file, line)) {
        if (line.length() == 0) continue;
        data *d = new data;
        d->set_normalized_feature_vector(new std::vector<double>());
        size_t position = 0;
        std::string token; // value in between delimeter;

        // Ignore the string to the first position that meets delimiter. It is Id. We only need double data.
        position = line.find(delimiter);
        line.erase(0, position + delimiter.length());

        // Now we get the inputs what we want which is double type.
        while ((position = line.find(delimiter)) != std::string::npos) {
            token = line.substr(0, position);
            d->append_to_normalized_feature_vector(std::stod(token));
            line.erase(0, position + delimiter.length());
        }
        // Now, only class string is left at line.
        if(classMap.find(line) != classMap.end()) {
            d->set_label(classMap[line]);
        }
        else {
            classMap[line] = num_classes;
            d->set_label(classMap[line]);
            num_classes++;
        }
        data_array->push_back(d);
        feature_vector_size = data_array->at(0)->get_normalized_feature_vector()->size();
    }
}

void data_handler::normalize() {
    std::vector<double> mins, maxs;
    data *d = data_array->at(0);
    // Find max value and min value for each input position.
    for (auto val : *(d->get_normalized_feature_vector())) {
        mins.push_back(val);
        maxs.push_back(val);
    }
    for (int i = 1; i < data_array->size(); i++) {
        d = data_array->at(i);
        for (int j = 0; j < d->get_normalized_feature_vector()->size(); j++) {
            double value = d->get_normalized_feature_vector()->at(j);
            if (mins[j] > value) { mins[j] = value; }
            if (maxs[j] < value) { maxs[j] = value; }
        }
    }
    // Normalize the values
    for (int i = 0; i < data_array->size(); i++) {
        d = data_array->at(i);
        for (int j = 0; j < d->get_normalized_feature_vector()->size(); j++) {
            if(maxs[j] == mins[j]) {
                d->get_normalized_feature_vector()->at(j) = 0.0;
            }
            double value = d->get_normalized_feature_vector()->at(j);
            d->get_normalized_feature_vector()->at(j) = (value - mins[j]) / (maxs[j] - mins[j]);
        }
    }
    std::cout << "Datas are successfully normalized." << std::endl;
}

void data_handler::read_feature_vector(std::string path) {
    uint32_t header[4]; // | MAGIC | NUM IMAGES | ROWSIZE | COLSIZE | (each is 1 byte size)
    unsigned char bytes[4];
    FILE *f = fopen(path.c_str(), "r");
    if (f) {
        for (int i = 0; i < 4; i++) {
            if (fread(bytes, sizeof(bytes), 1, f)) {
                header[i] = convert_to_litte_endian(bytes); // Given data is big endian style. 
            }
        }
        std::cout << "Done getting Input File header." << std::endl;

        int image_size = header[2] * header[3];
        for (int i = 0; i < header[1]; i++) {
            data *d = new data();
            uint8_t element[1];
            for (int j = 0; j < image_size; j++) {
                if (fread(element, sizeof(element), 1, f)) {
                    d->append_to_feature_vector(element[0]);
                }
                else {
                    throw YSException("Error Reading from file.");
                    // std::cout << "Error Reading from file." << std::endl;
                    // exit(1);
                }
            }
            data_array->push_back(d);
        }

        std::cout << "Successfully read and stored " << data_array->size() << " feature vectors." << std::endl;
    }
    else {
        throw YSException("Couldn't find file.");
        // std::cout << "Couldn't find file." << std::endl;
        // exit(1);
    }
}
void data_handler::read_feature_labels(std::string path) {
    uint32_t header[2]; // | MAGIC | NUM ITEMS (each is 1 byte size)
    unsigned char bytes[4];
    FILE *f = fopen(path.c_str(), "r");
    if (f) {
        for (int i = 0; i < 2; i++) {
            if (fread(bytes, sizeof(bytes), 1, f)) {
                header[i] = convert_to_litte_endian(bytes); // Given data is big endian style. 
            }
        }
        std::cout << "Done getting Label File header." << std::endl;

        for (int i = 0; i < header[1]; i++) {
            uint8_t element[1];
            if (fread(element, sizeof(element), 1, f)) {
                data_array->at(i)->set_label(element[0]);
            }
            else {
                throw YSException("Error Reading from file.");
                // std::cout << "Error Reading from file." << std::endl;
                // exit(1);
            }
        }
        
        std::cout << "Successfully read and stored label." << std::endl;
    }
    else {
        throw YSException("Couldn't find file.");
        // std::cout << "Couldn't find file." << std::endl;
        // exit(1);
    }
}
void data_handler::split_data() {
    std::unordered_set<int> used_indexes;
    int train_size = data_array->size() * TRAIN_SET_PERCENT;
    int test_size = data_array->size() * TEST_SET_PERCENT;
    int valid_size = data_array->size() * VALIDATION_PERCENT;

    //Training Data

    int count = 0;
    while (count < train_size) {
        int rand_index = rand() % data_array->size(); // 0 & data_array->size() - 1
        if (used_indexes.find(rand_index) == used_indexes.end()) {
            training_data->push_back(data_array->at(rand_index));
            used_indexes.insert(rand_index);
            count++;
        }
    }

    //Test Data

    count = 0;
    while (count < test_size) {
        int rand_index = rand() % data_array->size(); // 0 & data_array->size() - 1
        if (used_indexes.find(rand_index) == used_indexes.end()) {
            test_data->push_back(data_array->at(rand_index));
            used_indexes.insert(rand_index);
            count++;
        }
    }

    //Validation Data

    count = 0;
    while (count < valid_size) {
        int rand_index = rand() % data_array->size(); // 0 & data_array->size() - 1
        if (used_indexes.find(rand_index) == used_indexes.end()) {
            validation_data->push_back(data_array->at(rand_index));
            used_indexes.insert(rand_index);
            count++;
        }
    }

    std::cout << "Training Data Size: " << training_data->size() << std::endl;
    std::cout << "Test Data Size: " << test_data->size() << std::endl;
    std::cout << "Validation Data Size: " << validation_data->size() << std::endl;
}
void data_handler::count_classes() {
    int count = 0;
    for (unsigned i = 0; i < data_array->size(); i++) {
        if (class_map.find(data_array->at(i)->get_label()) == class_map.end()) {
            class_map[data_array->at(i)->get_label()] = count;
            data_array->at(i)->set_enumerated_label(count);
            count++;
        }
    }

    num_classes = count;
    std::cout << "Successfully Extracted " << num_classes << " Unique Classes." << std::endl;
}

void data_handler::dh_set_class_vectors() {
    for (data* d : *data_array) {
        d->set_class_vector(num_classes);
    }
}

uint32_t data_handler::convert_to_litte_endian(const unsigned char* bytes) {
    return static_cast<uint32_t>((bytes[0] << 24) | (bytes[1] << 16) | (bytes[2] << 8) | (bytes[3])); 
}

std::vector<data *> * data_handler::get_training_data() {
    return training_data;
}

std::vector<data *> * data_handler::get_test_data() {
    return test_data;
}

std::vector<data *> * data_handler::get_validation_data() {
    return validation_data;
}

int data_handler::get_class_counts() {
    return num_classes;
}