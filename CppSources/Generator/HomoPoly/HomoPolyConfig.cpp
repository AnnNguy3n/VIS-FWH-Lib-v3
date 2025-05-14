#pragma once
#include <string>
#include <vector>
#include "../../Utils/Logger.cpp"
#include "../../Utils/WorkWithFile.cpp"
using namespace std;


string get_value_by_key(vector<string> keys, vector<string> values, string target){
    for (int i=0; i<keys.size(); i++){
        if (keys[i] == target) return values[i];
    }
    raise_error("Khong tim thay key", target);
    return "";
}


class HomoPolyConfig {
public:
    string data_path;
    string folder_save;
    string lib_abs_path;

    double interest;
    double valuearg_threshold;
    double eval_threshold;
    // double threshold_delta;

    int num_cycle;
    int storage_size;
    // int eval_index;
    int timeout_in_minutes;
    int eval_method;

    int num_strategy;

    vector<string> filter_field;


    HomoPolyConfig() {}


    HomoPolyConfig(string config_path) {
        vector<string> keys, values;
        string temp;

        read_config(config_path, keys, values);
        data_path = get_value_by_key(keys, values, "data_path");
        folder_save = get_value_by_key(keys, values, "folder_save");
        lib_abs_path = get_value_by_key(keys, values, "lib_abs_path");

        interest = stod(get_value_by_key(keys, values, "interest"));
        valuearg_threshold = stod(get_value_by_key(keys, values, "valuearg_threshold"));
        eval_threshold = stod(get_value_by_key(keys, values, "eval_threshold"));
        // threshold_delta = stod(get_value_by_key(keys, values, "threshold_delta"));

        num_cycle = stoi(get_value_by_key(keys, values, "num_cycle"));
        storage_size = stoi(get_value_by_key(keys, values, "storage_size"));
        // eval_index = stoi(get_value_by_key(keys, values, "eval_index"));
        timeout_in_minutes = stoi(get_value_by_key(keys, values, "timeout_in_minutes"));
        eval_method = stoi(get_value_by_key(keys, values, "eval_method"));

        num_strategy = stoi(get_value_by_key(keys, values, "num_strategy"));

        istringstream iss(get_value_by_key(keys, values, "filter_field"));
        while (getline(iss, temp, ';')) filter_field.push_back(temp);
    }


    ~HomoPolyConfig() {}
};
