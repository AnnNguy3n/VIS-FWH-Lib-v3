#pragma once
#include <string>
#include <fstream>
#include <vector>
#include <sstream>
#include <cstdint>
#include "Logger.cpp"
using namespace std;


template <typename T>
void read_binary_file_1d(T *&array, int &length, string path){
    ifstream file(path, ios::binary);
    if (file.is_open()){
        file.read(reinterpret_cast<char*>(&length), 4);
        array = new T[length];
        size_t t = sizeof(T);
        for (int i=0; i<length; i++){
            file.read(reinterpret_cast<char*>(&array[i]), t);
            if (file.gcount() < t) raise_error("File thieu du lieu", path);
        }
    }
    else raise_error("Khong mo duoc file", path);
}


template <typename T>
void read_binary_file_2d(T *&array, int &rows, int &cols, string path){
    ifstream file(path, ios::binary);
    if (file.is_open()){
        file.read(reinterpret_cast<char*>(&rows), 4);
        file.read(reinterpret_cast<char*>(&cols), 4);
        array = new T[rows*cols];
        size_t t = sizeof(T);
        for (int i=0; i<rows*cols; i++){
            file.read(reinterpret_cast<char*>(&array[i]), t);
            if (file.gcount() < t) raise_error("File thieu du lieu", path);
        }
    }
    else raise_error("Khong mo duoc file", path);
}


void load_checkpoint(int64_t **&current, string folder_save, int64_t &num_opr_per_fml, string lib_abs_path){
    string db_path = folder_save + "/f.db ";
    string folder_input = folder_save + "/InputData ";
    string run_path = lib_abs_path + "/PySources/loadCheckpoint.py ";
    string command = "python " + run_path + db_path + folder_input;
    system(command.c_str());

    ifstream file(folder_save+"/InputData/checkpoint.bin", ios::binary);
    if (file.is_open()){
        file.read(reinterpret_cast<char*>(&num_opr_per_fml), 8);
        current = new int64_t*[3];
        current[0] = new int64_t[num_opr_per_fml-2];
        current[1] = new int64_t[1];
        current[2] = new int64_t[1];
        for (int i=0; i<num_opr_per_fml-2; i++){
            file.read(reinterpret_cast<char*>(&current[0][i]), 8);
        }
        file.read(reinterpret_cast<char*>(&current[1][0]), 8);
        file.read(reinterpret_cast<char*>(&current[2][0]), 8);
        num_opr_per_fml = (num_opr_per_fml - 2) / 2;
    }
    else raise_error("Khong mo duoc file", folder_save+"/InputData/checkpoint.bin");
}


void read_config(string config_path, vector<string> &keys, vector<string> &values){
    ifstream file(config_path);
    string line, temp;
    vector<string> tempLst;
    if (file.is_open()){
        while (getline(file, line)){
            istringstream iss(line);
            tempLst.clear();
            while (getline(iss, temp, '=')){
                temp.erase(0, temp.find_first_not_of(" "));
                temp.erase(temp.find_last_not_of(" ") + 1);
                tempLst.push_back(temp);
            }
            if (tempLst.size() != 2) raise_error("File config sai format", line);
            keys.push_back(tempLst[0]);
            values.push_back(tempLst[1]);
        }
    }
    else raise_error("Khong mo duoc file", config_path);
}
