#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <cfloat>


__global__ void cuda_set_array_value(double* array, int length, double value){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < length) array[index] = value;
};


__global__ void copy_from_operands(double *dest, double *operands, int *arrCpy, int length, int numCpy){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < numCpy*length){
        int i = index / length;
        int j = index % length;
        dest[index] = operands[arrCpy[i]*length + j];
    }
};


__global__ void update_temp_weight(double *temp_weight_new, double *temp_weight_old, double *operands, int *arrOpr, int length, int numOpr, bool isMul){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < numOpr*length){
        int i = index / length;
        int j = index % length;
        if (isMul)
            temp_weight_new[index] = temp_weight_old[j] * operands[arrOpr[i]*length + j];
        else
            temp_weight_new[index] = temp_weight_old[j] / operands[arrOpr[i]*length + j];
    }
};


__device__ double safe_root(double x, int deg) {
    if (x < 0.0) {
        if (deg % 2 == 0) return 0.0;
        else return -pow(-x, 1.0 / deg);
    }
    return pow(x, 1.0 / deg);
}


__global__ void update_last_weight(double *last_weight, double *curr_weight, double *temp_weight, int length, int numOpr, bool isAdd, int fml_deg, int eval_method){
    // eval_method: 0 - classic, 1 - root
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < numOpr*length){
        int j = index % length;
        double val;
        if (eval_method == 0) val = temp_weight[index];
        else val = safe_root(temp_weight[index], fml_deg);
        if (isAdd) last_weight[index] = curr_weight[j] + val;
        else last_weight[index] = curr_weight[j] - val;
    }
};


__global__ void update_last_weight_through_operands(double *last_weight, double *curr_weight, double *operands, int *arrOpr, int length, int numOpr, bool isAdd){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < numOpr*length){
        int i = index / length;
        int j = index % length;
        if (isAdd) last_weight[index] = curr_weight[j] + operands[arrOpr[i]*length + j];
        else last_weight[index] = curr_weight[j] - operands[arrOpr[i]*length + j];
    }
};


__global__ void replace_nan_and_inf(double *array, int length, int numOpr){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < numOpr*length){
        if (isnan(array[index]) || isinf(array[index]))
            array[index] = -DBL_MAX;
    }
};
