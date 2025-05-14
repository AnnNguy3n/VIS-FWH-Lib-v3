#pragma once
#include <cstdint>
#include <string>
#include "Logger.cpp"
using namespace std;


bool are_arrays_equal(uint8_t *arr1, int64_t *arr2, int s, int e){
    for (int i=s; i<e; i++){
        if (arr1[i] != arr2[i]) return false;
    }
    return true;
}


bool* get_valid_operator(int **f_struct, int idx, int start){
    bool *valid = new bool[2];
    for (int i=0; i<2; i++){
        if (i < start-2) valid[i] = 0;
        else valid[i] = 1;
    }
    if (idx/2 <= f_struct[0][1]/2) valid[1] = 0;
    return valid;
}


bool* get_valid_operand(uint8_t *formula, int **f_struct, int idx, int start, int cols, int groups){
    bool *valid = new bool[cols];
    int i, gr_idx = -1, pre_op, temp_idx, temp_idx_1;
    for (i=0; i<cols; i++){
        if (i < start) valid[i] = 0;
        else valid[i] = 1;
    }

    for (i=0; i<groups; i++){
        if (f_struct[i][2] + 2*f_struct[i][1] > idx){
            gr_idx = i;
            break;
        }
    }
    if (gr_idx == -1) raise_error("Algorithm ERROR", "gr_idx = -1");

    // Tranh hoan vi nhan chia trong nhom
    pre_op = formula[idx-1];
    if (pre_op >= 2){
        temp_idx = f_struct[gr_idx][2];
        if (pre_op == 2 && idx >= temp_idx+2) {
            for (i=start; i<formula[idx-2]; i++) valid[i] = 0;
        }
        else {
            temp_idx_1 = temp_idx + 2*f_struct[gr_idx][3];
            if (idx > temp_idx_1+2){
                for (i=start; i<formula[idx-2]; i++) valid[i] = 0;
            }

            // Tranh chia lai trong nhom
            for (i=temp_idx; i<temp_idx_1+1; i+=2) valid[formula[i]] = 0;
        }
    }

    // Tranh hoan vi cac nhom
    if (gr_idx){
        int gr_chk_idx = -1;
        for (i=gr_idx-1; i>-1; i--){
            if (f_struct[i][0] == f_struct[gr_idx][0]
                && f_struct[i][1] == f_struct[gr_idx][1]
                && f_struct[i][3] == f_struct[gr_idx][3])
            {
                gr_chk_idx = i;
                break;
            }
        }
        if (gr_chk_idx != -1){
            int idx_ = 0, idx_1, idx_2;
            while (true){
                idx_1 = f_struct[gr_idx][2] + idx_;
                idx_2 = f_struct[gr_chk_idx][2] + idx_;
                if (idx_1 == idx){
                    for (i=start; i<formula[idx_2]; i++) valid[i] = 0;
                    break;
                }
                if (formula[idx_1] != formula[idx_2]) break;
                idx_ += 2;
            }
        }

        // Tranh tru di cac nhom am
        if (f_struct[gr_idx][0] == 1
            && idx+2 == f_struct[gr_idx][2] + 2*f_struct[gr_idx][1])
        {
            int *list_gr_chk = new int[groups];
            bool chk;
            int j, tmp;
            for (i=0; i<groups; i++){
                if (!f_struct[i][0] && f_struct[i][1] == f_struct[gr_idx][1]
                    && f_struct[i][3] == f_struct[gr_idx][3]) list_gr_chk[i] = 1;
                else list_gr_chk[i] = 0;
            }
            for (i=0; i<groups; i++){
                if (!list_gr_chk[i]) continue;
                temp_idx = f_struct[i][2] + 2*f_struct[i][1] - 2;
                temp_idx_1 = f_struct[gr_idx][2] + 2*f_struct[gr_idx][1] - 2;
                chk = true;
                tmp = temp_idx - f_struct[i][2];
                for (j=0; j<tmp; j++){
                    if (formula[f_struct[i][2]+j] != formula[f_struct[gr_idx][2]+j]){
                        chk = false;
                        break;
                    }
                }
                if (chk) valid[formula[temp_idx]] = 0;
            }
            delete[] list_gr_chk;
        }
    }
    return valid;
}
