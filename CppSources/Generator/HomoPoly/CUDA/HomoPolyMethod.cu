#pragma once
#include "HomoPolyKernel.cu"
#include "../HomoPolyConfig.cpp"
#include "../../../Utils/WorkWithFile.cpp"
#include "../../../Utils/SuppFunc.cpp"
#include <chrono>


const int __SAVE_FREQ__ = 300;
const int __MAX_FTOS__ = 1'000'000;


class Generator {
public:
    HomoPolyConfig config;
    int64_t **current, num_opr_per_fml;

    int *INDEX, *SYMBOL, *BOOL_ARG, index_length, rows, cols;
    int num_symbol_unique;
    double *PROFIT, *OPERAND;

    double *temp_weight_storage;
    int count_temp_storage;
    uint8_t **temp_formula_storage;

    chrono::high_resolution_clock::time_point time_start;

    //
    int fml_shape, groups, num_per_grp;

    //
    double **fields_to_save;
    uint8_t **fmls_to_save;
    int64_t **fmls_idx;
    int *count_fml_save;

    //
    Generator(string config_path);
    ~Generator();

    bool fill_formula(uint8_t *formula, int **f_struct, int idx,
        double *temp_0, int temp_op, double *temp_1,
        int mode, bool add_sub, bool mul_div, int fml_deg
    );

    virtual bool compute_result(bool force_save);
    void run();
    bool save_result(bool force_save, double *h_final, int *is_save);

    chrono::high_resolution_clock::time_point start_t_temp;
};


Generator::Generator(string config_path){
    config = HomoPolyConfig(config_path);

    // Extract data
    int *_INDEX, *_SYMBOL, *_BOOL_ARG;
    double *_PROFIT, *_OPERAND;

    string command = "python " + config.lib_abs_path + "/PySources/base.py "
        + config.data_path + " " + to_string(config.interest) + " "
        + to_string(config.valuearg_threshold) + " " + config.folder_save + "/InputData";
    system(command.c_str());

    // Load data
    read_binary_file_1d(_INDEX, index_length, config.folder_save + "/InputData/INDEX.bin");
    read_binary_file_1d(_SYMBOL, rows, config.folder_save + "/InputData/SYMBOL.bin");
    read_binary_file_1d(_BOOL_ARG, rows, config.folder_save + "/InputData/BOOL_ARG.bin");
    read_binary_file_1d(_PROFIT, rows, config.folder_save + "/InputData/PROFIT.bin");
    read_binary_file_2d(_OPERAND, cols, rows, config.folder_save + "/InputData/OPERAND.bin");

    cudaMemcpyToSymbol(C_INDEX, _INDEX, index_length * 4, 0);

    cudaMalloc((void**)&INDEX, 4*index_length);
    cudaMalloc((void**)&SYMBOL, 4*rows);
    cudaMalloc((void**)&BOOL_ARG, 4*rows);
    cudaMalloc((void**)&PROFIT, 8*rows);
    cudaMalloc((void**)&OPERAND, 8*rows*cols);
    cudaMemcpy(INDEX, _INDEX, 4*index_length, cudaMemcpyHostToDevice);
    cudaMemcpy(SYMBOL, _SYMBOL, 4*rows, cudaMemcpyHostToDevice);
    cudaMemcpy(BOOL_ARG, _BOOL_ARG, 4*rows, cudaMemcpyHostToDevice);
    cudaMemcpy(PROFIT, _PROFIT, 8*rows, cudaMemcpyHostToDevice);
    cudaMemcpy(OPERAND, _OPERAND, 8*rows*cols, cudaMemcpyHostToDevice);

    // Tính num_symbol_unique
    int max_symbol = 0;
    for (int i = 0; i < rows; ++i) {
        if (_SYMBOL[i] > max_symbol) max_symbol = _SYMBOL[i];
    }
    num_symbol_unique = max_symbol + 1;

    delete[] _INDEX;
    delete[] _SYMBOL;
    delete[] _BOOL_ARG;
    delete[] _PROFIT;
    delete[] _OPERAND;

    // Load checkpoint
    load_checkpoint(current, config.folder_save, num_opr_per_fml, config.lib_abs_path);

    // Init storage
    cudaMalloc((void**)&temp_weight_storage, 8*(config.storage_size+cols)*rows);
    temp_formula_storage = new uint8_t*[config.storage_size+cols];
    count_temp_storage = 0;

    //
    time_start = chrono::high_resolution_clock::now();
    start_t_temp = chrono::high_resolution_clock::now();

    //
    fields_to_save = new double*[config.num_cycle];
    fmls_to_save = new uint8_t*[config.num_cycle];
    fmls_idx = new int64_t*[config.num_cycle];
    count_fml_save = new int[config.num_cycle];
    for (int c_i = 0; c_i < config.num_cycle; c_i++){
        fields_to_save[c_i] = new double[(__MAX_FTOS__ + config.storage_size + cols) * config.filter_field.size()];
        fmls_idx[c_i] = new int64_t[__MAX_FTOS__ + config.storage_size + cols];
    }
}


Generator::~Generator(){
    cudaFree(INDEX);
    cudaFree(SYMBOL);
    cudaFree(BOOL_ARG);
    cudaFree(PROFIT);
    cudaFree(OPERAND);

    for (int i=0; i<3; i++) delete[] current[i];
    delete[] current;

    cudaFree(temp_weight_storage);
    delete[] temp_formula_storage;

    for (int c_i = 0; c_i < config.num_cycle; c_i++){
        delete[] fields_to_save[c_i];
        delete[] fmls_idx[c_i];
    }
    delete[] fields_to_save;
    delete[] fmls_to_save;
    delete[] fmls_idx;
    delete[] count_fml_save;

    chrono::high_resolution_clock::time_point now = chrono::high_resolution_clock::now();
    chrono::duration<long long> duration = chrono::duration_cast<chrono::seconds>(now - time_start);
    cout << FG_GREEN << "Thoi gian Worker chay: " << duration.count() << " seconds.\n" << RESET_COLOR;
}


bool Generator::fill_formula(
    uint8_t *formula,
    int **f_struct,
    int idx,
    double *temp_0,
    int temp_op,
    double *temp_1,
    int mode,
    bool add_sub,
    bool mul_div,
    int fml_deg
) {
    if (!mode) /*Sinh dau cong tru*/ {
        int gr_idx = 2147483647, start = 0, i, k;
        bool new_add_sub;
        uint8_t *new_formula = new uint8_t[fml_shape];
        int **new_f_struct = new int*[groups];
        for (i=0; i<groups; i++) new_f_struct[i] = new int[4];

        // Xac dinh nhom
        for (i=0; i<groups; i++){
            if (f_struct[i][2]-1 == idx){
                gr_idx = i;
                break;
            }
        }

        // Xac dinh chi so bat dau
        if (are_arrays_equal(formula, current[0], 0, idx)) start = current[0][idx];

        // Loop
        for (k=start; k<2; k++){
            memcpy(new_formula, formula, fml_shape);
            for (i=0; i<groups; i++) memcpy(new_f_struct[i], f_struct[i], 16);
            new_formula[idx] = k;
            new_f_struct[gr_idx][0] = k;
            if (k == 1){
                new_add_sub = true;
                for (i=gr_idx+1; i<groups; i++){
                    new_formula[new_f_struct[i][2]-1] = 1;
                    new_f_struct[i][0] = 1;
                }
            } else new_add_sub = false;

            if (fill_formula(new_formula, new_f_struct, idx+1,
                            temp_0, temp_op, temp_1, 1, new_add_sub, mul_div, fml_deg)) return true;
        }

        // Giai phong bo nho
        delete[] new_formula;
        for (i=0; i<groups; i++) delete[] new_f_struct[i];
        delete[] new_f_struct;
    }
    else if (mode == 2) /*Sinh dau nhan chia*/ {
        int start = 2, i, j, k;
        bool new_mul_div;
        uint8_t *new_formula = new uint8_t[fml_shape];
        int **new_f_struct = new int*[groups];
        for (i=0; i<groups; i++) new_f_struct[i] = new int[4];

        // Xac dinh chi so bat dau
        if (are_arrays_equal(formula, current[0], 0, idx)) start = current[0][idx];
        if (!start) start = 2;

        // Loop
        bool *valid_operator = get_valid_operator(f_struct, idx, start);
        for (k=0; k<2; k++){
            if (!valid_operator[k]) continue;
            memcpy(new_formula, formula, fml_shape);
            for (i=0; i<groups; i++) memcpy(new_f_struct[i], f_struct[i], 16);
            new_formula[idx] = k + 2;
            int new_fml_deg = fml_deg;
            if (k == 1){
                new_fml_deg -= 1;
                new_mul_div = true;
                for (i=idx+2; i<2*new_f_struct[0][1]-1; i+=2){
                    new_formula[i] = 3;
                    new_fml_deg -= 1;
                }
                for (i=1; i<groups; i++){
                    for (j=0; j<new_f_struct[0][1]-1; j++)
                        new_formula[new_f_struct[i][2] + 2*j + 1] = new_formula[2 + 2*j];
                }
            } else {
                new_fml_deg += 1;
                new_mul_div = false;
                for (i=0; i<groups; i++) new_f_struct[i][3] += 1;
                if (idx == 2*new_f_struct[0][1]-2){
                    new_mul_div = true;
                    for (i=1; i<groups; i++){
                        for (j=0; j<new_f_struct[0][1]-1; j++)
                            new_formula[new_f_struct[i][2] + 2*j + 1] = new_formula[2 + 2*j];
                    }
                }
            }

            if (fill_formula(new_formula, new_f_struct, idx+1,
                            temp_0, temp_op, temp_1, 1, add_sub, new_mul_div, new_fml_deg)) return true;
        }

        // Giai phong bo nho
        delete[] valid_operator;
        delete[] new_formula;
        for (i=0; i<groups; i++) delete[] new_f_struct[i];
        delete[] new_f_struct;
    }
    else if (mode == 1){
        int start = 0, i, count = 0;

        // Xac dinh chi so bat dau
        if (are_arrays_equal(formula, current[0], 0, idx)) start = current[0][idx];

        // Xac dinh cac toan hang hop le va dem
        bool *valid_operand = get_valid_operand(formula, f_struct, idx, start, cols, groups);
        for (i=0; i<cols; i++){
            if (valid_operand[i]) count++;
        }

        if (count){
            int temp_op_new, new_idx, new_mode, k = 0;
            bool chk = false, temp_0_change;
            double *new_temp_0 = nullptr, *new_temp_1 = nullptr;
            uint8_t **new_formula = new uint8_t*[count];
            int *valid = new int[count];
            int *d_valid;
            int num_block = count*rows/256 + 1;
            cudaMalloc((void**)&d_valid, 4*count);

            for (i=0; i<cols; i++){
                if (!valid_operand[i]) continue;
                new_formula[k] = new uint8_t[fml_shape];
                memcpy(new_formula[k], formula, fml_shape);
                new_formula[k][idx] = i;
                valid[k] = i;
                k++;
            }
            cudaMemcpy(d_valid, valid, 4*count, cudaMemcpyHostToDevice);

            if (formula[idx-1] < 2){
                temp_op_new = formula[idx-1];
                if (num_per_grp != 1){
                    cudaMalloc((void**)&new_temp_1, 8*count*rows);
                    copy_from_operands<<<num_block, 256>>>(new_temp_1, OPERAND, d_valid, rows, count);
                    cudaDeviceSynchronize();
                }
            } else {
                temp_op_new = temp_op;
                cudaMalloc((void**)&new_temp_1, 8*count*rows);
                update_temp_weight<<<num_block, 256>>>(new_temp_1, temp_1, OPERAND, d_valid, rows, count, formula[idx-1] == 2);
                cudaDeviceSynchronize();
            }

            for (i=0; i<groups; i++){
                if (idx+2 == f_struct[i][2]){
                    chk = true;
                    break;
                }
            }
            if (chk || idx+1 == fml_shape){
                temp_0_change = true;
                cudaMalloc((void**)&new_temp_0, 8*count*rows);
                if (num_per_grp != 1){
                    update_last_weight<<<num_block, 256>>>(new_temp_0, temp_0, new_temp_1, rows, count, !temp_op_new, fml_deg, config.eval_method);
                } else {
                    update_last_weight_through_operands<<<num_block, 256>>>(
                        new_temp_0, temp_0, OPERAND, d_valid, rows, count, !temp_op_new
                    );
                } cudaDeviceSynchronize();
            }
            else temp_0_change = false;

            if (idx+1 != fml_shape){
                if (chk){
                    if (add_sub){
                        new_idx = idx + 2;
                        new_mode = 1;
                    } else {
                        new_idx = idx + 1;
                        new_mode = 0;
                    }
                } else {
                    if (mul_div){
                        new_idx = idx + 2;
                        new_mode = 1;
                    } else {
                        new_idx = idx + 1;
                        new_mode = 2;
                    }
                }

                if (temp_0_change){
                    if (num_per_grp != 1){
                        for (i=0; i<count; i++)
                            if (fill_formula(new_formula[i], f_struct, new_idx,
                                            new_temp_0+i*rows, temp_op_new, new_temp_1+i*rows,
                                            new_mode, add_sub, mul_div, fml_deg)) return true;
                    } else {
                        for (i=0; i<count; i++)
                            if (fill_formula(new_formula[i], f_struct, new_idx,
                                            new_temp_0+i*rows, temp_op_new, temp_1,
                                            new_mode, add_sub, mul_div, fml_deg)) return true;
                    }
                } else {
                    if (num_per_grp != 1){
                        for (i=0; i<count; i++)
                            if (fill_formula(new_formula[i], f_struct, new_idx,
                                            temp_0, temp_op_new, new_temp_1+i*rows,
                                            new_mode, add_sub, mul_div, fml_deg)) return true;
                    } else {
                        for (i=0; i<count; i++)
                            if (fill_formula(new_formula[i], f_struct, new_idx,
                                            temp_0, temp_op_new, temp_1,
                                            new_mode, add_sub, mul_div, fml_deg)) return true;
                    }
                }
            }
            else {
                cudaError_t status = cudaMemcpy(
                    temp_weight_storage+count_temp_storage*rows,
                    new_temp_0, 8*count*rows, cudaMemcpyDeviceToDevice
                );
                if (status){
                    raise_error("Cuda bad status", "cudaMemcpy");
                }
                for (i=0; i<count; i++){
                    memcpy(temp_formula_storage[count_temp_storage+i], new_formula[i], fml_shape);
                }
                count_temp_storage += count;
                for (i=0; i<fml_shape; i++) current[0][i] = formula[i];
                current[0][fml_shape-1] = cols;
                if (count_temp_storage >= config.storage_size){
                    replace_nan_and_inf<<<count_temp_storage*rows/256 + 1, 256>>>(
                        temp_weight_storage, rows, count_temp_storage
                    );
                    cudaDeviceSynchronize();
                    if (compute_result(false)) return true;
                }
            }

            // Giai phong bo nho
            for (i=0; i<count; i++){
                delete[] new_formula[i];
            }
            delete[] new_formula;
            delete[] valid;
            cudaFree(d_valid);
            if (temp_0_change) cudaFree(new_temp_0);
            if (num_per_grp != 1) cudaFree(new_temp_1);
        }

        // Giai phong bo nho
        delete[] valid_operand;
    }
    return false;
}


bool Generator::compute_result(bool force_save){
    raise_error("Ham khong chay duoc", "Generator::compute_result");
    return true;
}


void Generator::run(){
    bool first = true;
    int i;
    uint8_t *formula;
    int **f_struct;
    double *temp_0, *temp_1;
    cudaMalloc((void**)&temp_0, 8*rows);
    cudaMalloc((void**)&temp_1, 8*rows);

    double *h_temp_0 = new double[rows];
    for (i=0; i<rows; i++) h_temp_0[i] = 0.0;

    string command;
    command.reserve(200);

    // Loop num_opr_per_fml
    while (true){
        command = "python " + config.lib_abs_path + "/PySources/createTable.py "
            + config.folder_save + "/f.db 0 " + to_string(config.num_cycle-1) + " "
            + to_string(num_opr_per_fml) + " ";
        for (i=0; i<config.filter_field.size(); i++) command += config.filter_field[i] + " ";
        command.pop_back();
        system(command.c_str());
        fml_shape = num_opr_per_fml * 2;

        // Khoi tao formula
        formula = new uint8_t[fml_shape];
        for (i=0; i<config.storage_size+cols; i++)
            temp_formula_storage[i] = new uint8_t[fml_shape];

        for (i=0; i<config.num_cycle; i++){
            fmls_to_save[i] = new uint8_t[(__MAX_FTOS__ + config.storage_size + cols) * fml_shape];
            count_fml_save[i] = 0;
        }

        // Loop num_opr_per_grp
        for (num_per_grp=1; num_per_grp<=num_opr_per_fml; num_per_grp++){
            if (num_opr_per_fml%num_per_grp || num_per_grp < current[1][0]) continue;
            groups = num_opr_per_fml / num_per_grp;
            f_struct = new int*[groups];
            for (i=0; i<groups; i++){
                f_struct[i] = new int[4];
                f_struct[i][0] = 0;
                f_struct[i][1] = num_per_grp;
                f_struct[i][2] = 1 + 2*num_per_grp*i;
                f_struct[i][3] = 0;
            }
            current[1][0] = num_per_grp;
            for (i=0; i<2*num_opr_per_fml; i++) formula[i] = 0;

            cudaMemcpy(temp_0, h_temp_0, 8*rows, cudaMemcpyHostToDevice);
            cudaMemcpy(temp_1, h_temp_0, 8*rows, cudaMemcpyHostToDevice);

            if (first) first = false;
            else {
                for (i=0; i<fml_shape; i++) current[0][i] = 0;
            }
            if (fill_formula(formula, f_struct, 0, temp_0, 0, temp_1, 0, false, false, 1)) return;
            replace_nan_and_inf<<<count_temp_storage*rows/256 + 1, 256>>>(
                temp_weight_storage, rows, count_temp_storage
            );
            cudaDeviceSynchronize();
            if (compute_result(true)) return;

            //
            for (i=0; i<groups; i++) delete[] f_struct[i];
            delete[] f_struct;
        }

        //
        delete[] formula;
        for (i=0; i<config.storage_size+cols; i++) delete[] temp_formula_storage[i];

        //
        num_opr_per_fml += 1;
        for (i=0; i<config.num_cycle; i++)
            delete[] fmls_to_save[i];
        current[2][0] = 0;
        current[1][0] = 1;
        delete[] current[0];
        current[0] = new int64_t[2*num_opr_per_fml];
        for (i=0; i<2*num_opr_per_fml; i++)
            current[0][i] = 0;

        command = "python " + config.lib_abs_path + "/PySources/createCheckpoint.py "
            + config.folder_save + "/f.db " + to_string(num_opr_per_fml);
        system(command.c_str());
    }

    //
    delete[] h_temp_0;
    cudaFree(temp_0);
    cudaFree(temp_1);
}


bool Generator::save_result(bool force_save, double *h_final, int *is_save){
    int i, j;
    int64_t k;
    int num_field = config.filter_field.size();
    for (i=0; i<count_temp_storage; i++){
        for (j=0; j<config.num_cycle; j++){
            if (is_save[i*config.num_cycle+j]){
                k = current[2][0]+i;
                // Lưu vào mảng kết quả
                fmls_idx[j][count_fml_save[j]] = k;
                memcpy(fmls_to_save[j] + count_fml_save[j] * fml_shape, temp_formula_storage[i], fml_shape);
                memcpy(fields_to_save[j] + num_field * count_fml_save[j], h_final + i*config.num_cycle*num_field + j*num_field, 8*num_field);
                count_fml_save[j] ++;
            }
        }
    }

    current[2][0] += count_temp_storage;
    count_temp_storage = 0;

    bool save = false;
    if (force_save) save = true;
    if (!save){
        for (i=0; i<config.num_cycle; i++){
            if (count_fml_save[i] >= __MAX_FTOS__){
                save = true; break;
            }
        }
    }
    if (!save){
        chrono::high_resolution_clock::time_point now = chrono::high_resolution_clock::now();
        chrono::seconds duration = chrono::duration_cast<chrono::seconds>(now - start_t_temp);
        if (duration.count() >= __SAVE_FREQ__) save = true;
    }

    if (save){
        // Lưu các mảng vào các folder
        string all_query = "delete from checkpoint_" + to_string(fml_shape/2) + ";";
        all_query += "insert into checkpoint_" + to_string(fml_shape/2) + " values (";
        all_query += to_string(current[2][0]) + "," + to_string(current[1][0]) + ",";
        for (i=0; i<fml_shape; i++){
            all_query += to_string(current[0][i]) + ",";
        }
        all_query.pop_back();
        all_query += ");";

        // Write query
        string filename = config.folder_save + "/queries.bin";
        ofstream outFile(filename.c_str(), ios::binary);
        if (outFile.is_open()){
            outFile.write(all_query.c_str(), all_query.size());
            outFile.close();
        }
        else raise_error("Khong mo duoc file", filename);

        // Write array
        filename = config.folder_save + "/result.bin";
        ofstream outFile2(filename.c_str(), ios::binary);
        if (outFile2.is_open()){
            outFile2.write(reinterpret_cast<char*>(count_fml_save), 4 * config.num_cycle);
            for (i=0; i<config.num_cycle; i++){
                outFile2.write(reinterpret_cast<char*>(fmls_idx[i]), count_fml_save[i] * 8);
                outFile2.write(reinterpret_cast<char*>(fmls_to_save[i]), count_fml_save[i] * fml_shape);
                outFile2.write(reinterpret_cast<char*>(fields_to_save[i]), count_fml_save[i] * num_field * 8);
            }
            outFile2.close();
        }
        else raise_error("Khong mo duoc file", filename);

        //
        string command = "python " + config.lib_abs_path + "/PySources/insertResult.py "
            + config.folder_save + "/f.db " // db_path
            + to_string(config.num_cycle) + " " // num_cycle
            + to_string(fml_shape) + " " // fml_shape
            + to_string(num_field) + " " // num_field
            + to_string(cols); // cols
        system(command.c_str());

        //
        start_t_temp = chrono::high_resolution_clock::now();
        for (i=0; i<config.num_cycle; i++){
            count_fml_save[i] = 0;
        }

        chrono::high_resolution_clock::time_point check_timeout = chrono::high_resolution_clock::now();
        chrono::seconds total_runtime = chrono::duration_cast<chrono::seconds>(check_timeout - time_start);
        long long time_range = total_runtime.count();

        cout << FG_CYAN << "Running: " << time_range << "/" << config.timeout_in_minutes*60 << " secs!\n" << RESET_COLOR;
        cout << FG_BRIGHT_CYAN << current[2][0] << endl;
        cout << current[1][0] << endl;
        for (i=0; i<2*num_opr_per_fml; i++) cout << current[0][i] << " ";
        cout << endl << RESET_COLOR;

        if (time_range >= config.timeout_in_minutes*60) return true;
    }
    return false;
}
