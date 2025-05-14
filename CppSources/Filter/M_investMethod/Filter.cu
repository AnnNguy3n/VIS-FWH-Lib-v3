#pragma once
#include "Kernel.cu"
#include "../../Generator/HomoPoly/CUDA/HomoPolyMethod.cu"


class Multi_investMethod: public Generator {
public:
    double *d_threshold;
    double *d_result;
    double *d_final;
    double *h_final;
    int *d_check_save;
    int *h_check_save;

    double *d_tmp_profit;
    double *d_tmp_geomean;
    double *d_tmp_harmean;
    int    *d_invest_count;
    int    *d_symbol_streak;

    Multi_investMethod(string config_path);
    ~Multi_investMethod();

    bool compute_result(bool force_save);
};


Multi_investMethod::Multi_investMethod(string config_path)
: Generator(config_path) {
    int num_array     = config.storage_size + cols;
    int num_threshold = __NUM_THRESHOLD_PER_CYCLE__ * (index_length - 2);
    int num_cycle     = config.num_cycle;
    int num_strategy  = config.num_strategy;

    // Tổng số phần tử
    size_t total_thresholds = static_cast<size_t>(num_array) * num_threshold;
    size_t result_size      = total_thresholds * 2 * num_cycle * num_strategy;
    size_t final_size       = static_cast<size_t>(num_array) * num_cycle * num_strategy * 4;
    size_t check_size = static_cast<size_t>(num_array) * num_cycle;
    size_t total_threads = static_cast<size_t>(num_array) * num_threshold;

    // Cấp phát
    cudaMalloc((void**)&d_threshold, total_thresholds * sizeof(double));
    cudaMalloc((void**)&d_result, result_size * sizeof(double));
    cudaMalloc((void**)&d_final, final_size * sizeof(double));
    cudaMalloc((void**)&d_check_save, check_size * sizeof(int));

    // Cấp phát bộ nhớ tạm (per-thread)
    cudaMalloc((void**)&d_tmp_profit,     total_threads * num_strategy * sizeof(double));
    cudaMalloc((void**)&d_tmp_geomean,    total_threads * num_strategy * sizeof(double));
    cudaMalloc((void**)&d_tmp_harmean,    total_threads * num_strategy * sizeof(double));
    cudaMalloc((void**)&d_invest_count,   total_threads * num_strategy * sizeof(int));
    cudaMalloc((void**)&d_symbol_streak,  total_threads * num_symbol_unique * sizeof(int));

    // Khởi tạo giá trị 0 cho kết quả
    int threads = 256;
    int blocks_result = (result_size + threads - 1) / threads;
    cuda_set_array_value<<<blocks_result, threads>>>(d_result, result_size, 0.0);
    cudaDeviceSynchronize();

    // Cấp phát host
    h_final = new double[final_size];
    h_check_save = new int[check_size];
}


Multi_investMethod::~Multi_investMethod(){
    cudaFree(d_threshold);
    cudaFree(d_result);
    cudaFree(d_final);
    cudaFree(d_check_save);
    cudaFree(d_tmp_profit);
    cudaFree(d_tmp_geomean);
    cudaFree(d_tmp_harmean);
    cudaFree(d_invest_count);
    cudaFree(d_symbol_streak);
    delete[] h_final;
    delete[] h_check_save;
}


bool Multi_investMethod::compute_result(bool force_save){
    int num_array     = count_temp_storage;
    int num_threshold = __NUM_THRESHOLD_PER_CYCLE__ * (index_length - 2);
    int num_cycle     = config.num_cycle;
    int num_strategy  = config.num_strategy;
    int array_length  = rows;

    dim3 threads(256);

    // 1. Sinh threshold cho từng (array, cycle) → d_threshold
    int num_fill = num_array * (index_length - 2);  // = số cycle áp dụng
    int blocks_fill = (num_fill + threads.x - 1) / threads.x;
    fill_thresholds<<<blocks_fill, threads>>>(
        temp_weight_storage, d_threshold,
        INDEX, index_length, num_array
    );
    cudaDeviceSynchronize();

    // 2. Chạy kernel đầu tư song song trên (array × threshold)
    int total_threads = num_array * num_threshold;
    int blocks_invest = (total_threads + threads.x - 1) / threads.x;
    M_investMethod<<<blocks_invest, threads>>>(
        temp_weight_storage,
        d_threshold,
        PROFIT,
        INDEX,
        SYMBOL,
        BOOL_ARG,

        d_result,                        // N_result
        d_tmp_profit,                    // các buffer phụ đã malloc
        d_tmp_geomean,
        d_tmp_harmean,
        d_invest_count,
        d_symbol_streak,

        config.interest,
        index_length,
        num_cycle,
        num_symbol_unique,
        num_strategy,
        num_array,
        num_threshold
    );
    cudaDeviceSynchronize();

    // 3. Tìm threshold tối ưu (geo + har) → d_final
    int final_threads = num_array * num_cycle * num_strategy;
    int blocks_final = (final_threads + threads.x - 1) / threads.x;

    find_best_results<<<blocks_final, threads>>>(
        d_result,
        d_threshold,
        d_final,
        num_array,
        num_threshold,
        num_cycle,
        num_strategy
    );
    cudaDeviceSynchronize();

    // 4. Copy kết quả từ device về host
    size_t final_size = static_cast<size_t>(num_array) * num_cycle * num_strategy * 4;
    cudaMemcpy(h_final, d_final, final_size * sizeof(double), cudaMemcpyDeviceToHost);

    // 5. Tính toán d_check_save
    dim3 blocks_check(count_temp_storage);       // mỗi block xử lý 1 array
    dim3 threads_check(config.num_cycle);        // mỗi thread xử lý 1 cycle

    mark_check_save_from_final<<<blocks_check, threads_check>>>(
        d_final,
        d_check_save,
        count_temp_storage,
        config.num_cycle,
        config.num_strategy,
        config.eval_threshold
    );
    cudaDeviceSynchronize();

    // 6. Copy
    cudaMemcpy(h_check_save, d_check_save, sizeof(int) * count_temp_storage * config.num_cycle, cudaMemcpyDeviceToHost);

    //
    return save_result(force_save, h_final, h_check_save);
}
