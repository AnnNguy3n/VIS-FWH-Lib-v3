#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cfloat>

const float test = FLT_MAX;

#ifndef _NUM_THRESHOLD_PER_CYCLE_
#define _NUM_THRESHOLD_PER_CYCLE_
const int __NUM_THRESHOLD_PER_CYCLE__ = 10;
#endif


__constant__ int C_INDEX[256];

__constant__ int    index_size;
__constant__ int    num_cycle_result;
__constant__ int    __num_symbol_unique__;
__constant__ int    num_strategy;
__constant__ double interest;


__device__ __forceinline__
void _M_investMethod(
    const double* __restrict__ weight,                // length = array_length
    const double* __restrict__ profit,                // length = array_length
    // const int*    __restrict__ index,                 // length = index_size
    const int*    __restrict__ symbol,                // length = array_length
    const int*    __restrict__ sufficient_liquidity,  // length = array_length

          double* __restrict__ result,                // length = 2·num_cycle_result·num_strategy
          float* __restrict__ tmp_profit,            // length = num_strategy
          float* __restrict__ tmp_geomean,           // length = num_strategy
          float* __restrict__ tmp_harmean,           // length = num_strategy
          int*    __restrict__ invest_count,          // length = num_strategy
          uint8_t*    __restrict__ symbol_streak,         // length = num_symbol_unique

    const double threshold,
    // const double interest,
    const int    threshold_cycle_idx                 // = x / __NUM_THRESHOLD_PER_CYCLE__
    
) {
    // Khởi tạo
    int market_streak = 0;

    for (int i = 0; i < __num_symbol_unique__; ++i) symbol_streak[i] = 0;
    for (int i = 0; i < num_strategy;      ++i) {
        tmp_geomean[i] = 0.0;
        tmp_harmean[i] = 0.0;
    }

    // Duyệt ngược qua từng chu kỳ (bỏ qua chu kỳ gần nhất – “index_size‑1”)
    for (int cycle_idx = index_size - 2; cycle_idx >= 1; --cycle_idx) {
        const int start = C_INDEX[cycle_idx];
        const int end   = C_INDEX[cycle_idx + 1];

        // Reset bộ đếm đầu tư cho chu kỳ hiện tại
        for (int k = 0; k < num_strategy; ++k) {
            tmp_profit[k]   = 0.0;
            invest_count[k] = 0;
        }

        const int num_applicable_strategy =
            min(index_size - 1 - cycle_idx, num_strategy);

        bool any_pass_threshold = false;

        // Duyệt qua tất cả công ty trong chu kỳ hiện tại
        for (int i = start; i < end; ++i) {
            if (weight[i] <= threshold) { // Không vượt ngưỡng
                symbol_streak[symbol[i]] = 0;
                continue;
            }

            any_pass_threshold = true;
            const int sym = symbol[i];
            const int sym_streak = ++symbol_streak[sym];

            if (!sufficient_liquidity[i]) continue;

            const double p = profit[i];

            // Update cho từng strategy k (tương ứng độ dài streak yêu cầu)
            for (int k = 0; k < num_applicable_strategy; ++k) {
                if (sym_streak > min(market_streak, k)) {
                    tmp_profit[k] += p;
                    ++invest_count[k];
                }
            }
        }

        // Cập nhật geo / har tích luỹ
        for (int k = 0; k < num_applicable_strategy; ++k) {
            const double avg_p = invest_count[k] ?
                                 tmp_profit[k] / (double)invest_count[k] :
                                 interest;
            tmp_geomean[k] += log(avg_p);
            tmp_harmean[k] += 1.0 / avg_p;
        }

        // Lưu kết quả nếu cycle nằm trong vùng cần ghi
        if (cycle_idx <= num_cycle_result && threshold_cycle_idx + 1 >= cycle_idx) {
            const int segment = num_cycle_result - cycle_idx;
            for (int k = 0; k < num_applicable_strategy; ++k) {
                const int n   = index_size - 1 - cycle_idx - k;
                const int idx = 2 * (segment * num_strategy + k);

                // result[idx    ] = exp(tmp_geomean[k] / (double)n);  // geo‑mean
                // result[idx + 1] = (double)n / tmp_harmean[k];       // har‑mean
                reinterpret_cast<double2*>(&result[idx])[0] = make_double2(exp(tmp_geomean[k] / (double)n), (double)n / tmp_harmean[k]);
            }
        }

        // Cập nhật market streak
        market_streak = any_pass_threshold ? market_streak + 1 : 0;
    }
}


__global__ void M_investMethod(
    const double* __restrict__ N_weight,                // num_array · array_length
    const double* __restrict__ N_threshold,             // num_array · num_threshold_per_array = num_kernel
    const double* __restrict__ profit,                  // array_length
    // const int*    __restrict__ index,                   // index_size
    const int*    __restrict__ symbol,                  // array_length
    const int*    __restrict__ sufficient_liquidity,    // array_length

          double* __restrict__ N_result,                // num_kernel × 2·num_cycle_result·num_strategy
        //   double* __restrict__ N_tmp_profit,            // num_kernel × num_strategy
        //   double* __restrict__ N_tmp_geomean,           // num_kernel × num_strategy
        //   double* __restrict__ N_tmp_harmean,           // num_kernel × num_strategy
        //   int*    __restrict__ N_invest_count,          // num_kernel × num_strategy
        //   int*    __restrict__ N_symbol_streak,         // num_kernel × num_symbol_unique

    // const double interest,
    // const int    index_size,
    // const int    num_cycle_result,
    // const int    num_symbol_unique,
    // const int    num_strategy,
    const int    num_array,
    const int    num_threshold_per_array
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_threads = num_array * num_threshold_per_array;
    if (tid >= total_threads) return;

    const int thr_idx   =  tid % num_threshold_per_array;   // x
    const int array_idx =  tid / num_threshold_per_array;   // y

    const int array_length = C_INDEX[index_size - 1];

    // Pointers/offsets dành riêng cho thread hiện tại
    const double* weight_ptr    = N_weight + (size_t)array_idx * array_length;
    const double  threshold_val = N_threshold[tid];

    double* result_ptr     = N_result +
        (size_t)tid * 2 * num_cycle_result * num_strategy;

    // double* tmp_profit_ptr = N_tmp_profit    + (size_t)tid * num_strategy;
    // double* tmp_geo_ptr    = N_tmp_geomean   + (size_t)tid * num_strategy;
    // double* tmp_har_ptr    = N_tmp_harmean   + (size_t)tid * num_strategy;
    // int*    icount_ptr     = N_invest_count  + (size_t)tid * num_strategy;
    // int*    streak_ptr     = N_symbol_streak + (size_t)tid * num_symbol_unique;

    // Shared memory cho symbol_streak
    extern __shared__ uint8_t shared_streak[];
    uint8_t* streak_ptr = &shared_streak[threadIdx.x * __num_symbol_unique__];

    float* N_tmp_profit = reinterpret_cast<float*>(&shared_streak[blockDim.x * __num_symbol_unique__]);
    float* tmp_profit_ptr = &N_tmp_profit[threadIdx.x * num_strategy];

    float* N_tmp_geomean = reinterpret_cast<float*>(&N_tmp_profit[blockDim.x * num_strategy]);
    float* tmp_geo_ptr = &N_tmp_geomean[threadIdx.x * num_strategy];

    float* N_tmp_harmean = reinterpret_cast<float*>(&N_tmp_geomean[blockDim.x * num_strategy]);
    float* tmp_har_ptr = &N_tmp_harmean[threadIdx.x * num_strategy];

    int* N_invest_count = reinterpret_cast<int*>(&N_tmp_harmean[blockDim.x * num_strategy]);
    int* icount_ptr = &N_invest_count[threadIdx.x * num_strategy];

    const int threshold_cycle_idx = thr_idx / __NUM_THRESHOLD_PER_CYCLE__;

    _M_investMethod(
        weight_ptr, profit, symbol, sufficient_liquidity,
        result_ptr, tmp_profit_ptr, tmp_geo_ptr, tmp_har_ptr, icount_ptr, streak_ptr,
        threshold_val, threshold_cycle_idx
    );
}


__global__ void find_best_results(
    const double* __restrict__ results,     // num_array · num_threshold_per_array × 2·num_cycle_result·num_strategy
    const double* __restrict__ thresholds,  // num_array · num_threshold_per_array
          double* __restrict__ finals,      // num_array · num_cycle_result · num_strategy · 4
    const int num_array,
    const int num_threshold_per_array,
    const int num_cycle_result,
    const int num_strategy
) {
    const int tid            = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_triplets = num_array * num_cycle_result * num_strategy;
    if (tid >= total_triplets) return;

    const int strategy =  tid %  num_strategy;
    const int cycle    = (tid /  num_strategy) % num_cycle_result;
    const int array    =  tid / (num_strategy * num_cycle_result);

    const int stride_per_threshold = 2 * num_cycle_result * num_strategy;

    double best_geo_val = -DBL_MAX, best_geo_thr = 0.0;
    double best_har_val = -DBL_MAX, best_har_thr = 0.0;

    for (int t = 0; t < num_threshold_per_array; ++t) {
        const int th_global = array * num_threshold_per_array + t;

        // offset tới cặp (geo, har) tương ứng với (cycle, strategy) trong threshold hiện tại
        const double* result_t = results + (size_t)th_global * stride_per_threshold;
        const int     off      = 2 * (cycle * num_strategy + strategy);

        const double geo_val = result_t[off];
        const double har_val = result_t[off + 1];

        if (geo_val > best_geo_val) {
            best_geo_val = geo_val;
            best_geo_thr = thresholds[th_global];
        }
        if (har_val > best_har_val) {
            best_har_val = har_val;
            best_har_thr = thresholds[th_global];
        }
    }

    // Ghi kết quả ra finals
    double* final_slot = finals + tid * 4;

    final_slot[0] = best_geo_thr;
    final_slot[1] = best_geo_val;
    final_slot[2] = best_har_thr;
    final_slot[3] = best_har_val;
}


__device__ __forceinline__
void topNUnique(
    const double* __restrict__ array,
    int left, int right,
          double* __restrict__ result,
    int start, int n
) {
    int size = 0;

    for (int i = left; i < right; ++i) {
        const double val = array[i];
        bool duplicated = false;
        for (int j = 0; j < size; ++j) {
            if (val == result[start + j]) { duplicated = true; break; }
        }
        if (duplicated) continue;

        if (size < n) {
            result[start + size++] = val;
        } else {
            int min_pos = start;
            for (int j = 1; j < n; ++j) {
                const int idx = start + j;
                if (result[idx] < result[min_pos]) min_pos = idx;
            }
            if (val > result[min_pos]) result[min_pos] = val;
        }
    }

    for (int i = size; i < n; ++i)
        result[start + i] = -DBL_MAX;

    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            const int a = start + i, b = start + j;
            if (result[b] > result[a]) {
                const double tmp = result[b];
                result[b] = result[a];
                result[a] = tmp;
            }
        }
    }
}


__global__ void fill_thresholds(
    const double* __restrict__ weights,
          double* __restrict__ thresholds,
    const int*    __restrict__ index,
    const int index_size,
    const int num_array
) {
    const int tid          = blockIdx.x * blockDim.x + threadIdx.x;
    const int num_cycle    = index_size - 2;
    const int total        = num_array * num_cycle;
    if (tid >= total) return;

    const int cycle_idx  =  tid % num_cycle;                // ix
    const int array_idx  =  tid / num_cycle;                // iy
    const int array_len  =  index[index_size - 1];

    // Offset con trỏ
    const double* w_ptr   = weights +
        (size_t)array_idx * array_len;

    double* thr_ptr = thresholds +
        ((size_t)array_idx * num_cycle + cycle_idx) *
        __NUM_THRESHOLD_PER_CYCLE__;

    const int left  = index[cycle_idx + 1];
    const int right = index[cycle_idx + 2];

    topNUnique(
        w_ptr, left, right,
        thr_ptr, 0, __NUM_THRESHOLD_PER_CYCLE__
    );
}


__global__ void mark_check_save_from_final(
    const double* __restrict__ final,          // d_final
          int*    __restrict__ check_save,     // d_check_save
    const int num_array,
    const int num_cycle,
    const int num_strategy,
    const double eval_threshold
) {
    int array_idx = blockIdx.x;
    int cycle_idx = threadIdx.x;

    if (array_idx >= num_array || cycle_idx >= num_cycle) return;

    bool should_save = false;

    for (int s = 0; s < num_strategy; ++s){
        size_t offset = ((size_t)array_idx * num_cycle + cycle_idx) * num_strategy + s;
        double har_val = final[offset * 4 + 3]; // chỉ số 3 = har-mean value
        if (har_val > eval_threshold) {
            should_save = true;
            break;
        }
    }

    check_save[array_idx * num_cycle + cycle_idx] = should_save ? 1 : 0;
}