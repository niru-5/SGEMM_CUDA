#include <iostream>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <omp.h>
#include <benchmark/benchmark.h>


void sgemm_simple(float *A, float *B, float *C, int m, int n, int k) {
    // #pragma omp parallel for
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            for (int l = 0; l < k; l++) {
                C[i * n + j] += A[i * k + l] * B[l * n + j];
            }
        }
    }
}  

// Modify your sgemm_simple function to accept benchmark::State
static void BM_SGEMM_Simple(benchmark::State& state) {
    // Setup your matrices
    int m = 1024;
    int n = 1024;
    int k = 1024;
    float *A = (float *)malloc(m * k * sizeof(float));
    float *B = (float *)malloc(k * n * sizeof(float));
    float *C = (float *)malloc(m * n * sizeof(float));

    // Initialize matrices
    for (int i = 0; i < m * k; i++) A[i] = rand();
    for (int i = 0; i < k * n; i++) B[i] = rand();
    for (int i = 0; i < m * n; i++) C[i] = 0;

    // Benchmark loop
    for (auto _ : state) {
        sgemm_simple(A, B, C, m, n, k);
    }

    // Cleanup
    free(A);
    free(B);
    free(C);
}

// Register the benchmarks
BENCHMARK(BM_SGEMM_Simple)
    ->Unit(benchmark::kMillisecond);

void sgemm_cache_aware(float *A, float *B, float *C, int m, int n, int k) {
    #pragma omp parallel for
    for (int i = 0; i < m; i++) {
        for (int l = 0; l < k; l++) {
            for (int j = 0; j < n; j++) {
                C[i * n + j] += A[i * k + l] * B[l * n + j];
            }
        }
    }
}

void sgemm_cache_aware_tiling(float *A, float *B, float *C, int m, int n, int k) {

    int k_div = k/2;

    #pragma omp parallel for
    for (int i = 0; i < m; i++) {
        for (int l = 0; l < k_div; l++) {
            for (int j = 0; j < n; j++) {
                C[i * n + j] += A[i * k + l] * B[l * n + j];
            }
        }
    }

    #pragma omp parallel for
    for (int i = 0; i < m; i++) {
        for (int l = k_div; l < k; l++) {
            for (int j = 0; j < n; j++) {
                C[i * n + j] += A[i * k + l] * B[l * n + j];
            }
        }
    }
}




void sgemm_simd(float *A, float *B, float *C, int m, int n, int k) {
    #pragma omp parallel for
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0;
            for (int l = 0; l < k; l++) {
                sum += A[i * k + l] * B[l * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}

void sgemm_cache_aware_save_mult(float *A, float *B, float *C, int m, int n, int k) {
    
    int Cth_row = 0;
    int Ath_row = 0;
    int Bth_row = 0;

    // #pragma omp parallel for
    for (int i = 0; i < m; i++) {
        Bth_row = 0;
        for (int l = 0; l < k; l++) {
            float A_val = A[Ath_row + l];
            for (int j = 0; j < n; j++) {
                C[Cth_row + j] += A_val * B[Bth_row + j];
            }
            Bth_row += n;
        }
        
        Cth_row += n;
        Ath_row += k;
    }
}


void sgemm_cache_aware_save_mult_tiling(float *A, float *B, float *C, int m, int n, int k) {
    // lets break it down to 16 tiles. 
    // for now, lets assume that m,n,k are perfectly divisible by 16. 
    int div_val = 128;
    int new_m = m/div_val;
    int new_n = n/div_val;
    int new_k = k/div_val;

    // #pragma omp parallel for collapse(2) schedule(static)
    // #pragma omp parallel for
    for (int i = 0; i < new_m; i++){
        for (int j =0; j< new_n; j++){
            for (int l=0; l< new_k; l++){
                sgemm_cache_aware(&A[i*m*div_val + l*div_val], 
                            &B[l*n*div_val + j*div_val], 
                            &C[i*n*div_val +j*div_val], 
                            div_val, div_val, div_val);

            }
        }
    }


}

// Similarly for cache aware version
static void BM_SGEMM_Cache_Aware(benchmark::State& state) {
    int m = 1024;
    int n = 1024;
    int k = 1024;
    float *A = (float *)malloc(m * k * sizeof(float));
    float *B = (float *)malloc(k * n * sizeof(float));
    float *C = (float *)malloc(m * n * sizeof(float));

    for (int i = 0; i < m * k; i++) A[i] = rand();
    for (int i = 0; i < k * n; i++) B[i] = rand();
    for (int i = 0; i < m * n; i++) C[i] = 0;

    for (auto _ : state) {
        sgemm_cache_aware(A, B, C, m, n, k);
    }

    free(A);
    free(B);
    free(C);
}

// Register the benchmarks
BENCHMARK(BM_SGEMM_Cache_Aware)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(10)
    ->MinTime(2)
    ->ReportAggregatesOnly()
    ->Repetitions(3);

// Add other benchmark functions similarly...

// Replace your main() with BENCHMARK_MAIN()
BENCHMARK_MAIN();