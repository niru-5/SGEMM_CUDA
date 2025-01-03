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
    int div_val = 64;
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

void sgemm_cache_aware_save_mult_tiling_same_function(float *A, float *B, float *C, int m, int n, int k) {
    // lets break it down to 16 tiles. 
    // for now, lets assume that m,n,k are perfectly divisible by 16. 
    int div_val = 128;
    int new_m = m/div_val;
    int new_n = n/div_val;
    int new_k = k/div_val;

    // 
    // #pragma omp parallel for
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < new_m; i++){
        for (int j =0; j< new_n; j++){
            for (int l=0; l< new_k; l++){
                
                float *A_block = &A[i*m*div_val + l*div_val];
                float *B_block = &B[l*n*div_val + j*div_val];
                float *C_block = &C[i*n*div_val +j*div_val];
                
                #pragma omp parallel for
                for (int i = 0; i < div_val; i++) {
                    for (int l = 0; l < div_val; l++) {
                        for (int j = 0; j < div_val; j++) {
                            C_block[i * div_val + j] += 
                                        A_block[i * div_val + l] * 
                                        B_block[l * div_val + j];
                        }
                    }
                }

            }
        }
    }
}



// First, create a fixture class that contains the common setup
class SGEMMFixture : public benchmark::Fixture {
public:
    void SetUp(const ::benchmark::State& state) {
        // Ensure state has at least one range value
        if (state.range(0) <= 0) {
            m = n = k = 1024;  // Default size if no range provided
        } else {
            m = n = k = state.range(0);
        }
        
        // Allocate matrices
        A = (float *)malloc(m * k * sizeof(float));
        B = (float *)malloc(k * n * sizeof(float));
        C = (float *)malloc(m * n * sizeof(float));

        // Initialize matrices with same random values for all tests
        srand(42);  // Fixed seed for reproducibility
        for (int i = 0; i < m * k; i++) A[i] = rand();
        for (int i = 0; i < k * n; i++) B[i] = rand();
    }

    void TearDown(const ::benchmark::State& state) {
        free(A);
        free(B);
        free(C);
    }

    // Reset C matrix between iterations
    void ResetC() {
        for (int i = 0; i < m * n; i++) C[i] = 0;
    }

    float *A, *B, *C;
    int m, n, k;
};

// Define benchmarks using the fixture
BENCHMARK_DEFINE_F(SGEMMFixture, Simple)(benchmark::State& state) {
    for (auto _ : state) {
        ResetC();
        sgemm_simple(A, B, C, m, n, k);
    }
}

BENCHMARK_DEFINE_F(SGEMMFixture, CacheAware)(benchmark::State& state) {
    for (auto _ : state) {
        ResetC();
        sgemm_cache_aware(A, B, C, m, n, k);
    }
}

BENCHMARK_DEFINE_F(SGEMMFixture, SIMD)(benchmark::State& state) {
    for (auto _ : state) {
        ResetC();
        sgemm_simd(A, B, C, m, n, k);
    }
}

BENCHMARK_DEFINE_F(SGEMMFixture, CacheAwareSaveMult)(benchmark::State& state) {
    for (auto _ : state) {
        ResetC();
        sgemm_cache_aware_save_mult(A, B, C, m, n, k);
    }
}

BENCHMARK_DEFINE_F(SGEMMFixture, CacheAwareSaveMultTiling)(benchmark::State& state) {
    for (auto _ : state) {
        ResetC();
        sgemm_cache_aware_save_mult_tiling(A, B, C, m, n, k);
    }
}



BENCHMARK_DEFINE_F(SGEMMFixture, CacheAwareSaveMultTilingSameFunction)(benchmark::State& state) {
    for (auto _ : state) {
        ResetC();
        sgemm_cache_aware_save_mult_tiling_same_function(A, B, C, m, n, k);
    }
}

// Register all benchmarks with same settings
template <typename T>
void ConfigureBenchmark(T* b) {
    b->Unit(benchmark::kMillisecond)
     ->Iterations(10)
     ->MinTime(5)
     ->ReportAggregatesOnly()
     ->Repetitions(10)
     ->Arg(1024);  // This must be uncommented
}

// Apply settings to all benchmarks
// BENCHMARK_REGISTER_F(SGEMMFixture, Simple)->Apply(ConfigureBenchmark);
// BENCHMARK_REGISTER_F(SGEMMFixture, CacheAware)->Apply(ConfigureBenchmark);
// BENCHMARK_REGISTER_F(SGEMMFixture, SIMD)->Apply(ConfigureBenchmark);
// BENCHMARK_REGISTER_F(SGEMMFixture, CacheAwareSaveMult)->Apply(ConfigureBenchmark);
// BENCHMARK_REGISTER_F(SGEMMFixture, CacheAwareSaveMultTiling)->Apply(ConfigureBenchmark);
BENCHMARK_REGISTER_F(SGEMMFixture, CacheAwareSaveMultTilingSameFunction)->Apply(ConfigureBenchmark);

BENCHMARK_MAIN();