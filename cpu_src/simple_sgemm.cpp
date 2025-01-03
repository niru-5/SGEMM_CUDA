#include <iostream>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <omp.h>


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

    #pragma omp parallel for
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





int main() {
    omp_set_num_threads(6);
    int m = 1024;
    int n = 1024;
    int k = 1024;
    float *A = (float *)malloc(m * k * sizeof(float));
    float *B = (float *)malloc(k * n * sizeof(float));
    float *C = (float *)malloc(m * n * sizeof(float));

    // assign random values to A and B
    for (int i = 0; i < m * k; i++) {
        A[i] = rand();
    }
    for (int i = 0; i < k * n; i++) {
        B[i] = rand();
    }

    for (int i = 0; i < m * n; i++) {
        C[i] = 0;
    }


    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i<1; i++){
        sgemm_simple(A, B, C, m, n, k);
        // sgemm_cache_aware_save_mult(A, B, C, m, n, k);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Time taken for sgemm simple: " << duration.count() << " seconds" << std::endl;


    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i<1; i++){
        sgemm_cache_aware(A, B, C, m, n, k);
        // sgemm_cache_aware_save_mult(A, B, C, m, n, k);
    }
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    std::cout << "Time taken for sgemm cache aware: " << duration.count() << " seconds" << std::endl;


    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i<1; i++){
        // sgemm_cache_aware_save_mult(A, B, C, m, n, k);
        sgemm_cache_aware_save_mult(A, B, C, m, n, k);
    }
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    std::cout << "Time taken for sgemm cache aware save mult: " << duration.count() << " seconds" << std::endl;

    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i<1; i++){
        // sgemm_cache_aware_save_mult(A, B, C, m, n, k);
        sgemm_simd(A, B, C, m, n, k);
    }
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    std::cout << "Time taken for sgemm simd: " << duration.count() << " seconds" << std::endl;


    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i<1; i++){
        // sgemm_cache_aware_save_mult(A, B, C, m, n, k);
        sgemm_cache_aware_tiling(A, B, C, m, n, k);
    }
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    std::cout << "Time taken for sgemm cache aware tiling: " << duration.count() << " seconds" << std::endl;

    

    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i<1; i++){
        // sgemm_cache_aware_save_mult(A, B, C, m, n, k);
        sgemm_cache_aware_save_mult_tiling(A, B, C, m, n, k);
    }
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    std::cout << "Time taken for sgemm cache aware save mult tiling: " << duration.count() << " seconds" << std::endl;



    free(A);
    free(B);
    free(C);
    return 0;
}