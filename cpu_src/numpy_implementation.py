import numpy as np
import argparse
import time

def sgemm(A, B):
    return np.dot(A, B)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=1024)
    parser.add_argument("--n", type=int, default=1024)
    parser.add_argument("--k", type=int, default=1024)
    parser.add_argument("--num-iterations", type=int, default=1000)
    args = parser.parse_args()
    m = args.m
    n = args.n
    k = args.k
    num_iterations = args.num_iterations

    A = np.random.rand(m, k).astype(np.float32)
    B = np.random.rand(k, n).astype(np.float32)
    # Measure execution time
    start_time = time.time()
    for _ in range(num_iterations):
        C = sgemm(A, B)
    end_time = time.time()

    # Calculate average time per iteration
    avg_time = (end_time - start_time) / num_iterations

    # Calculate FLOPS
    flops = (2 * m * n * k) / avg_time

    print(f"Time taken for sgemm numpy implementation of matrix {m}x{k} with matrix {k}x{n} over {num_iterations} iterations: {avg_time:.8f} seconds per iteration")
    print(f"FLOPS: {flops:.2e}")

if __name__ == "__main__":
    main()