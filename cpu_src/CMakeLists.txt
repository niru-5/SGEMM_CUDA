cmake_minimum_required(VERSION 3.10)
project(SGEMM_Benchmark)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenMP
find_package(OpenMP REQUIRED)

# Find Google Benchmark
find_package(benchmark REQUIRED)

# Add executable
add_executable(sgemm_benchmark sgemm_benchmark.cpp)

# Link libraries
target_link_libraries(sgemm_benchmark
    PRIVATE
    benchmark::benchmark
    OpenMP::OpenMP_CXX
)

# Enable optimizations for Release build
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

# Add compiler warnings
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(sgemm_benchmark PRIVATE -Wall -Wextra)
endif()