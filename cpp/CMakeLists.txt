cmake_minimum_required(VERSION 3.24)

# nvcc 12.0 (paket Ubuntu) belum mengenal sm_120 (RTX 50xx). Kode dikompilasi sebagai PTX
# compute_90 lalu di-JIT oleh driver ke GPU yang terpasang saat modul pertama kali dimuat.
# Dengan CUDA >= 12.8 bisa diganti, mis. -DCMAKE_CUDA_ARCHITECTURES=120.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 90-virtual)
endif()

project(rag_hpc LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Python 3 COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
find_package(OpenMP REQUIRED)

# Kernel CUDA dibangun sebagai library statis terpisah agar flag pybind11 (LTO, visibility)
# tidak diteruskan ke nvcc.
add_library(rag_hpc_cuda STATIC src/topk_cuda.cu)
target_compile_options(rag_hpc_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 -lineinfo>)
set_target_properties(rag_hpc_cuda PROPERTIES CUDA_RUNTIME_LIBRARY Static)

pybind11_add_module(rag_hpc src/bindings.cpp src/bm25_omp.cpp)
target_include_directories(rag_hpc PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_compile_options(rag_hpc PRIVATE -O3 -march=native)
target_link_libraries(rag_hpc PRIVATE rag_hpc_cuda OpenMP::OpenMP_CXX)
set_target_properties(rag_hpc PROPERTIES
  CUDA_RUNTIME_LIBRARY Static
  LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/..)
