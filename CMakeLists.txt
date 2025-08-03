cmake_minimum_required(VERSION 3.25)
project(HEGraphEmbeddings LANGUAGES CXX CUDA)

# Set C++ and CUDA standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Options
option(BUILD_TESTS "Build test suite" ON)
option(BUILD_BENCHMARKS "Build benchmarks" ON)
option(BUILD_PYTHON_BINDINGS "Build Python bindings" ON)
option(ENABLE_GPU "Enable GPU support" ON)

# Find packages
find_package(CUDAToolkit REQUIRED)
find_package(OpenMP)

# Set CUDA architectures
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86 89 90)
endif()

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O3 -march=native")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math")

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -G -O0")
endif()

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/src/ckks
    ${CMAKE_SOURCE_DIR}/src/cuda
    ${CMAKE_SOURCE_DIR}/include
    ${CUDAToolkit_INCLUDE_DIRS}
)

# Source files
set(CKKS_SOURCES
    src/ckks/ckks_context.cpp
)

set(CUDA_SOURCES
    src/cuda/ckks_kernels.cu
)

# Create CKKS library
add_library(ckks_core STATIC ${CKKS_SOURCES})
target_link_libraries(ckks_core PUBLIC ${CMAKE_THREAD_LIBS_INIT})

if(OpenMP_CXX_FOUND)
    target_link_libraries(ckks_core PUBLIC OpenMP::OpenMP_CXX)
endif()

# Create CUDA library
if(ENABLE_GPU)
    add_library(ckks_cuda STATIC ${CUDA_SOURCES})
    target_link_libraries(ckks_cuda PUBLIC 
        CUDA::cudart
        CUDA::cufft
        CUDA::curand
    )
    set_target_properties(ckks_cuda PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON
    )
endif()

# Create main library
add_library(he_graph_embeddings SHARED)
target_link_libraries(he_graph_embeddings PUBLIC ckks_core)

if(ENABLE_GPU)
    target_link_libraries(he_graph_embeddings PUBLIC ckks_cuda)
endif()

# Python bindings
if(BUILD_PYTHON_BINDINGS)
    find_package(pybind11 CONFIG)
    if(pybind11_FOUND)
        pybind11_add_module(he_graph_cpp src/python/bindings.cpp)
        target_link_libraries(he_graph_cpp PRIVATE he_graph_embeddings)
    else()
        message(WARNING "pybind11 not found, Python bindings will not be built")
    endif()
endif()

# Tests
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Benchmarks
if(BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

# Installation
install(TARGETS he_graph_embeddings
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY src/ckks/
    DESTINATION include/he_graph
    FILES_MATCHING PATTERN "*.h"
)

# Package configuration
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/HEGraphEmbeddingsConfigVersion.cmake"
    VERSION 0.1.0
    COMPATIBILITY AnyNewerVersion
)

# Print configuration summary
message(STATUS "========================================")
message(STATUS "HE-Graph-Embeddings Configuration:")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "  Build tests: ${BUILD_TESTS}")
message(STATUS "  Build benchmarks: ${BUILD_BENCHMARKS}")
message(STATUS "  Python bindings: ${BUILD_PYTHON_BINDINGS}")
message(STATUS "  GPU support: ${ENABLE_GPU}")
message(STATUS "========================================")