cmake_minimum_required(VERSION 3.15)
project(shap_enhanced LANGUAGES CXX)

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# --- Optional override for Torch_DIR from environment ---
if(NOT Torch_DIR AND DEFINED ENV{TORCH_DIR})
    set(Torch_DIR "$ENV{TORCH_DIR}")
    message(STATUS "Using Torch_DIR from: ${Torch_DIR}")
endif()

# --- CMake settings ---
set(USE_CUDA OFF CACHE BOOL "Disable CUDA")
set(CAFFE2_USE_CUDA OFF CACHE BOOL "Disable Caffe2 CUDA")
set(TORCH_CUDA_PROFILE OFF CACHE BOOL "Disable CUDA profiling (kineto)")
set(CMAKE_SUPPRESS_DEVELOPER_WARNINGS 1 CACHE INTERNAL "No dev warnings")

# --- Dependencies ---
set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(Torch REQUIRED)

# --- Source paths ---
set(SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/shap_enhanced)
set(ALGORITHMS_SRC_DIR ${SOURCE_DIR}/algorithms/_cpp)
set(DATASETS_SRC_DIR ${SOURCE_DIR}/datasets/_cpp)

# --- Module lists ---
set(ALGORITHMS
    approximation
    attention
    distance_metrics
    integration
    interpolation
    masking
    perturbation
    sampling
    shapley_kernel
)

set(DATASETS
    synthetic_sequential
    synthetic_sparse
    synthetic_tabular
)

# --- Helper function ---
function(build_pybind_module module_name source_dir output_dir)
    set(SOURCE_FILE "${source_dir}/${module_name}.cpp")
    if(EXISTS ${SOURCE_FILE})
        pybind11_add_module(${module_name} ${SOURCE_FILE})
        set_target_properties(${module_name} PROPERTIES
            LIBRARY_OUTPUT_DIRECTORY "${output_dir}"
            OUTPUT_NAME "${module_name}"
        )
        target_link_libraries(${module_name} PRIVATE ${TORCH_LIBRARIES})
        target_include_directories(${module_name} PRIVATE ${TORCH_INCLUDE_DIRS})
    else()
        message(WARNING "Source file for ${module_name} not found at ${SOURCE_FILE}")
    endif()
endfunction()

# --- Build algorithm modules ---
foreach(module IN LISTS ALGORITHMS)
    build_pybind_module(${module} ${ALGORITHMS_SRC_DIR} "${SOURCE_DIR}/algorithms")
endforeach()

# --- Build dataset modules ---
foreach(module IN LISTS DATASETS)
    build_pybind_module(${module} ${DATASETS_SRC_DIR} "${SOURCE_DIR}/datasets")
endforeach()
