cmake_minimum_required(VERSION 3.14)
project(puiann_bindings)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_definitions(-w)

find_package(pybind11 REQUIRED)
find_package(OpenMP REQUIRED)

set(index_include_dir ${CMAKE_SOURCE_DIR}/include)
set(index_source_dir ${CMAKE_SOURCE_DIR}/src)

include_directories(${index_include_dir})

execute_process(
    COMMAND python -m pybind11 --includes
    OUTPUT_VARIABLE pybind11_include_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

message(STATUS "pybind11 include directories: ${pybind11_include_dir}")

include_directories(${pybind11_include_dir})

file(GLOB_RECURSE index_source ${index_source_dir}/*.cpp)

add_library(puiann_bindings STATIC ${index_source})
target_compile_options(puiann_bindings PUBLIC -fPIC)

target_link_libraries(puiann_bindings PUBLIC OpenMP::OpenMP_CXX)
target_compile_options(puiann_bindings PUBLIC 
  -Ofast 
  -march=native 
  -mtune=native 
  -lrt
)

pybind11_add_module(puiann ${CMAKE_SOURCE_DIR}/puiann_bindings.cpp)

target_link_libraries(puiann PRIVATE puiann_bindings)

install(TARGETS puiann
    LIBRARY DESTINATION ${CMAKE_SOURCE_DIR}/modules
)
