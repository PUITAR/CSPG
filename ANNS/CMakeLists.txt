cmake_minimum_required(VERSION 3.14)
project(anns_bindings)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_definitions(-w)

find_package(pybind11 REQUIRED)
find_package(OpenMP REQUIRED)

set(index_include_dir ${CMAKE_SOURCE_DIR}/include)

include_directories(${index_include_dir})

execute_process(
    COMMAND python -m pybind11 --includes
    OUTPUT_VARIABLE pybind11_include_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

message(STATUS "pybind11 include directories: ${pybind11_include_dir}")

include_directories(${pybind11_include_dir})

file(GLOB_RECURSE index_source ${index_include_dir}/*.cpp)

add_library(anns_bindings STATIC ${index_source})
target_compile_options(anns_bindings PUBLIC -fPIC)

target_link_libraries(anns_bindings PUBLIC OpenMP::OpenMP_CXX)
target_compile_options(anns_bindings PUBLIC 
  -Ofast 
  -march=native 
  -mtune=native 
  -lrt
)

pybind11_add_module(anns ${CMAKE_SOURCE_DIR}/anns_bindings.cpp)

target_link_libraries(anns PRIVATE anns_bindings)

install(TARGETS anns
    LIBRARY DESTINATION ${CMAKE_SOURCE_DIR}/modules
)
