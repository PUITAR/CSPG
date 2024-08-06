cmake_minimum_required(VERSION 3.12)

project(Experiment C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_definitions(-w)

set(index_root "${CMAKE_SOURCE_DIR}/../PuiANN")
set(index_include_dir "${index_root}/include")
set(index_source_dir "${index_root}/src")
include_directories(${index_include_dir})
file(GLOB_RECURSE index_source ${index_source_dir}/*.cpp)

find_package(OpenMP REQUIRED) 

add_subdirectory(overall)

add_subdirectory(partition)

add_subdirectory(redundancy)

add_subdirectory(stages)

add_subdirectory(scale)

add_subdirectory(waste)

add_subdirectory(hard)

add_subdirectory(grid)

add_subdirectory(distribution)