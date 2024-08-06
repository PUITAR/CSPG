#ifndef INCLUDE_BINARY_IO_HPP
#define INCLUDE_BINARY_IO_HPP

#include <fstream>
#include <vector>
#include <utility>
#include <iostream>
#include <cassert>

namespace utils {


/// @brief Write variable ref to file in binary type.
/// @tparam T 
/// @param out 
/// @param ref  
template <typename T> static void WriteBinary(std::ofstream & out, const T & ref)
{
    if (! out.is_open()) {
        std::cout << "Error: file not open." << std::endl;
        exit(1);
    }
    out.write((char *) &ref, sizeof(T));
}


/// @brief Read variable ref from file in binary type.
/// @tparam T 
/// @param in 
/// @param ref  
template <typename T> static void ReadBinary(std::ifstream & in, T & ref)
{
    if (! in.is_open()) {
        std::cout << "Error: file not open." << std::endl;
        exit(1);
    }
    in.read((char *) &ref, sizeof(T));
}


template <typename T> static void WriteBinary(std::ofstream & out, const T * buffer, size_t n)
{
    if (! out.is_open()) {
        std::cout << "Error: file not open." << std::endl;
        exit(1);
    }
    out.write((char *) buffer, sizeof(T) * n);
}


template <typename T> static void ReadBinary(std::ifstream & in, T * buffer, size_t n)
{
    if (! in.is_open()) {
        std::cout << "Error: file not open." << std::endl;
        exit(1);
    }
    in.read((char *) buffer, sizeof(T) * n);
}


//////////////////////////////////////
/// @brief Some Vector IO Funtions ///
//////////////////////////////////////

// return the dimention of corresponding dataset
template <typename T> std::pair<size_t, size_t> LoadFromFile(std::vector<T>& data, const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        throw;
    }

    int D;
    file.read(reinterpret_cast<char*>(&D), sizeof(int));
    
    file.seekg(0, std::ios::end);
    size_t file_size = file.tellg();
    size_t N = (file_size) / ((D) * sizeof(T) + sizeof(int));

    file.seekg(0, std::ios::beg); 
    data.resize(N * D);
    data.shrink_to_fit();

    int sep;
    for (size_t n = 0; n < N; ++n) {
        file.read(reinterpret_cast<char*>(&sep), sizeof(int));
        file.read(reinterpret_cast<char*>(data.data() + n * D), D * sizeof(T));
    }
    // printf("%s: [%zu x %d] has loaded!\n", filename.data(), N, D);
    file.close();
    return {N, D};
}



template<typename T> std::pair<size_t, size_t> LoadFromFile(std::vector<T>& data, const std::string& filename, size_t expect_read_n) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        throw;
    }

    int D;
    file.read(reinterpret_cast<char*>(&D), sizeof(int));
    
    file.seekg(0, std::ios::end);
    size_t file_size = file.tellg();
    size_t N = (file_size) / ((D) * sizeof(T) + sizeof(int));

    assert(expect_read_n <= N);

    file.seekg(0, std::ios::beg); 
    data.resize(expect_read_n * D);
    data.shrink_to_fit();

    int sep;
    for (size_t n = 0; n < expect_read_n; ++n) {
        file.read(reinterpret_cast<char*>(&sep), sizeof(int));
        file.read(reinterpret_cast<char*>(data.data() + n * D), D * sizeof(T));
    }
    // printf("All %s: [%zu x %d] has loaded!\n", filename.data(), expect_read_n, D);
    file.close();
    return {expect_read_n, D};
}

template<typename T> std::pair<size_t, size_t> LoadFromFileBin(std::vector<T>& data, const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        throw;
    }

    uint32_t D, N;
    file.read(reinterpret_cast<char*>(&N), sizeof(uint32_t));
    file.read(reinterpret_cast<char*>(&D), sizeof(uint32_t));
    
    data.resize(N * D);
    data.shrink_to_fit();

    file.read(reinterpret_cast<char*>(data.data()), N * D * sizeof(T));

    // printf("All %s: [%zu x %d] has loaded!\n", filename.data(), expect_read_n, D);
    file.close();
    return {N, D};
}

template<typename T> void WriteToFile(const std::vector<T>& data, std::pair<size_t, size_t> dimension, const std::string& filename) {
    std::ofstream file(filename, std::ios::binary | std::ios::trunc);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        throw;
    }
    auto [N, D] = dimension;
    assert(data.size() == N * D);

    // char sep[4] = {(char)D, 0, 0, 0};
    int sep = D;
    // 4 + d * sizeof(T) for each vector
    for (size_t n = 0; n < N; ++n) {
        file.write(reinterpret_cast<char*>(&sep), sizeof(int));
        file.write(reinterpret_cast<char*>(const_cast<T*>(data.data()) + n * D), D * sizeof(T));
    }
    // printf("%s: [%zu x %zu] has written!\n", filename.data(), N, D);
    file.close();
}


}; // namespace utils


#endif