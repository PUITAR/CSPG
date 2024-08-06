#ifndef INCLUDE_DISKANN_HPP
#define INCLUDE_DISKANN_HPP

// #include <index_status.hpp>
// #include <graph/visited_list_pool.hpp>

// #include <stdlib.h>

#include <vector_ops.hpp>
// #include <graph/visited_list_pool.hpp>
// #include <vector_ops.hpp>

#include <random>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <map>
#include <cmath>
#include <queue>
#include <memory>
#include <mutex>
// #include <quantizer.hpp>

#ifdef _MSC_VER
#include <immintrin.h>
#else
#include <x86intrin.h>
#endif

#include <atomic>

//

namespace puiann
{

  namespace graph
  {

    template <typename vdim_t>
    class DiskANN
    {
    public:
      size_t max_elements_{0};
      size_t cur_element_count_{0};
      size_t size_data_per_element_{0};
      size_t size_links_per_element_{0};
      size_t R_{0}; // Graph degree limit
      size_t data_size_{0};
      size_t D_{0}; // vector dimension
      size_t offset_data_{0};
      // std::unique_ptr<VisitedListPool> visited_list_pool_{nullptr};
      id_t enterpoint_node_{0};
      std::unique_ptr<std::vector<char>> data_memory_;
      int random_seed_{123};
      bool ready_{false};
      size_t num_threads_{1};
      std::mutex global_;
      std::unique_ptr<std::vector<std::mutex>> link_list_locks_;
      struct PHash
      {
        id_t operator()(const std::pair<float, id_t> &pr) const
        {
          return pr.second;
        }
      };
      std::atomic<size_t> comparison_{0};

      DiskANN(
          size_t D,
          size_t max_elements,
          size_t R = 16,
          int random_seed = 123);

      void BuildIndex(const std::vector<vdim_t> &raw_data, float alpha, size_t L);

      void BuildIndex(const std::vector<const vdim_t*> &raw_data, float alpha, size_t L);

      void Search(
          const vdim_t *query_data, size_t k, size_t L,
          std::vector<std::pair<float, id_t>> &visited);

      void Search(
          const vdim_t *query_data, size_t k, size_t L, id_t ep,
          std::vector<std::pair<float, id_t>> &visited);

      void Search(const std::vector<std::vector<vdim_t>> &queries, size_t k, size_t L, std::vector<std::vector<id_t>> &vids, std::vector<std::vector<float>> &dists);

      bool Ready();

      size_t GetNumThreads();

      void SetNumThreads(size_t num_threads);

      size_t GetComparisonAndClear();

      size_t IndexSize() const;

      inline const vdim_t *GetDataByInternalID(id_t id) const;

      inline void WriteDataByInternalID(id_t id, const vdim_t * data_point);

      inline char *GetLinkByInternalID(id_t id) const;

      void RobustPrune(id_t node_id, float alpha, std::vector<std::pair<float, id_t>> &candidates);

      id_t GetClosestPoint(const vdim_t *data_point);

      std::vector<float> GetSearchLength(const vdim_t *query_data, size_t k, size_t L,
                                         std::vector<std::pair<float, id_t>> &visited);

      std::vector<std::vector<float>> GetSearchLength(
          const std::vector<std::vector<vdim_t>> &queries, size_t k, size_t L,
          std::vector<std::vector<id_t>> &vids, std::vector<std::vector<float>> &dists);
    };

    template class DiskANN<uint8_t>;
    template class DiskANN<float>;
    // template class DiskANN<int32_t>;

  } // namespace graph

} // namespace index

#endif // INCLUDE_DISKANN_HPP