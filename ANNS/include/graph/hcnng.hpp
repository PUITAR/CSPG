#pragma once

#include <vector_ops.hpp>

#include <random>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
// #include <unordered_map>
#include <unordered_set>
#include <map>
#include <cmath>
#include <queue>
#include <memory>
// #include <quantizer.hpp>

#ifdef _MSC_VER
#include <immintrin.h>
#else
#include <x86intrin.h>
#endif

#include <atomic>
#include <mutex>

namespace anns
{

  namespace graph
  {

    struct Edge
    {
      id_t src;
      id_t dst;
      float weight;

      bool operator<(const Edge &other) const
      {
        return this->weight < other.weight;
      }
    };

    template <typename vdim_t>
    class HCNNG
    {

    public:
      size_t max_elements_{0};
      size_t cur_element_count_{0};
      size_t data_size_{0};

      size_t D_;

      std::vector<const vdim_t*> data_memory_;
      std::vector<std::vector<id_t>> adj_memory_;

      bool ready_{false};

      size_t num_threads_{1};

      int random_seed_{100};

      // std::unique_ptr<VisitedListPool> visited_list_pool_{nullptr};

      std::unique_ptr<std::vector<std::mutex>> link_list_locks_{nullptr};

      std::atomic<size_t> comparison_{0};

      HCNNG(size_t D, size_t max_elements, int random_seed = 123);

      // HCNNG(
      //     const std::string &info_path,
      //     const std::string &data_path,
      //     const std::string &edge_path);

      // ~HCNNG();

      /// @brief Build Hcnng Index
      /// @param raw_data
      /// @param num_random_clusters
      /// @param min_size_clusters
      /// @param max_mst_degree
      void CreateHCNNG(
          const std::vector<vdim_t> &raw_data,
          size_t num_random_clusters,
          size_t min_size_clusters,
          size_t max_mst_degree);

      void CreateHCNNG(
          const std::vector<const vdim_t*> &raw_data,
          size_t num_random_clusters,
          size_t min_size_clusters,
          size_t max_mst_degree);

      bool Ready() const;

      size_t GetNumThreads() const;

      void SetNumThreads(size_t num_threads);

      void Search(
          const std::vector<std::vector<vdim_t>> &queries,
          size_t k,
          size_t ef,
          std::vector<std::vector<id_t>> &vids,
          std::vector<std::vector<float>> &dists);

      /// @brief get top-k nearest neighbors of query
      /// @param query
      /// @param k
      /// @param max_cal max vector to search
      /// @param result
      void Search(
          const vdim_t *query,
          size_t k,
          size_t ef,
          std::priority_queue<std::pair<float, id_t>> &result);

      void Search(
          const vdim_t *query,
          size_t k,
          size_t ef,
          id_t ep,
          std::priority_queue<std::pair<float, id_t>> &result);

      // void SaveInfo(const std::string &info_path);

      // void SaveEdges(const std::string &edge_path);

      // void SaveData(const std::string &data_path);

      /// @brief Prune the neighbors of each vertex to remain the knn
      /// @param max_neigh
      void PruneNeigh(size_t max_neigh);

      // void SaveHCNNG(const std::string &info_path, const std::string &edge_path, const std::string &data_path);

      size_t GetComparisonAndClear();

      size_t IndexSize() const;

      // void SaveEdgesTXT(const std::string &path);
      // void SaveGraphTxt()

      inline const vdim_t *GetDataByInternalID(id_t id) const;

      inline void WriteDataByInternalID(id_t id, const vdim_t * data_point);

      // #define Graph std::vector<std::vector<Edge>>

      // void LoadInfo(const std::string &info_path);

      // void LoadEdges(const std::string &edge_path);

      // void LoadData(const std::string &data_path);

      /// @brief Create exact MST graph
      /// @param idx_points
      /// @param left
      /// @param right
      /// @param max_mst_degree
      /// @return
      std::vector<std::vector<Edge>> CreateExactMST(
          const std::vector<id_t> &idx_points,
          size_t left, size_t right, size_t max_mst_degree);

      /// @brief Hierachical Clustering for ids in [left, right] from idx_points
      /// @param idx_points
      /// @param left
      /// @param right
      /// @param graph
      /// @param min_size_clusters
      /// @param max_mst_degree
      void CreateClusters(
          std::vector<id_t> &idx_points,
          size_t left, size_t right,
          size_t min_size_clusters,
          size_t max_mst_degree);

      id_t GetClosestPoint(const vdim_t *data_point);

      // std::vector<std::pair<float, id_t>> GetSearchLength(const vdim_t * query_data, size_t max_extend);

      std::vector<float> GetSearchLength(
          const vdim_t *query,
          size_t k,
          size_t ef,
          std::priority_queue<std::pair<float, id_t>> &result);

      std::vector<std::vector<float>> GetSearchLength(
          const std::vector<std::vector<vdim_t>> &queries,
          size_t k,
          size_t ef,
          std::vector<std::vector<id_t>> &vids,
          std::vector<std::vector<float>> &dists);
    };

    template class HCNNG<uint8_t>;

    template class HCNNG<float>;

    // template class HCNNG<int32_t>;

  } // namespace graph

} // namespace index
