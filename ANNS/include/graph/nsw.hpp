#pragma once

// #include <index_status.hpp>
// #include <graph/visited_list_pool.hpp>
#include <vector_ops.hpp>

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

#include <atomic>

// #include <quantizer.hpp>

#ifdef _MSC_VER
#include <immintrin.h>
#else
#include <x86intrin.h>
#endif

namespace anns
{

    namespace graph
    {

        template <typename vdim_t>
        class NSW
        {

        public: // variables
            size_t max_elements_{0};
            size_t cur_element_count_{0};
            size_t size_data_per_element_{0};
            size_t size_links_per_element_{0};

            size_t M_{0}; // number of established connections, suggest let M between 8 and 32
            // size_t                              Mmax_{0}; // maximum number of connections for each element per layer
            size_t Mmax_{0}; // maximum number of connections for each element in layer0

            size_t ef_construction_{0}; // usually been set to 128

            // std::unique_ptr<VisitedListPool>    visited_list_pool_{nullptr};

            id_t enterpoint_node_{0};

            size_t offset_data_{0};

            std::vector<char> data_level0_memory_; // vector data start pointer of memory.

            size_t data_size_{0};
            size_t D_{0}; // vector dimensions

            int random_seed_{100};

            // const std::string                   graph_edges_{"graph_edges"};
            // const std::string                   quantizer_centers_{"_centers"};
            // const std::string                   enterpoints_file_{"_epids"};

            bool ready_{false};

            size_t num_threads_{1};

            std::mutex global_;
            std::unique_ptr<std::vector<std::mutex>> link_list_locks_;

            std::atomic<size_t> comparison_{0};

            
            NSW(
                size_t D,
                size_t max_elements,
                size_t M = 16, // [8, 32]
                size_t ef_construction = 128,
                size_t random_seed = 123);

            // NSW (
            //     const std::string & info_path,
            //     const std::string & data_path,
            //     const std::string & edge_path
            // );

            /// @brief Add point to hnsw index
            void AddPoint(const vdim_t *data_point);

            // void Train(const std::vector<vdim_t> & raw_data, size_t nsamples);

            void Populate(const std::vector<vdim_t> &raw_data);

            void Populate(const std::vector<const vdim_t *> &raw_data);

            std::priority_queue<std::pair<float, id_t>> Search(const vdim_t *query_data, size_t k, size_t ef);

            std::priority_queue<std::pair<float, id_t>> Search(const vdim_t *query_data, size_t k, size_t ef, id_t ep);

            void Search(const std::vector<std::vector<vdim_t>> &queries, size_t k, size_t ef, std::vector<std::vector<id_t>> &vids, std::vector<std::vector<float>> &dists);

            // void SaveInfo(const std::string & info_path);

            // void SaveEdges(const std::string & edge_path);

            // void SaveData(const std::string & data_path);

            bool Ready();

            size_t GetNumThreads();

            void SetNumThreads(size_t num_threads);

            // void SaveEdgesTXT(const std::string & path);

            size_t IndexSize() const;

            inline const vdim_t *GetDataByInternalID(id_t id) const;

            inline void WriteDataByInternalID(id_t id, const vdim_t *data_point);

            inline char *GetLinkByInternalID(id_t id) const;

            /// @brief Connection new element and return next cloest element id
            /// @param data_point
            /// @param id
            /// @param top_candidates
            /// @param layer
            /// @return
            void MutuallyConnectNewElement(
                const vdim_t *data_point,
                id_t id,
                std::priority_queue<std::pair<float, id_t>> &top_candidates);

            void GetNeighborsByHeuristic(std::priority_queue<std::pair<float, id_t>> &top_results, size_t NN);

            // int GetRandomLevel(double reverse_size); // typeof (level id) == int

            std::priority_queue<std::pair<float, id_t>> SearchBaseLayer(
                id_t ep_id,
                const vdim_t *data_point,
                size_t ef);

            // void LoadInfo(const std::string & info_path);

            // void LoadEdges(const std::string & edge_path);

            // void LoadData(const std::string & data_path);

            size_t GetComparisonAndClear();

            id_t GetClosestPoint(const vdim_t *data_point);
    
        };

        /// @brief Template Class

        template class NSW<float>;
        template class NSW<uint8_t>;

    }; // namespace graph

}; // namespace index
