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

#include <utils/binary_io.hpp>
#include <utils/stimer.hpp>

#include <algorithm>
#include <stdexcept>

#include <omp.h>

#ifdef _MSC_VER
#include <immintrin.h>
#else
#include <x86intrin.h>
#endif

namespace puiann
{

  namespace graph
  {

    template <typename vdim_t>
    class HNSW
    {

    public:
      size_t max_elements_{0};
      size_t cur_element_count_{0};
      size_t size_data_per_element_{0};
      size_t size_links_per_element_{0};

      size_t M_{0};     // number of established connections, suggest let M between 8 and 32
      size_t Mmax_{0};  // maximum number of connections for each element per layer
      size_t Mmax0_{0}; // maximum number of connections for each element in layer0

      size_t ef_construction_{0}; // usually been set to 128

      double mult_{0.0};
      double rev_size_{0.0};
      int max_level_{0};

      // std::unique_ptr<VisitedListPool> visited_list_pool_{nullptr};

      std::mutex global_;
      std::unique_ptr<std::vector<std::mutex>> link_list_locks_;

      id_t enterpoint_node_{0};

      size_t size_links_level0_{0};
      size_t offset_data_{0};

      std::vector<char> data_level0_memory_; // vector data start pointer of memory.
      std::vector<std::vector<char>> link_lists_;
      std::vector<int> element_levels_; // keeps level of each element

      size_t data_size_{0};
      size_t D_{0}; // vector dimensions

      std::default_random_engine level_generator_;
      int random_seed_{100};

      bool ready_{false};

      size_t num_threads_{1};

      // bool mlock_{false};

      std::atomic<size_t> comparison_{0};

      HNSW(
          size_t D,
          size_t max_elements,
          size_t M = 16, // [8, 32]
          size_t ef_construction = 128,
          size_t random_seed = 123) :

                                      D_(D), max_elements_(max_elements), M_(M), Mmax_(M), Mmax0_(2 * M),
                                      ef_construction_(std::max(ef_construction, M)), random_seed_(random_seed), element_levels_(max_elements)
      {
        // random seed
        level_generator_.seed(random_seed);

        data_size_ = D * sizeof(vdim_t);
        size_links_level0_ = Mmax0_ * sizeof(id_t) + sizeof(size_t);
        size_data_per_element_ = size_links_level0_ + sizeof(vdim_t *);
        offset_data_ = size_links_level0_;
        // offset_level0_ = 0;

        // std::cout << "Size GB: " << (1.0 * max_elements_ * size_data_per_element_ / (1024 * 1024 * 1024)) << std::endl;
        data_level0_memory_.resize(max_elements_ * size_data_per_element_, 0x00);

        cur_element_count_ = 0;

        // visited_list_pool_ = std::make_unique<VisitedListPool>(1, max_elements_);

        // initializations for special treatment of the first node
        enterpoint_node_ = -1;
        max_level_ = -1;

        link_lists_.resize(max_elements);
        link_list_locks_ = std::make_unique<std::vector<std::mutex>>(max_elements_);
        element_levels_.resize(max_elements, -1);

        size_links_per_element_ = Mmax_ * sizeof(id_t) + sizeof(size_t);

        mult_ = 1 / log(1.0 * M_);
        rev_size_ = 1.0 / mult_;
      }

      /// @brief Add point to hnsw index
      void AddPoint(const vdim_t *data_point)
      {
        if (cur_element_count_ >= max_elements_)
        {
          std::cerr << "The number of elements exceeds the specified limit" << std::endl;
          exit(1);
        }

        id_t cur_id;
        {
          std::unique_lock<std::mutex> temp_lock(global_);
          cur_id = cur_element_count_++;
        }

        std::unique_lock<std::mutex> lock_el((*link_list_locks_)[cur_id]);

        int cur_level = GetRandomLevel(mult_);

        element_levels_[cur_id] = cur_level;

        std::unique_lock<std::mutex> temp_lock(global_);
        int max_level_copy = max_level_;
        id_t cur_obj = enterpoint_node_;
        id_t enterpoint_node_copy = enterpoint_node_;
        if (cur_level <= max_level_)
          temp_lock.unlock();

        // Clear edge-slot and copy vector into graph buffer.
        WriteDataByInternalID(cur_id, data_point);

        if (cur_level)
        {
          link_lists_[cur_id].resize(size_links_per_element_ * cur_level, 0x00);
        }

        if (enterpoint_node_copy != -1)
        { // not first element
          if (cur_level < max_level_copy)
          {
            // find the closet node in upper layers
            float cur_dist = vec_L2sqr(data_point, GetDataByInternalID(cur_obj), D_);
            for (int lev = max_level_copy; lev > cur_level; lev--)
            {
              bool changed = true;
              while (changed)
              {
                changed = false;
                std::unique_lock<std::mutex> wlock(link_list_locks_->at(cur_obj));
                size_t *ll_cur = (size_t *)GetLinkByInternalID(cur_obj, lev);
                size_t num_neighbors = *ll_cur;
                id_t *neighbors = (id_t *)(ll_cur + 1);

                for (size_t i = 0; i < num_neighbors; i++)
                {
                  id_t cand = neighbors[i];
                  if (cand < 0 || cand > max_elements_)
                  {
                    std::cerr << "cand error" << std::endl;
                    exit(1);
                  }
                  float d = vec_L2sqr(data_point, GetDataByInternalID(cand), D_);
                  if (d < cur_dist)
                  {
                    cur_dist = d;
                    cur_obj = cand;
                    changed = true;
                  }
                }
              }
            }
          }
          /// add edges to lower layers from the closest node
          for (int lev = std::min(cur_level, max_level_copy); lev >= 0; lev--)
          {
            auto top_candidates = SearchBaseLayer(cur_obj, data_point, lev, ef_construction_);
            cur_obj = MutuallyConnectNewElement(data_point, cur_id, top_candidates, lev);
          }
        }
        else
        {
          // Do nothing for the first element
          enterpoint_node_ = 0;
          max_level_ = cur_level;
        }

        // Releasing lock for the maximum level
        if (cur_level > max_level_copy)
        {
          enterpoint_node_ = cur_id;
          max_level_ = cur_level;
        }
      }

      void Populate(const std::vector<vdim_t> &raw_data)
      {
        size_t N = raw_data.size() / D_;
        assert(N <= max_elements_ && "data size too large!");

#pragma omp parallel for schedule(dynamic, 1) num_threads(num_threads_)
        for (size_t i = 0; i < N; i++)
        {
          AddPoint(raw_data.data() + i * D_);
          if (i == 0) {
                    std::ofstream fout("/home/dbcloud/ym/CSPG/experiment/distribution/a.txt");
        fout << "In AddPoint " << omp_get_num_threads() << std::endl;
          }
        }

        ready_ = true;
      }

      void Populate(const std::vector<const vdim_t *> &raw_data)
      {
        size_t N = raw_data.size();
        assert(N <= max_elements_ && "data size too large!");

#pragma omp parallel for schedule(dynamic, 1) num_threads(num_threads_)
        for (size_t i = 0; i < N; i++) {
           AddPoint(raw_data[i]);
            if (i == 0) {
                    std::ofstream fout("/home/dbcloud/ym/CSPG/experiment/distribution/a.txt");
        fout << "In AddPoint " << omp_get_num_threads() << std::endl;
          }
        }
         
    
        ready_ = true;
      }

      std::priority_queue<std::pair<float, id_t>> Search(const vdim_t *query_data, size_t k, size_t ef)
      {
        assert(ready_ && "Index uninitialized!");

        assert(ef >= k && "ef > k!");

        if (cur_element_count_ == 0)
          return std::priority_queue<std::pair<float, id_t>>();

        size_t comparison = 0;

        id_t cur_obj = enterpoint_node_;
        float cur_dist = vec_L2sqr(query_data, GetDataByInternalID(enterpoint_node_), D_);
        comparison++;

        for (int lev = max_level_; lev > 0; lev--)
        {
          // find the closet node in upper layers
          bool changed = true;
          while (changed)
          {
            changed = false;
            size_t *ll_cur = (size_t *)GetLinkByInternalID(cur_obj, lev);
            size_t num_neighbors = *ll_cur;
            id_t *neighbors = (id_t *)(ll_cur + 1);

            for (size_t i = 0; i < num_neighbors; i++)
            {
              id_t cand = neighbors[i];
              if (cand < 0 || cand > max_elements_)
              {
                std::cerr << "cand error" << std::endl;
                exit(1);
              }
              float d = vec_L2sqr(query_data, GetDataByInternalID(cand), D_);
              if (d < cur_dist)
              {
                cur_dist = d;
                cur_obj = cand;
                changed = true;
              }
            }
            comparison += num_neighbors;
          }
        }

        auto top_candidates = SearchBaseLayer(cur_obj, query_data, 0, ef);

        while (top_candidates.size() > k)
        {
          top_candidates.pop();
        }

        comparison_.fetch_add(comparison);

        return top_candidates;
      }

      std::priority_queue<std::pair<float, id_t>> Search(const vdim_t *query_data, size_t k, size_t ef, id_t ep)
      {
        assert(ready_ && "Index uninitialized!");

        assert(ef >= k && "ef > k!");

        if (cur_element_count_ == 0)
          return std::priority_queue<std::pair<float, id_t>>();

        size_t comparison = 0;

        id_t cur_obj = ep;
        float cur_dist = vec_L2sqr(query_data, GetDataByInternalID(enterpoint_node_), D_);
        comparison++;

        for (int lev = element_levels_[ep]; lev > 0; lev--)
        {
          // find the closet node in upper layers
          bool changed = true;
          while (changed)
          {
            changed = false;
            size_t *ll_cur = (size_t *)GetLinkByInternalID(cur_obj, lev);
            size_t num_neighbors = *ll_cur;
            id_t *neighbors = (id_t *)(ll_cur + 1);

            for (size_t i = 0; i < num_neighbors; i++)
            {
              id_t cand = neighbors[i];
              if (cand < 0 || cand > max_elements_)
              {
                std::cerr << "cand error" << std::endl;
                exit(1);
              }
              float d = vec_L2sqr(query_data, GetDataByInternalID(cand), D_);
              if (d < cur_dist)
              {
                cur_dist = d;
                cur_obj = cand;
                changed = true;
              }
            }
            comparison += num_neighbors;
          }
        }

        auto top_candidates = SearchBaseLayer(cur_obj, query_data, 0, ef);

        while (top_candidates.size() > k)
        {
          top_candidates.pop();
        }

        comparison_.fetch_add(comparison);

        return top_candidates;
      }

      void Search(const std::vector<std::vector<vdim_t>> &queries, size_t k, size_t ef, std::vector<std::vector<id_t>> &vids, std::vector<std::vector<float>> &dists)
      {
        size_t nq = queries.size();
        vids.clear();
        dists.clear();
        vids.resize(nq);
        dists.resize(nq);

#pragma omp parallel for schedule(dynamic, 1) num_threads(num_threads_)
        for (size_t i = 0; i < nq; i++)
        {
          const auto &query = queries[i];
          auto &vid = vids[i];
          auto &dist = dists[i];

          auto r = Search(query.data(), k, ef);
          vid.reserve(r.size());
          dist.reserve(r.size());
          while (r.size())
          {
            const auto &te = r.top();
            vid.emplace_back(te.second);
            dist.emplace_back(te.first);
            r.pop();
          }
        }
      }

      bool Ready() { return ready_; }

      size_t GetNumThreads()
      {
        return num_threads_;
      }

      void SetNumThreads(size_t num_threads)
      {
        num_threads_ = num_threads;
      }

      void SetReady(bool ready) { ready_ = ready; }

      size_t GetComparisonAndClear()
      {
        return comparison_.exchange(0);
      }

      size_t IndexSize() const
      {
        size_t sz = 0;
        sz += data_level0_memory_.size() * sizeof(char);
        std::for_each(link_lists_.begin(), link_lists_.end(), [&](const std::vector<char> &bytes_arr)
                      { sz += bytes_arr.size() * sizeof(char); });
        // element levels
        sz += cur_element_count_ * sizeof(int);
        return sz;
      }

      inline const vdim_t *GetDataByInternalID(id_t id) const
      {
        return *((vdim_t **)(data_level0_memory_.data() + id * size_data_per_element_ + offset_data_));
      }

      inline void WriteDataByInternalID(id_t id, const vdim_t *data_point)
      {
        *((const vdim_t **)(data_level0_memory_.data() + id * size_data_per_element_ + offset_data_)) = data_point;
      }

      inline char *GetLinkByInternalID(id_t id, int level) const
      {
        if (level > 0)
          return (char *)(link_lists_[id].data() + (level - 1) * size_links_per_element_);

        return (char *)(data_level0_memory_.data() + id * size_data_per_element_);
      }

      /// @brief Connection new element and return next cloest element id
      /// @param data_point
      /// @param id
      /// @param top_candidates
      /// @param layer
      /// @return
      id_t MutuallyConnectNewElement(
          const vdim_t *data_point,
          id_t id,
          std::priority_queue<std::pair<float, id_t>> &top_candidates,
          int level)
      {
        size_t Mcurmax = level ? Mmax_ : Mmax0_;

        GetNeighborsByHeuristic(top_candidates, M_);

        std::vector<id_t> selected_neighbors;
        selected_neighbors.reserve(M_);
        while (top_candidates.size())
        {
          selected_neighbors.emplace_back(top_candidates.top().second);
          top_candidates.pop();
        }

        id_t next_closet_entry_point = selected_neighbors.back();

        /// @brief Edge-slots check and Add neighbors for current vector
        {
          // lock only during the update
          // because during the addition the lock for cur_c is already acquired
          std::unique_lock<std::mutex> lock((*link_list_locks_)[id], std::defer_lock);
          size_t *ll_cur = (size_t *)GetLinkByInternalID(id, level);
          size_t num_neighbors = *ll_cur;

          if (num_neighbors)
          {
            std::cerr << "The newly inserted element should have blank link list" << std::endl;
            exit(1);
          }

          *ll_cur = selected_neighbors.size();

          id_t *neighbors = (id_t *)(ll_cur + 1);
          for (size_t i = 0; i < selected_neighbors.size(); i++)
          {
            if (neighbors[i])
            {
              std::cerr << "Possible memory corruption" << std::endl;
              exit(1);
            }
            if (level > element_levels_[selected_neighbors[i]])
            {
              std::cerr << "Trying to make a link on a non-existent level" << std::endl;
              exit(1);
            }

            neighbors[i] = selected_neighbors[i];
          }
        }

        for (size_t i = 0; i < selected_neighbors.size(); i++)
        {
          std::unique_lock<std::mutex> lock((*link_list_locks_)[selected_neighbors[i]]);

          size_t *ll_other = (size_t *)GetLinkByInternalID(selected_neighbors[i], level);
          size_t sz_link_list_other = *ll_other;

          if (sz_link_list_other > Mcurmax || sz_link_list_other < 0)
          {
            std::cerr << "Bad value of sz_link_list_other" << std::endl;
            exit(1);
          }
          if (selected_neighbors[i] == id)
          {
            std::cerr << "Trying to connect an element to itself" << std::endl;
            exit(1);
          }
          if (level > element_levels_[selected_neighbors[i]])
          {
            std::cerr << "Trying to make a link on a non-existent level" << std::endl;
            exit(1);
          }

          id_t *neighbors = (id_t *)(ll_other + 1);

          if (sz_link_list_other < Mcurmax)
          {
            neighbors[sz_link_list_other] = id;
            *ll_other = sz_link_list_other + 1;
          }
          else
          {
            // finding the "farest" element to replace it with the new one
            float d_max = vec_L2sqr(GetDataByInternalID(id), GetDataByInternalID(selected_neighbors[i]), D_);
            // Heuristic:
            std::priority_queue<std::pair<float, id_t>> candidates;
            candidates.emplace(d_max, id);

            for (size_t j = 0; j < sz_link_list_other; j++)
            {
              candidates.emplace(vec_L2sqr(GetDataByInternalID(neighbors[j]), GetDataByInternalID(selected_neighbors[i]), D_), neighbors[j]);
            }

            GetNeighborsByHeuristic(candidates, Mcurmax);

            // Copy neighbors and add edges.
            size_t nn = 0;
            while (candidates.size())
            {
              neighbors[nn] = candidates.top().second;
              candidates.pop();
              nn++;
            }
            *ll_other = nn;
          }
        }

        return next_closet_entry_point;
      }

      void GetNeighborsByHeuristic(std::priority_queue<std::pair<float, id_t>> &top_candidates, size_t NN)
      {
        if (top_candidates.size() < NN)
        {
          return;
        }

        std::priority_queue<std::pair<float, id_t>> queue_closest; // min heap
        std::vector<std::pair<float, id_t>> return_list;

        while (top_candidates.size())
        { // replace top_candidates with a min-heap, so that each poping can return the nearest neighbor.
          const auto &te = top_candidates.top();
          queue_closest.emplace(-te.first, te.second);
          top_candidates.pop();
        }

        while (queue_closest.size())
        {
          if (return_list.size() >= NN)
          {
            break;
          }

          const auto curen = queue_closest.top();
          float dist2query = -curen.first;
          const vdim_t *curenv = GetDataByInternalID(curen.second);
          queue_closest.pop();
          bool good = true;
          for (const auto &curen2 : return_list)
          {
            float dist2curenv2 = vec_L2sqr(GetDataByInternalID(curen2.second), curenv, D_);

            if (dist2curenv2 < dist2query)
            {
              good = false;
              break;
            }
          }
          if (good)
          {
            return_list.emplace_back(curen);
          }
        }

        for (const auto &elem : return_list)
        {
          top_candidates.emplace(-elem.first, elem.second);
        }
      }

      int GetRandomLevel(double reverse_size)
      {
        std::uniform_real_distribution<double> distribution(0.0, 1.0);
        double r = -log(distribution(level_generator_)) * reverse_size;
        return (int)r;
      }

      std::priority_queue<std::pair<float, id_t>> SearchBaseLayer(
          id_t ep_id,
          const vdim_t *data_point,
          int level,
          size_t ef)
      {
        size_t comparison = 0;

        // auto vl = visited_list_pool_->GetFreeVisitedList();
        // auto mass_visited = vl->mass_.data();
        // auto curr_visited = vl->curr_visited_;

        auto mass_visited = std::make_unique<std::vector<bool>>(max_elements_, false);

        std::priority_queue<std::pair<float, id_t>> top_candidates;
        std::priority_queue<std::pair<float, id_t>> candidate_set;

        float dist = vec_L2sqr(data_point, GetDataByInternalID(ep_id), D_);
        comparison++;

        top_candidates.emplace(dist, ep_id); // max heap
        candidate_set.emplace(-dist, ep_id); // min heap
        // mass_visited[ep_id] = curr_visited;
        mass_visited->at(ep_id) = true;

        /// @brief Branch and Bound Algorithm
        float low_bound = dist;
        while (candidate_set.size())
        {
          auto curr_el_pair = candidate_set.top();
          if (-curr_el_pair.first > low_bound && top_candidates.size() == ef)
            break;

          candidate_set.pop();
          id_t curr_node_id = curr_el_pair.second;

          std::unique_lock<std::mutex> lock((*link_list_locks_)[curr_node_id]);

          size_t *ll_cur = (size_t *)GetLinkByInternalID(curr_node_id, level);
          size_t num_neighbors = *ll_cur;
          id_t *neighbors = (id_t *)(ll_cur + 1);

          // #if defined(__SSE__)
          //   /// @brief Prefetch cache lines to speed up cpu caculation.
          //   _mm_prefetch((char *) (mass_visited + *neighbors), _MM_HINT_T0);
          //   _mm_prefetch((char *) (mass_visited + *neighbors + 64), _MM_HINT_T0);
          //   _mm_prefetch((char *) (GetDataByInternalID(*neighbors)), _MM_HINT_T0);
          // #endif

          for (size_t j = 0; j < num_neighbors; j++)
          {
            id_t neighbor_id = neighbors[j];

            // #if defined(__SSE__)
            //   _mm_prefetch((char *) (mass_visited + *(neighbors + j + 1)), _MM_HINT_T0);
            //   _mm_prefetch((char *) (GetDataByInternalID(*(neighbors + j + 1))), _MM_HINT_T0);
            // #endif

            if (mass_visited->at(neighbor_id) == false)
            {
              // mass_visited[neighbor_id] = curr_visited;
              mass_visited->at(neighbor_id) = true;

              float dist = vec_L2sqr(data_point, GetDataByInternalID(neighbor_id), D_);
              comparison++;

              /// @brief If neighbor is closer than farest vector in top result, and result.size still less than ef
              if (top_candidates.top().first > dist || top_candidates.size() < ef)
              {
                candidate_set.emplace(-dist, neighbor_id);
                top_candidates.emplace(dist, neighbor_id);

                // #if defined(__SSE__)
                //   _mm_prefetch((char *) (GetLinkByInternalID(candidate_set.top().second, 0)), _MM_HINT_T0);
                // #endif

                if (top_candidates.size() > ef) // give up farest result so far
                  top_candidates.pop();

                if (top_candidates.size())
                  low_bound = top_candidates.top().first;
              }
            }
          }
        }
        // visited_list_pool_->ReleaseVisitedList(vl);
        comparison_.fetch_add(comparison);

        return top_candidates;
      }

      id_t GetClosestPoint(const vdim_t *data_point)
      {
        if (cur_element_count_ == 0)
          throw std::runtime_error("empty graph");
        id_t wander = enterpoint_node_;
        size_t comparison = 0;

        float dist = vec_L2sqr(data_point, GetDataByInternalID(wander), D_);
        comparison++;

        for (int lev = max_level_; lev > 0; lev--)
        {
          bool moving = true;
          while (moving)
          {
            moving = false;
            size_t *ll = (size_t *)GetLinkByInternalID(wander, lev);
            size_t sz = *ll;
            id_t *adj = (id_t *)(ll + 1);
            for (size_t i = 0; i < sz; i++)
            {
              id_t cand = adj[i];
              float d = vec_L2sqr(data_point, GetDataByInternalID(cand), D_);
              if (d < dist)
              {
                dist = d;
                wander = cand;
                moving = true;
              }
            }
            comparison += sz;
          }
        }

        comparison_.fetch_add(comparison);

        return wander;
      }

      std::vector<float> GetSearchLengthLevel0(id_t ep_id, const vdim_t *query_data, size_t k, size_t ef, std::priority_queue<std::pair<float, id_t>> &top_candidates)
      {
        std::vector<float> length;

        size_t comparison = 0;

        // auto vl = visited_list_pool_->GetFreeVisitedList();
        // auto mass_visited = vl->mass_.data();
        // auto curr_visited = vl->curr_visited_;

        auto mass_visited = std::make_unique<std::vector<bool>>(max_elements_, false);

        // std::priority_queue<std::pair<float, id_t>> top_candidates;
        std::priority_queue<std::pair<float, id_t>> candidate_set;

        float dist = vec_L2sqr(query_data, GetDataByInternalID(ep_id), D_);
        comparison++;

        top_candidates.emplace(dist, ep_id); // max heap
        candidate_set.emplace(-dist, ep_id); // min heap
        // mass_visited[ep_id] = curr_visited;
        mass_visited->at(ep_id) = true;

        /// @brief Branch and Bound Algorithm
        float low_bound = dist;
        while (candidate_set.size())
        {
          auto curr_el_pair = candidate_set.top();

          length.emplace_back(-curr_el_pair.first);

          if (-curr_el_pair.first > low_bound && top_candidates.size() == ef)
            break;

          candidate_set.pop();
          id_t curr_node_id = curr_el_pair.second;

          std::unique_lock<std::mutex> lock((*link_list_locks_)[curr_node_id]);

          size_t *ll_cur = (size_t *)GetLinkByInternalID(curr_node_id, 0);
          size_t num_neighbors = *ll_cur;
          id_t *neighbors = (id_t *)(ll_cur + 1);

          // #if defined(__SSE__)
          //   /// @brief Prefetch cache lines to speed up cpu caculation.
          //   _mm_prefetch((char *) (mass_visited + *neighbors), _MM_HINT_T0);
          //   _mm_prefetch((char *) (mass_visited + *neighbors + 64), _MM_HINT_T0);
          //   _mm_prefetch((char *) (GetDataByInternalID(*neighbors)), _MM_HINT_T0);
          // #endif

          for (size_t j = 0; j < num_neighbors; j++)
          {
            id_t neighbor_id = neighbors[j];

            // #if defined(__SSE__)
            //   _mm_prefetch((char *) (mass_visited + *(neighbors + j + 1)), _MM_HINT_T0);
            //   _mm_prefetch((char *) (GetDataByInternalID(*(neighbors + j + 1))), _MM_HINT_T0);
            // #endif

            if (mass_visited->at(neighbor_id) == false)
            {
              mass_visited->at(neighbor_id) = true;

              float dist = vec_L2sqr(query_data, GetDataByInternalID(neighbor_id), D_);
              comparison++;

              /// @brief If neighbor is closer than farest vector in top result, and result.size still less than ef
              if (top_candidates.top().first > dist || top_candidates.size() < ef)
              {
                candidate_set.emplace(-dist, neighbor_id);
                top_candidates.emplace(dist, neighbor_id);

                // #if defined(__SSE__)
                //   _mm_prefetch((char *) (GetLinkByInternalID(candidate_set.top().second, 0)), _MM_HINT_T0);
                // #endif

                if (top_candidates.size() > ef) // give up farest result so far
                  top_candidates.pop();

                if (top_candidates.size())
                  low_bound = top_candidates.top().first;
              }
            }
          }
        }
        // visited_list_pool_->ReleaseVisitedList(vl);
        comparison_.fetch_add(comparison);

        return length;
      }

      std::vector<std::vector<float>> GetSearchLength(
          const std::vector<std::vector<vdim_t>> &queries, size_t k, size_t ef, std::vector<std::vector<id_t>> &vids, std::vector<std::vector<float>> &dists)
      {
        size_t nq = queries.size();
        vids.clear();
        dists.clear();
        vids.resize(nq);
        dists.resize(nq);

        std::vector<std::vector<float>> lengths(nq);

#pragma omp parallel for schedule(dynamic, 1) num_threads(num_threads_)
        for (size_t i = 0; i < nq; i++)
        {
          const auto &query = queries[i];
          auto &vid = vids[i];
          auto &dist = dists[i];

          std::priority_queue<std::pair<float, id_t>> r;

          lengths[i] = GetSearchLength(query.data(), k, ef, r);
          vid.reserve(r.size());
          dist.reserve(r.size());
          while (r.size())
          {
            const auto &te = r.top();
            vid.emplace_back(te.second);
            dist.emplace_back(te.first);
            r.pop();
          }
        }

        return lengths;
      }

      std::vector<float> GetSearchLength(const vdim_t *query_data, size_t k, size_t ef, std::priority_queue<std::pair<float, id_t>> &top_candidates)
      {
        assert(ready_ && "Index uninitialized!");

        assert(ef >= k && "ef > k!");

        if (cur_element_count_ == 0)
          return std::vector<float>();

        std::vector<float> length;

        size_t comparison = 0;

        id_t cur_obj = enterpoint_node_;
        float cur_dist = vec_L2sqr(query_data, GetDataByInternalID(enterpoint_node_), D_);
        comparison++;

        for (int lev = max_level_; lev > 0; lev--)
        {
          // find the closet node in upper layers
          if (length.size())
            length.pop_back();
          bool changed = true;
          while (changed)
          {
            length.emplace_back(cur_dist);
            changed = false;
            size_t *ll_cur = (size_t *)GetLinkByInternalID(cur_obj, lev);
            size_t num_neighbors = *ll_cur;
            id_t *neighbors = (id_t *)(ll_cur + 1);

            for (size_t i = 0; i < num_neighbors; i++)
            {
              id_t cand = neighbors[i];
              float d = vec_L2sqr(query_data, GetDataByInternalID(cand), D_);
              if (d < cur_dist)
              {
                cur_dist = d;
                cur_obj = cand;
                changed = true;
              }
            }
            comparison += num_neighbors;
          }
        }

        // std::priority_queue<std::pair<float, id_t>> top_candidates;

        auto length0 = GetSearchLengthLevel0(cur_obj, query_data, 0, ef, top_candidates);

        while (top_candidates.size() > k)
        {
          top_candidates.pop();
        }

        comparison_.fetch_add(comparison);

        length.insert(length.end(), length0.begin(), length0.end());

        return length;
      }
    };

    /// @brief Template Class

    template class HNSW<float>;
    template class HNSW<uint8_t>;

  }; // namespace graph

}; // namespace index
