#include <vector_ops.hpp>
#include <graph/nsw.hpp>
#include <utils/binary_io.hpp>
#include <utils/stimer.hpp>

#include <algorithm>
// #include <stdexcept>

#include <omp.h>

namespace anns
{

  namespace graph
  {

    template <typename vdim_t>
    NSW<vdim_t>::NSW(
        size_t D,
        size_t max_elements,
        size_t M,
        size_t ef_construction,
        size_t random_seed) : D_(D), max_elements_(max_elements), M_(M), Mmax_(2 * M),
                              ef_construction_(std::max(ef_construction, M)), random_seed_(random_seed)
    {
      data_size_ = D * sizeof(vdim_t);
      size_links_per_element_ = Mmax_ * sizeof(id_t) + sizeof(size_t);
      size_data_per_element_ = size_links_per_element_ + sizeof(vdim_t *);
      offset_data_ = size_links_per_element_;

      // std::cout << "Size GB: " << (1.0 * max_elements_ * size_data_per_element_ / (1024 * 1024 * 1024)) << std::endl;
      data_level0_memory_.resize(max_elements_ * size_data_per_element_, 0x00);

      cur_element_count_ = 0;

      // visited_list_pool_ = std::make_unique<VisitedListPool> (1, max_elements_);
      link_list_locks_ = std::make_unique<std::vector<std::mutex>>(max_elements_);

      // initializations for special treatment of the first node
      enterpoint_node_ = -1;
    }

    template <typename vdim_t>
    inline const vdim_t *
    NSW<vdim_t>::GetDataByInternalID(id_t id) const
    {
      return *(const vdim_t **)(data_level0_memory_.data() + id * size_data_per_element_ + offset_data_);
    }

    template <typename vdim_t>
    inline void
    NSW<vdim_t>::WriteDataByInternalID(id_t id, const vdim_t *data_point)
    {
      *(const vdim_t **)(data_level0_memory_.data() + id * size_data_per_element_ + offset_data_) = data_point;
    }

    template <typename vdim_t>
    inline char *
    NSW<vdim_t>::GetLinkByInternalID(id_t id) const
    {
      return (char *)(data_level0_memory_.data() + id * size_data_per_element_);
    }

    /// @brief Search knn in with early stop
    template <typename vdim_t>
    std::priority_queue<std::pair<float, id_t>>
    NSW<vdim_t>::SearchBaseLayer(id_t ep_id, const vdim_t *data_point, size_t ef)
    {
      auto mass_visited = std::make_unique<std::vector<bool>>(max_elements_, false);

      std::priority_queue<std::pair<float, id_t>> top_candidates;
      std::priority_queue<std::pair<float, id_t>> candidate_set;

      size_t comparison = 0;

      float dist = vec_L2sqr(data_point, GetDataByInternalID(ep_id), D_);
      comparison++;

      top_candidates.emplace(dist, ep_id); // max heap
      candidate_set.emplace(-dist, ep_id); // min heap
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

        std::unique_lock<std::mutex> lock(link_list_locks_->at(curr_node_id));

        size_t *ll_cur = (size_t *)GetLinkByInternalID(curr_node_id);
        size_t num_neighbors = *ll_cur;
        id_t *neighbors = (id_t *)(ll_cur + 1);

        // #if defined(__SSE__)
        //         /// @brief Prefetch cache lines to speed up cpu caculation.
        //         _mm_prefetch((char *) (mass_visited + *neighbors), _MM_HINT_T0);
        //         _mm_prefetch((char *) (mass_visited + *neighbors + 64), _MM_HINT_T0);
        //         _mm_prefetch((char *) (GetDataByInternalID(*neighbors)), _MM_HINT_T0);
        // #endif

        for (size_t j = 0; j < num_neighbors; j++)
        {
          id_t neighbor_id = neighbors[j];

          // #if defined(__SSE__)
          //             _mm_prefetch((char *) (mass_visited + *(neighbors + j + 1)), _MM_HINT_T0);
          //             _mm_prefetch((char *) (GetDataByInternalID(*(neighbors + j + 1))), _MM_HINT_T0);
          // #endif

          if (!mass_visited->at(neighbor_id))
          {
            mass_visited->at(neighbor_id) = true;

            float dd = vec_L2sqr(data_point, GetDataByInternalID(neighbor_id), D_);
            comparison++;

            /// @brief If neighbor is closer than farest vector in top result, and result.size still less than ef
            if (top_candidates.top().first > dd || top_candidates.size() < ef)
            {
              candidate_set.emplace(-dd, neighbor_id);
              top_candidates.emplace(dd, neighbor_id);

              // #if defined(__SSE__)
              //                     _mm_prefetch((char *) (GetLinkByInternalID(candidate_set.top().second)), _MM_HINT_T0);
              // #endif

              if (top_candidates.size() > ef) // give up farest result so far
                top_candidates.pop();

              if (top_candidates.size())
                low_bound = top_candidates.top().first;
            }
          }
        }
      }

      comparison_.fetch_add(comparison);

      // visited_list_pool_->ReleaseVisitedList(vl);
      return top_candidates;
    }

    /// @brief Extend NN nearest neighbors (within cluster and between clusters) from given top_candidates (By current distance and neighbors).
    template <typename vdim_t>
    void
    NSW<vdim_t>::GetNeighborsByHeuristic(std::priority_queue<std::pair<float, id_t>> &top_candidates, size_t NN)
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

    /// @brief connect edges and return closest node id
    template <typename vdim_t>
    void NSW<vdim_t>::MutuallyConnectNewElement(
        const vdim_t *data_point,
        id_t id,
        std::priority_queue<std::pair<float, id_t>> &top_candidates)
    {
      GetNeighborsByHeuristic(top_candidates, M_);

      std::vector<id_t> selected_neighbors;
      selected_neighbors.reserve(M_);
      while (top_candidates.size())
      {
        selected_neighbors.emplace_back(top_candidates.top().second);
        top_candidates.pop();
      }

      /// Edge-slots check and Add neighbors for current vector
      {
        // lock only during the update
        // because during the addition the lock for cur_c is already acquired
        std::unique_lock<std::mutex> lock((*link_list_locks_)[id], std::defer_lock);
        size_t *ll_cur = (size_t *)GetLinkByInternalID(id);
        size_t num_neighbors = *ll_cur;

        if (num_neighbors)
        {
          std::cout << "The newly inserted element should have blank link list" << std::endl;
          exit(1);
        }

        *ll_cur = selected_neighbors.size();

        id_t *neighbors = (id_t *)(ll_cur + 1);
        for (size_t i = 0; i < selected_neighbors.size(); i++)
        {
          if (neighbors[i])
          {
            std::cout << "Possible memory corruption" << std::endl;
            exit(1);
          }
          neighbors[i] = selected_neighbors[i];
        }
      }

      for (size_t i = 0; i < selected_neighbors.size(); i++)
      {
        std::unique_lock<std::mutex> lock((*link_list_locks_)[selected_neighbors[i]]);

        size_t *ll_other = (size_t *)GetLinkByInternalID(selected_neighbors[i]);
        size_t sz_link_list_other = *ll_other;

        if (sz_link_list_other > Mmax_ || sz_link_list_other < 0)
        {
          std::cout << "Bad value of sz_link_list_other" << std::endl;
          exit(1);
        }
        if (selected_neighbors[i] == id)
        {
          std::cout << "Trying to connect an element to itself" << std::endl;
          exit(1);
        }

        id_t *neighbors = (id_t *)(ll_other + 1);

        if (sz_link_list_other < Mmax_)
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

          GetNeighborsByHeuristic(candidates, Mmax_);

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
    }

    template <typename vdim_t>
    void NSW<vdim_t>::AddPoint(const vdim_t *data_point)
    {
      if (cur_element_count_ >= max_elements_)
      {
        std::cout << "The number of elements exceeds the specified limit" << std::endl;
        exit(1);
      }

      id_t cur_id, cur_obj, enterpoint_node_copy;
      {
        std::unique_lock<std::mutex> temp_lock(global_);
        cur_id = cur_element_count_++;
        cur_obj = enterpoint_node_;
        enterpoint_node_copy = enterpoint_node_;
      }

      WriteDataByInternalID(cur_id, data_point);

      // link cur_id node with other nodes
      if (enterpoint_node_copy != -1)
      { // not first element
        // find the closet node in upper layers
        float cur_dist = vec_L2sqr(data_point, GetDataByInternalID(cur_obj), D_);
        bool changed = true;
        while (changed)
        {
          changed = false;
          std::unique_lock<std::mutex> wlock(link_list_locks_->at(cur_obj));
          size_t *ll_cur = (size_t *)GetLinkByInternalID(cur_obj);
          size_t num_neighbors = *ll_cur;
          id_t *neighbors = (id_t *)(ll_cur + 1);

          for (size_t i = 0; i < num_neighbors; i++)
          {
            id_t cand = neighbors[i];
            if (cand < 0 || cand > max_elements_)
            {
              std::cout << "cand error" << std::endl;
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
        /// add edges to lower layers from the closest node
        auto top_candidates = SearchBaseLayer(cur_obj, data_point, ef_construction_);
        MutuallyConnectNewElement(data_point, cur_id, top_candidates);
      }
      else
      {
        // Do nothing for the first element
        enterpoint_node_ = 0;
      }
    }

    template <typename vdim_t>
    std::priority_queue<std::pair<float, id_t>>
    NSW<vdim_t>::Search(const vdim_t *query_data, size_t k, size_t ef)
    {
      assert(ready_ && "Index uninitialized!");
      assert(ef >= k && "ef >= k!");
      if (cur_element_count_ == 0)
        return std::priority_queue<std::pair<float, id_t>>();
      /// find the closet node in upper layers
      id_t cur_obj = enterpoint_node_; // random?
      size_t comparison = 0;

      float cur_dist = vec_L2sqr(query_data, GetDataByInternalID(cur_obj), D_);
      comparison++;

      bool changed = true;
      while (changed)
      {
        changed = false;
        std::unique_lock<std::mutex> wlock((*link_list_locks_)[cur_obj]);
        size_t *ll_cur = (size_t *)GetLinkByInternalID(cur_obj);
        size_t num_neighbors = *ll_cur;
        id_t *neighbors = (id_t *)(ll_cur + 1);

        for (size_t i = 0; i < num_neighbors; i++)
        {
          id_t cand = neighbors[i];
          if (cand < 0 || cand > max_elements_)
          {
            std::cout << "cand error" << std::endl;
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
      // std::cout << "search base layer" << std::endl;
      // auto top_candidates = SearchBaseLayer(enterpoint_node_, query_data, ef);
      auto top_candidates = SearchBaseLayer(cur_obj, query_data, ef);
      while (top_candidates.size() > k)
      {
        top_candidates.pop();
      }

      comparison_.fetch_add(comparison);
      return top_candidates;
    }

    template <typename vdim_t>
    std::priority_queue<std::pair<float, id_t>>
    NSW<vdim_t>::Search(const vdim_t *query_data, size_t k, size_t ef, id_t ep)
    {
      assert(ready_ && "Index uninitialized!");

      assert(ef >= k && "ef > k!");

      if (cur_element_count_ == 0)
        return std::priority_queue<std::pair<float, id_t>>();

      id_t cur_obj = ep;
      size_t comparison = 0;

      float cur_dist = vec_L2sqr(query_data, GetDataByInternalID(cur_obj), D_);
      comparison++;

      bool changed = true;
      while (changed)
      {
        changed = false;
        std::unique_lock<std::mutex> wlock((*link_list_locks_)[cur_obj]);
        size_t *ll_cur = (size_t *)GetLinkByInternalID(cur_obj);
        size_t num_neighbors = *ll_cur;
        id_t *neighbors = (id_t *)(ll_cur + 1);

        for (size_t i = 0; i < num_neighbors; i++)
        {
          id_t cand = neighbors[i];
          if (cand < 0 || cand > max_elements_)
          {
            std::cout << "cand error" << std::endl;
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
      // auto top_candidates = SearchBaseLayer(ep, query_data, ef);
      auto top_candidates = SearchBaseLayer(cur_obj, query_data, ef);
      while (top_candidates.size() > k)
      {
        top_candidates.pop();
      }

      comparison_.fetch_add(comparison);
      return top_candidates;
    }

    template <typename vdim_t>
    void NSW<vdim_t>::Search(
        const std::vector<std::vector<vdim_t>> &queries,
        size_t k,
        size_t ef,
        std::vector<std::vector<id_t>> &vids,
        std::vector<std::vector<float>> &dists)
    {

      size_t nq = queries.size();
      vids.clear();
      dists.clear();
      vids.resize(nq);
      dists.resize(nq);
      // omp_set_num_threads(num_threads_);
#pragma omp parallel for schedule(dynamic, 64) num_threads(num_threads_)
      for (size_t i = 0; i < nq; i++)
      {
        const auto &query = queries[i];
        auto &vid = vids[i];
        auto &dist = dists[i];
        // std::cout << 1212 << std::endl;
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
      // std::cout << 3134 << std::endl;
    }

    template <typename vdim_t>
    void
    NSW<vdim_t>::Populate(const std::vector<vdim_t> &raw_data)
    {
      size_t N = raw_data.size() / D_;
      assert(N <= max_elements_ && "data size too large!");
      // omp_set_num_threads(num_threads_);
#pragma omp parallel for schedule(dynamic, 1) num_threads(num_threads_)
      for (size_t i = 0; i < N; i++)
        AddPoint(raw_data.data() + i * D_);

      ready_ = true;
    }

    template <typename vdim_t>
    void
    NSW<vdim_t>::Populate(const std::vector<const vdim_t *> &raw_data)
    {
      size_t N = raw_data.size();
      assert(N <= max_elements_ && "data size too large!");
#pragma omp parallel for schedule(dynamic, 1) num_threads(num_threads_)
      for (size_t i = 0; i < N; i++) 
        AddPoint(raw_data[i]);

      ready_ = true;
    }

    template <typename vdim_t>
    bool
    NSW<vdim_t>::Ready() { return ready_; }

    template <typename vdim_t>
    size_t NSW<vdim_t>::GetNumThreads()
    {
      return num_threads_;
    }

    template <typename vdim_t>
    void
    NSW<vdim_t>::SetNumThreads(size_t num_threads) { num_threads_ = num_threads; }

    // template <typename vdim_t> void NSW<vdim_t>::SaveEdgesTXT(const std::string & path) {
    //     std::ofstream output(path);
    //     for (id_t u0 = 0; u0 < cur_element_count_; u0 ++) {
    //         size_t * h = (size_t *) GetLinkByInternalID(u0);
    //         size_t n = *h;
    //         id_t * ll = (id_t *) (h + 1);
    //         for (size_t j = 0; j < n; j++) {
    //             output << u0 << " " << ll[j] << std::endl;
    //         }
    //     }
    // }

    template <typename vdim_t>
    size_t NSW<vdim_t>::IndexSize() const
    {
      return data_level0_memory_.size() * sizeof(char);
    }

    template <typename vdim_t>
    size_t NSW<vdim_t>::GetComparisonAndClear()
    {
      return comparison_.exchange(0);
    }

    template <typename vdim_t>
    id_t NSW<vdim_t>::GetClosestPoint(const vdim_t *data_point)
    {
      if (cur_element_count_ == 0)
        throw std::runtime_error("empty graph");
      id_t wander = enterpoint_node_;
      size_t comparison = 0;

      float dist = vec_L2sqr(data_point, GetDataByInternalID(wander), D_);
      comparison++;

      bool moving = true;
      while (moving)
      {
        moving = false;
        size_t *ll = (size_t *)GetLinkByInternalID(wander);
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

      comparison_.fetch_add(comparison);

      return wander;
    }

  }; // namespace graph

}; // namespace index
