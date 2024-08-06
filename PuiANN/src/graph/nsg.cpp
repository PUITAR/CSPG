#include <graph/nsg.hpp>
#include <utils/binary_io.hpp>
#include <utils/stimer.hpp>

#include <algorithm>
#include <bitset>
// #include <stdexcept>

#include <omp.h>

// #include <sys/mman.h>

namespace puiann
{

  namespace graph
  {

    template <typename vdim_t>
    NSG<vdim_t>::NSG(
        size_t D,
        size_t max_elements,
        size_t R,
        int random_seed) : D_(D), max_elements_(max_elements), R_(R), random_seed_(random_seed)
    {
      data_size_ = D_ * sizeof(vdim_t);
      size_links_per_element_ = R_ * sizeof(id_t) + sizeof(size_t);
      size_data_per_element_ = size_links_per_element_ + sizeof(vdim_t *);
      offset_data_ = size_links_per_element_;

      data_memory_.resize(max_elements_ * size_data_per_element_, 0x00);
      // mlock for data_memory
      // assert(mlock(data_memory_.data(), data_memory_.size()) == 0);

      cur_element_count_ = 0;

      // visited_list_pool_ = std::make_unique<VisitedListPool>(1, max_elements_);
      link_list_locks_ = std::make_unique<std::vector<std::mutex>>(max_elements_);

      enterpoint_node_ = -1;
    }

    template <typename vdim_t>
    inline const vdim_t *
    NSG<vdim_t>::GetDataByInternalID(id_t id) const
    {
      return *((const vdim_t **)(data_memory_.data() + id * size_data_per_element_ + offset_data_));
    }

    template <typename vdim_t>
    inline void
    NSG<vdim_t>::WriteDataByInternalID(id_t id, const vdim_t *data_point)
    {
      *((const vdim_t **)(data_memory_.data() + id * size_data_per_element_ + offset_data_)) = data_point;
    }

    template <typename vdim_t>
    inline char *
    NSG<vdim_t>::GetLinkByInternalID(id_t id) const
    {
      return (char *)(data_memory_.data() + id * size_data_per_element_);
    }

    template <typename vdim_t>
    size_t NSG<vdim_t>::GetNumThreads()
    {
      return num_threads_;
    }

    template <typename vdim_t>
    void NSG<vdim_t>::SetNumThreads(size_t num_threads)
    {
      num_threads_ = num_threads;
    }

    template <typename vdim_t>
    bool NSG<vdim_t>::Ready() { return ready_; }

    /// @brief the single thread search for a query
    /// @tparam vdim_t
    /// @param query_data
    /// @param k
    /// @param L
    /// @param visited return the tuple (-distance2query, vector id)
    template <typename vdim_t>
    void NSG<vdim_t>::Search(const vdim_t *query_data, size_t k, size_t L, std::vector<std::pair<float, id_t>> &visited)
    {
      static const size_t rb = 2;
      assert(L >= k);

      size_t comparison = 0;

      /// @brief Search top-K NNs in a gready way
      // visited.clear();
      // auto vl = visited_list_pool_->GetFreeVisitedList();
      // auto visited_set = vl->mass_.data();
      // auto curr_visited = vl->curr_visited_;
      auto visited_set = std::make_unique<std::vector<bool>>(max_elements_, false);
      std::priority_queue<std::pair<float, id_t>> top_candidates; /* min-heap to remain the top-L NNs */
      id_t ep = rand() % rb;
      top_candidates.emplace(
          -vec_L2sqr(GetDataByInternalID(ep), query_data, D_),
          ep);
      comparison++;

      while (top_candidates.size())
      {
        auto [pstar_dist, pstar] = top_candidates.top();
        // std::cout << "pstar_dist: " << pstar_dist << ", pstar: " << pstar << std::endl;
        visited_set->at(pstar) = true;              // update visited_set
        visited.emplace_back(top_candidates.top()); // update visited
        top_candidates.pop();

        {
          std::unique_lock<std::mutex> lock((*link_list_locks_)[pstar]);
          size_t *llc = (size_t *)GetLinkByInternalID(pstar);
          size_t num_neighbors = *llc;
          id_t *neighbors = (id_t *)(llc + 1);
          for (size_t i = 0; i < num_neighbors; i++)
          {
            id_t neighbor_id = neighbors[i];
            // std::cout << "neighbor_id: " << neighbor_id << std::endl;
            if (visited_set->at(neighbor_id) == false)
            {
              top_candidates.emplace(
                  -vec_L2sqr(GetDataByInternalID(neighbor_id), query_data, D_),
                  neighbor_id);
              comparison++;
            }
          }
        }

        size_t candL = visited.size() > L ? 0 : L - visited.size();
        // std::cout << candL << std::endl;
        std::priority_queue<std::pair<float, id_t>> temp_candidates;
        while (candL-- && top_candidates.size())
        {
          temp_candidates.emplace(top_candidates.top());
          top_candidates.pop();
        }
        top_candidates.swap(temp_candidates);
      }

      // sort visited array to get results
      std::partial_sort(
          visited.begin(), visited.begin() + k, visited.end(),
          [](const std::pair<float, id_t> &a, const std::pair<float, id_t> &b)
          {
            return a.first > b.first;
          });

      // visited_list_pool_->ReleaseVisitedList(vl);
      comparison_.fetch_add(comparison);
    }

    template <typename vdim_t>
    void NSG<vdim_t>::Search(const vdim_t *query_data, size_t k, size_t L, id_t ep, std::vector<std::pair<float, id_t>> &visited)
    {
      assert(L >= k);
      size_t comparison = 0;

      /// @brief Search top-K NNs in a gready way
      // visited.clear();
      // auto vl = visited_list_pool_->GetFreeVisitedList();
      // auto visited_set = vl->mass_.data();
      // auto curr_visited = vl->curr_visited_;
      auto visited_set = std::make_unique<std::vector<bool>>(max_elements_, false);
      std::priority_queue<std::pair<float, id_t>> top_candidates; /* min-heap to remain the top-L NNs */
      top_candidates.emplace(
          -vec_L2sqr(GetDataByInternalID(ep), query_data, D_),
          ep);
      comparison++;

      while (top_candidates.size())
      {
        auto [pstar_dist, pstar] = top_candidates.top();
        // std::cout << "pstar_dist: " << pstar_dist << ", pstar: " << pstar << std::endl;
        // visited_set[pstar] = curr_visited;          // update visited_set
        visited_set->at(pstar) = true;
        visited.emplace_back(top_candidates.top()); // update visited
        top_candidates.pop();

        /* update L with Nout(p*) */
        {
          std::unique_lock<std::mutex> lock(link_list_locks_->at(pstar));
          size_t *llc = (size_t *)GetLinkByInternalID(pstar);
          size_t num_neighbors = *llc;
          id_t *neighbors = (id_t *)(llc + 1);
          for (size_t i = 0; i < num_neighbors; i++)
          {
            id_t neighbor_id = neighbors[i];
            // std::cout << "neighbor_id: " << neighbor_id << std::endl;
            if (visited_set->at(neighbor_id) == false)
            {
              top_candidates.emplace(
                  -vec_L2sqr(GetDataByInternalID(neighbor_id), query_data, D_),
                  neighbor_id);
              comparison++;
            }
          }
        }

        size_t candL = visited.size() > L ? 0 : L - visited.size();
        // std::cout << candL << std::endl;
        std::priority_queue<std::pair<float, id_t>> temp_candidates;
        while (candL--)
        {
          temp_candidates.emplace(top_candidates.top());
          top_candidates.pop();
        }
        top_candidates.swap(temp_candidates);

        comparison_.fetch_add(comparison);
      }

      // sort visited array to get results
      std::partial_sort(
          visited.begin(), visited.begin() + k, visited.end(),
          [](const std::pair<float, id_t> &a, const std::pair<float, id_t> &b)
          {
            return a.first > b.first;
          });

      // visited_list_pool_->ReleaseVisitedList(vl);
    }

    /// @brief Prune function
    /// @tparam vdim_t
    /// @param node_id
    /// @param alpha
    /// @param candidates
    template <typename vdim_t>
    void NSG<vdim_t>::RobustPrune(
        id_t node_id,
        float alpha,
        std::vector<std::pair<float, id_t>> &candidates)
    {
      assert(alpha >= 1);
      const vdim_t *data_node = GetDataByInternalID(node_id);

      // Ps: It will make a dead-lock if locked here, so make sure the code have locked the link-list of
      // the pruning node outside of the function `RobustPrune` in caller
      // std::unique_lock < std::mutex > lock( (* link_list_locks_) [node_id] );
      size_t *llc = (size_t *)GetLinkByInternalID(node_id);
      size_t num_neighbors = *llc;
      id_t *neighbors = (id_t *)(llc + 1);

      assert(num_neighbors <= R_);

      for (size_t i = 0; i < num_neighbors; i++)
      {
        candidates.emplace_back(
            -vec_L2sqr(GetDataByInternalID(neighbors[i]), data_node, D_),
            neighbors[i]);
      }
      *llc = 0; // clear link list

      {
        std::unordered_set<std::pair<float, id_t>, PHash> cand_set;
        for (const auto &item : candidates)
        {
          if (item.second != node_id)
          {
            cand_set.insert(item);
          }
        }
        candidates.assign(cand_set.begin(), cand_set.end());
      }

      std::make_heap(candidates.begin(), candidates.end());
      while (candidates.size())
      {
        if (*llc >= R_)
          break;
        auto [pstar_dist, pstar] = candidates.front();
        // std::cout << "pstar_dist: " << pstar_dist << " pstar: " << pstar << std::endl;
        // insert p* into Nout(p)
        neighbors[(*llc)++] = pstar;

        const vdim_t *data_pstar = GetDataByInternalID(pstar);
        for (size_t i = 0; i < candidates.size();)
        {
          auto [d, id] = candidates[i];
          if (alpha * vec_L2sqr(data_pstar, GetDataByInternalID(id), D_) <= -d)
          {
            candidates[i] = candidates.back();
            candidates.pop_back();
          }
          else
          {
            i++;
          }
        }

        std::make_heap(candidates.begin(), candidates.end());
      }

      // std::cout << "prune finished" << std::endl;
    }

    template <typename vdim_t>
    void NSG<vdim_t>::BuildIndex(
        const std::vector<vdim_t> &raw_data,
        size_t L)
    {
      const size_t num_points = raw_data.size() / D_;
      // std::cout << num_points << " " << max_elements_ << " " << num_points << std::endl;
      assert(num_points <= max_elements_ && num_points > 0);
      cur_element_count_ = num_points;

      // Initialize graph index to a random R-regular directed graph
#pragma omp parallel for schedule(dynamic, 1) num_threads(num_threads_)
      for (id_t id = 0; id < num_points; id++)
      {
        WriteDataByInternalID(id, raw_data.data() + id * D_);
        size_t *ll_cur = (size_t *)GetLinkByInternalID(id);
        *ll_cur = R_;
        id_t *neighbors = (id_t *)(ll_cur + 1);
        for (size_t i = 0; i < R_; i++)
        {
          id_t rid = id;
          while (rid == id)
          {
            rid = (id_t)(rand() % num_points);
          }
          neighbors[i] = rid;
        }
      }
      // std::cout << "Initialized graph to a random R-regular directed graph" << std::endl;

      // Compute medoid of the raw dataset
      std::vector<long double> dim_sum(D_, .0);
      std::vector<vdim_t> medoid(D_, 0);
      auto dim_lock_list = std::make_unique<std::vector<std::mutex>>(D_);

#pragma omp parallel for schedule(dynamic, 512) num_threads(num_threads_)
      for (id_t id = 0; id < num_points; id++)
      {
        const vdim_t *vec = raw_data.data() + id * D_;
        for (size_t i = 0; i < D_; i++)
        {
          std::unique_lock<std::mutex> lock((*dim_lock_list)[i]);
          dim_sum[i] += vec[i];
        }
      } //

#pragma omp parallel for schedule(dynamic, 1) num_threads(num_threads_)
      for (size_t i = 0; i < D_; i++)
      {
        medoid[i] = static_cast<vdim_t>(dim_sum[i] / num_points);
      }
      float nearest_dist = std::numeric_limits<float>::max();
      id_t nearest_node = -1;

#pragma omp parallel for schedule(dynamic, 512) num_threads(num_threads_)
      for (id_t id = 0; id < num_points; id++)
      {
        float dist = vec_L2sqr(medoid.data(), raw_data.data() + id * D_, D_);
        std::unique_lock<std::mutex> lock(global_);
        if (dist < nearest_dist)
        {
          nearest_dist = dist;
          nearest_node = id;
        }
      }
      enterpoint_node_ = nearest_node;
      // std::cout << "Computed enterpoint node: " << enterpoint_node_ << std::endl;

      // Generate a random permutation sigma
      std::vector<id_t> sigma(num_points);

#pragma omp parallel for schedule(dynamic, 512) num_threads(num_threads_)
      for (id_t id = 0; id < num_points; id++)
      {
        sigma[id] = id;
      }
      std::random_shuffle(sigma.begin(), sigma.end());
      // std::cout << "Generated random permutation sigma" << std::endl;

      // Building pass begin
      auto pass = [&](float beta)
      {

#pragma omp parallel for schedule(dynamic, 512) num_threads(num_threads_)
        for (size_t i = 0; i < num_points; i++)
        {
          id_t cur_id = sigma[i];
          std::vector<std::pair<float, id_t>> top_candidates;

          Search(GetDataByInternalID(cur_id), 1, L, top_candidates);

          std::unique_lock<std::mutex> lock(link_list_locks_->at(cur_id));
          RobustPrune(cur_id, beta, top_candidates);
          size_t *llc = (size_t *)GetLinkByInternalID(cur_id);
          size_t num_neighbors = *llc;
          id_t *neighbors = (id_t *)(llc + 1);
          std::vector<id_t> neighbors_copy(neighbors, neighbors + num_neighbors);
          lock.unlock();
          if (num_neighbors > R_)
          {
            std::cerr << num_neighbors << " " << R_ << std::endl;
          }

          for (size_t j = 0; j < num_neighbors; j++)
          {
            id_t neij = neighbors_copy[j];
            std::unique_lock<std::mutex> lock_neij(link_list_locks_->at(neij));
            size_t *llc_other = (size_t *)GetLinkByInternalID(neij);
            size_t num_neighbors_other = *llc_other;
            id_t *neighbors_other = (id_t *)(llc_other + 1);

            bool find_cur_id = false;
            for (size_t k = 0; k < num_neighbors_other; k++)
            {
              if (cur_id == neighbors_other[k])
              {
                find_cur_id = true;
                break;
              }
            }

            if (!find_cur_id)
            {
              if (num_neighbors_other == R_)
              {
                std::vector<std::pair<float, id_t>> temp_cand_set = {{-vec_L2sqr(GetDataByInternalID(neij), GetDataByInternalID(cur_id), D_), cur_id}};
                RobustPrune(neij, beta, temp_cand_set);
              }
              else if (num_neighbors_other < R_)
              {
                neighbors_other[num_neighbors_other] = cur_id;
                (*llc_other)++;
              }
              else
              {
                throw std::runtime_error("adjency overflow");
              }
            }
          }
        }
      };

      /// First pass with alpha = 1.0
      pass(1.0);
      // std::cout << "First pass done!" << std::endl;
      /// Second pass with user-defined alpha
      // pass(1.2);
      // std::cout << "Second pass done!" << std::endl;

      ready_ = true;
    }

    template <typename vdim_t>
    void NSG<vdim_t>::BuildIndex(
        const std::vector<const vdim_t *> &raw_data,
        size_t L)
    {
      const size_t num_points = raw_data.size();
      // std::cout << num_points << " " << max_elements_ << " " << num_points << std::endl;
      assert(num_points <= max_elements_ && num_points > 0);
      cur_element_count_ = num_points;

      // Initialize graph index to a random R-regular directed graph
#pragma omp parallel for schedule(dynamic, 1) num_threads(num_threads_)
      for (id_t id = 0; id < num_points; id++)
      {
        WriteDataByInternalID(id, raw_data[id]);
        size_t *ll_cur = (size_t *)GetLinkByInternalID(id);
        *ll_cur = R_;
        id_t *neighbors = (id_t *)(ll_cur + 1);
        for (size_t i = 0; i < R_; i++)
        {
          id_t rid = id;
          while (rid == id)
          {
            rid = (id_t)(rand() % num_points);
          }
          neighbors[i] = rid;
        }
      }
      // std::cout << "Initialized graph to a random R-regular directed graph" << std::endl;

      // Compute medoid of the raw dataset
      std::vector<long double> dim_sum(D_, .0);
      std::vector<vdim_t> medoid(D_, 0);
      auto dim_lock_list = std::make_unique<std::vector<std::mutex>>(D_);

#pragma omp parallel for schedule(dynamic, 512) num_threads(num_threads_)
      for (id_t id = 0; id < num_points; id++)
      {
        const vdim_t *vec = raw_data[id];
        for (size_t i = 0; i < D_; i++)
        {
          std::unique_lock<std::mutex> lock((*dim_lock_list)[i]);
          dim_sum[i] += vec[i];
        }
      } //

#pragma omp parallel for schedule(dynamic, 1) num_threads(num_threads_)
      for (size_t i = 0; i < D_; i++)
      {
        medoid[i] = static_cast<vdim_t>(dim_sum[i] / num_points);
      }
      float nearest_dist = std::numeric_limits<float>::max();
      id_t nearest_node = -1;

#pragma omp parallel for schedule(dynamic, 512) num_threads(num_threads_)
      for (id_t id = 0; id < num_points; id++)
      {
        float dist = vec_L2sqr(medoid.data(), raw_data[id], D_);
        std::unique_lock<std::mutex> lock(global_);
        if (dist < nearest_dist)
        {
          nearest_dist = dist;
          nearest_node = id;
        }
      }
      enterpoint_node_ = nearest_node;
      // std::cout << "Computed enterpoint node: " << enterpoint_node_ << std::endl;

      // Generate a random permutation sigma
      std::vector<id_t> sigma(num_points);

#pragma omp parallel for schedule(dynamic, 512) num_threads(num_threads_)
      for (id_t id = 0; id < num_points; id++)
      {
        sigma[id] = id;
      }
      std::random_shuffle(sigma.begin(), sigma.end());
      // std::cout << "Generated random permutation sigma" << std::endl;

      // Building pass begin
      auto pass = [&](float beta)
      {
#pragma omp parallel for schedule(dynamic, 512) num_threads(num_threads_)
        for (size_t i = 0; i < num_points; i++)
        {
          id_t cur_id = sigma[i];
          std::vector<std::pair<float, id_t>> top_candidates;

          Search(GetDataByInternalID(cur_id), 1, L, top_candidates);

          std::unique_lock<std::mutex> lock(link_list_locks_->at(cur_id));
          RobustPrune(cur_id, beta, top_candidates);
          size_t *llc = (size_t *)GetLinkByInternalID(cur_id);
          size_t num_neighbors = *llc;
          id_t *neighbors = (id_t *)(llc + 1);
          std::vector<id_t> neighbors_copy(neighbors, neighbors + num_neighbors);
          lock.unlock();
          if (num_neighbors > R_)
          {
            std::cerr << num_neighbors << " " << R_ << std::endl;
          }

          for (size_t j = 0; j < num_neighbors; j++)
          {
            id_t neij = neighbors_copy[j];
            std::unique_lock<std::mutex> lock_neij(link_list_locks_->at(neij));
            size_t *llc_other = (size_t *)GetLinkByInternalID(neij);
            size_t num_neighbors_other = *llc_other;
            id_t *neighbors_other = (id_t *)(llc_other + 1);

            bool find_cur_id = false;
            for (size_t k = 0; k < num_neighbors_other; k++)
            {
              if (cur_id == neighbors_other[k])
              {
                find_cur_id = true;
                break;
              }
            }

            if (!find_cur_id)
            {
              if (num_neighbors_other == R_)
              {
                std::vector<std::pair<float, id_t>> temp_cand_set = {{-vec_L2sqr(GetDataByInternalID(neij), GetDataByInternalID(cur_id), D_), cur_id}};
                RobustPrune(neij, beta, temp_cand_set);
              }
              else if (num_neighbors_other < R_)
              {
                neighbors_other[num_neighbors_other] = cur_id;
                (*llc_other)++;
              }
              else
              {
                throw std::runtime_error("adjency overflow");
              }
            }
          }
        }
      };

      /// First pass with alpha = 1.0
      pass(1.0);
      // std::cout << "First pass done!" << std::endl;
      /// Second pass with user-defined alpha
      // pass(1.2);
      // std::cout << "Second pass done!" << std::endl;

      ready_ = true;
    }

    template <typename vdim_t>
    void NSG<vdim_t>::Search(
        const std::vector<std::vector<vdim_t>> &queries, size_t k, size_t L,
        std::vector<std::vector<id_t>> &vids, std::vector<std::vector<float>> &dists)
    {
      const size_t nq = queries.size();
      vids.clear();
      dists.clear();
      vids.resize(nq);
      dists.resize(nq);

#pragma omp parallel for schedule(dynamic, 64) num_threads(num_threads_)
      for (size_t i = 0; i < nq; i++)
      {
        // std::cerr << "query " << i << " finished" << std::endl;

        const auto &query = queries[i];
        auto &vid = vids[i];
        auto &dist = dists[i];

        std::vector<std::pair<float, id_t>> results;
        Search(query.data(), k, L, results);

        // std::cerr << results.size() << std::endl;
        size_t actual_k = std::min(k, results.size());
        results.resize(actual_k);
        results.shrink_to_fit();
        vid.reserve(actual_k);
        dist.reserve(actual_k);

        for (const auto &[d, id] : results)
        {
          vid.emplace_back(id);
          dist.emplace_back(-d);
          // std::cout << id << " -> " << -d << std::endl;
        }
      }
    }

    // template <typename vdim_t>
    // void NSG<vdim_t>::SaveNSG(const std::string &info_path, const std::string &edge_path, const std::string &data_path)
    // {
    //   SaveInfo(info_path);
    //   SaveData(data_path);
    //   SaveEdges(edge_path);
    // }

    template <typename vdim_t>
    size_t NSG<vdim_t>::GetComparisonAndClear()
    {
      return comparison_.exchange(0);
    }

    template <typename vdim_t>
    size_t NSG<vdim_t>::IndexSize() const
    {
      size_t sz = data_memory_.size() * sizeof(char);
      return sz;
    }

    template <typename vdim_t>
    id_t NSG<vdim_t>::GetClosestPoint(const vdim_t *data_point)
    {
      if (cur_element_count_ == 0)
      {
        throw std::runtime_error("empty graph");
      }
      size_t comparison = 0;
      id_t wander = enterpoint_node_;
      float dist = vec_L2sqr(data_point, GetDataByInternalID(wander), D_);
      comparison++;
      bool moving = true;
      while (moving)
      {
        moving = false;
        size_t *ll = (size_t *)GetLinkByInternalID(wander);
        size_t n = *ll;
        id_t *adj = (id_t *)(ll + 1);
        for (size_t i = 0; i < n; i++)
        {
          id_t cand = adj[i];
          float d = vec_L2sqr(data_point, GetDataByInternalID(cand), D_);
          if (d < dist)
          {
            wander = cand;
            dist = d;
            moving = true;
          }
        }
        comparison += n;
      }
      comparison_.fetch_add(comparison);
      return wander;
    }

    template <typename vdim_t>
    std::vector<float> NSG<vdim_t>::GetSearchLength(const vdim_t *query_data, size_t k, size_t L, std::vector<std::pair<float, id_t>> &visited)
    {
      std::vector<float> length;
      // static const size_t rb = 2;
      assert(L >= k);

      size_t comparison = 0;

      /// @brief Search top-K NNs in a gready way
      // visited.clear();
      // auto vl = visited_list_pool_->GetFreeVisitedList();
      // auto visited_set = vl->mass_.data();
      // auto curr_visited = vl->curr_visited_;
      auto visited_set = std::make_unique<std::vector<bool>>(max_elements_, false);
      std::priority_queue<std::pair<float, id_t>> top_candidates; /* min-heap to remain the top-L NNs */
      id_t ep = 0;
      top_candidates.emplace(
          -vec_L2sqr(GetDataByInternalID(ep), query_data, D_),
          ep);
      comparison++;

      while (top_candidates.size())
      {
        auto [pstar_dist, pstar] = top_candidates.top();
        length.emplace_back(-pstar_dist);
        // std::cout << "pstar_dist: " << -pstar_dist << ", pstar: " << pstar << std::endl;
        // visited_set[pstar] = curr_visited;          // update visited_set
        visited_set->at(pstar) = true;
        visited.emplace_back(top_candidates.top()); // update visited
        top_candidates.pop();

        {
          std::unique_lock<std::mutex> lock((*link_list_locks_)[pstar]);
          size_t *llc = (size_t *)GetLinkByInternalID(pstar);
          size_t num_neighbors = *llc;
          id_t *neighbors = (id_t *)(llc + 1);
          for (size_t i = 0; i < num_neighbors; i++)
          {
            id_t neighbor_id = neighbors[i];
            // std::cout << "neighbor_id: " << neighbor_id << std::endl;
            if (visited_set->at(neighbor_id) == false)
            {
              top_candidates.emplace(
                  -vec_L2sqr(GetDataByInternalID(neighbor_id), query_data, D_),
                  neighbor_id);
              comparison++;
            }
          }
        }

        size_t candL = visited.size() > L ? 0 : L - visited.size();
        // std::cout << candL << std::endl;
        std::priority_queue<std::pair<float, id_t>> temp_candidates;
        while (candL--)
        {
          temp_candidates.emplace(top_candidates.top());
          top_candidates.pop();
        }
        top_candidates.swap(temp_candidates);
      }

      // sort visited array to get results
      std::partial_sort(
          visited.begin(), visited.begin() + k, visited.end(),
          [](const std::pair<float, id_t> &a, const std::pair<float, id_t> &b)
          {
            return a.first > b.first;
          });

      // visited_list_pool_->ReleaseVisitedList(vl);
      comparison_.fetch_add(comparison);

      return length;
    }

    template <typename vdim_t>
    std::vector<std::vector<float>> NSG<vdim_t>::GetSearchLength(
        const std::vector<std::vector<vdim_t>> &queries, size_t k, size_t L,
        std::vector<std::vector<id_t>> &vids, std::vector<std::vector<float>> &dists)
    {
      const size_t nq = queries.size();
      vids.clear();
      dists.clear();
      vids.resize(nq);
      dists.resize(nq);
      std::vector<std::vector<float>> lengths(nq);

#pragma omp parallel for schedule(dynamic, 64) num_threads(num_threads_)
      for (size_t i = 0; i < nq; i++)
      {
        // std::cerr << "query " << i << " finished" << std::endl;

        const auto &query = queries[i];
        auto &vid = vids[i];
        auto &dist = dists[i];

        std::vector<std::pair<float, id_t>> results;
        lengths[i] = GetSearchLength(query.data(), k, L, results);

        // std::cerr << results.size() << std::endl;
        size_t actual_k = std::min(k, results.size());
        results.resize(actual_k);
        results.shrink_to_fit();
        vid.reserve(actual_k);
        dist.reserve(actual_k);

        for (const auto &[d, id] : results)
        {
          vid.emplace_back(id);
          dist.emplace_back(-d);
          // std::cout << id << " -> " << -d << std::endl;
        }
      }

      return lengths;
    }

  }

}