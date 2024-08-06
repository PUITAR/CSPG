#include <graph/hcnng.hpp>

#include <utils/binary_io.hpp>
#include <utils/stimer.hpp>

#include <vector_ops.hpp>

#include <algorithm>
#include <memory>

#include <omp.h>
// #include <sys/mman.h>

#include <graph/disjoint_set.hpp>

namespace anns
{

  namespace graph
  {

    template <typename vdim_t>
    HCNNG<vdim_t>::HCNNG(
        size_t D,
        size_t max_elements,
        int random_seed) : D_(D), max_elements_(max_elements), random_seed_(random_seed)
    {
      data_size_ = D_ * sizeof(vdim_t);

      data_memory_.resize(max_elements_);
      // assert(mlock(data_memory_.data(), data_memory_.size()) == 0);
      adj_memory_.resize(max_elements_);

      cur_element_count_ = 0;

      // visited_list_pool_ = std::make_unique<VisitedListPool>(1, max_elements_);

      link_list_locks_ = std::make_unique<std::vector<std::mutex>>(max_elements_);
    }

    // template <typename vdim_t>
    // HCNNG<vdim_t>::~HCNNG()
    // {
    //   assert(munlock(data_memory_.data(), data_memory_.size()) == 0);
    // }

    template <typename vdim_t>
    inline const vdim_t *HCNNG<vdim_t>::GetDataByInternalID(id_t id) const
    {
      return data_memory_[id];
    }

    template <typename vdim_t>
    inline void HCNNG<vdim_t>::WriteDataByInternalID(id_t id, const vdim_t * data_point)
    {
      data_memory_[id] = data_point;
    }

    template <typename vdim_t>
    size_t HCNNG<vdim_t>::GetNumThreads() const { return num_threads_; }

    template <typename vdim_t>
    void HCNNG<vdim_t>::SetNumThreads(size_t num_threads) { num_threads_ = num_threads; }

    template <typename vdim_t>
    bool HCNNG<vdim_t>::Ready() const { return ready_; }

    template <typename vdim_t>
    std::vector<std::vector<Edge>>
    HCNNG<vdim_t>::CreateExactMST(
        const std::vector<id_t> &idx_points,
        size_t left, size_t right, size_t max_mst_degree)
    {
      size_t num_points = right - left + 1;

      // float cost = 0;
      std::vector<Edge> full_graph;
      std::vector<std::vector<Edge>> mst(num_points);
      full_graph.reserve(num_points * (num_points - 1));

      // pick up all edges into full_graph
      for (size_t i = 0; i < num_points; i++)
      {
        for (size_t j = 0; j < num_points; j++)
        {
          if (i != j)
          {
            full_graph.emplace_back(
                Edge{i, j,
                     vec_L2sqr(
                         GetDataByInternalID(idx_points[left + i]),
                         GetDataByInternalID(idx_points[left + j]),
                         D_)});
          }
        }
      }

      // Kruskal algorithm
      std::sort(full_graph.begin(), full_graph.end());
      auto ds = std::make_unique<DisjointSet>(num_points);
      for (const auto &e : full_graph)
      {
        id_t src = e.src;
        id_t dst = e.dst;
        float weight = e.weight;
        if (ds->Find(src) != ds->Find(dst) && mst[src].size() < max_mst_degree && mst[dst].size() < max_mst_degree)
        {
          mst[src].emplace_back(e);
          mst[dst].emplace_back(Edge{dst, src, weight});
          ds->UnionSet(src, dst);
          // cost += weight;
        }
      }

      return mst;
    }

    template <typename vdim_t>
    void HCNNG<vdim_t>::CreateHCNNG(
        const std::vector<vdim_t> &raw_data,
        size_t num_random_clusters,
        size_t min_size_clusters,
        size_t max_mst_degree)
    {
      size_t num_points = raw_data.size() / D_;
      assert(num_points <= max_elements_);
      cur_element_count_ = num_points;

// initialize graph data
#pragma omp parallel for schedule(dynamic, 16) num_threads(num_threads_)
      for (id_t id = 0; id < num_points; id++)
      {
        WriteDataByInternalID(id, raw_data.data()+id*D_);
        adj_memory_[id].reserve(max_mst_degree * num_random_clusters);
      }
#pragma omp parallel for schedule(dynamic, 1) num_threads(num_threads_)
      for (size_t i = 0; i < num_random_clusters; i++)
      {
        // std::cout << "building mst: " << i << "/" << num_random_clusters << std::endl;

        auto idx_points = std::make_unique<std::vector<id_t>>(num_points);

        for (size_t j = 0; j < num_points; j++)
        {
          idx_points->at(j) = j;
        }

        CreateClusters(*idx_points, 0, num_points - 1, min_size_clusters, max_mst_degree);
      }

      ready_ = true;
    }

    template <typename vdim_t>
    void HCNNG<vdim_t>::CreateHCNNG(
        const std::vector<const vdim_t*> &raw_data,
        size_t num_random_clusters,
        size_t min_size_clusters,
        size_t max_mst_degree)
    {
      size_t num_points = raw_data.size();
      assert(num_points <= max_elements_);
      cur_element_count_ = num_points;

// initialize graph data
#pragma omp parallel for schedule(dynamic, 16) num_threads(num_threads_)
      for (id_t id = 0; id < num_points; id++)
      {
        WriteDataByInternalID(id, raw_data[id]);
        adj_memory_[id].reserve(max_mst_degree * num_random_clusters);
      }
#pragma omp parallel for schedule(dynamic, 1) num_threads(num_threads_) 
      for (size_t i = 0; i < num_random_clusters; i++)
      {
        // std::cout << "building mst: " << i << "/" << num_random_clusters << std::endl;

        auto idx_points = std::make_unique<std::vector<id_t>>(num_points);

        for (size_t j = 0; j < num_points; j++)
        {
          idx_points->at(j) = j;
        }

        CreateClusters(*idx_points, 0, num_points - 1, min_size_clusters, max_mst_degree);
      }

      ready_ = true;
    }

    template <typename vdim_t>
    void HCNNG<vdim_t>::CreateClusters(
        std::vector<id_t> &idx_points,
        size_t left, size_t right,
        size_t min_size_clusters,
        size_t max_mst_degree)
    {
      size_t num_points = right - left + 1;
      // std::cout << "hierarchical clustering, N = " << num_points << std::endl;

      if (num_points <= min_size_clusters)
      {
        // std::cout << "come to leaf !" << std::endl;
        auto mst = CreateExactMST(idx_points, left, right, max_mst_degree);

        // Add edges to graph
        for (size_t i = 0; i < num_points; i++)
        {
          for (size_t j = 0; j < mst[i].size(); j++)
          {
            std::unique_lock<std::mutex> lock0((*link_list_locks_)[idx_points[left + i]]);

            bool is_neighbor = false;
            auto &neigh0 = adj_memory_[idx_points[left + i]];

            for (const auto &nid0 : neigh0)
            {
              if (nid0 == idx_points[left + mst[i][j].dst])
              {
                is_neighbor = true;
                break;
              }
            }
            if (!is_neighbor)
            {
              neigh0.emplace_back(idx_points[left + mst[i][j].dst]);
            }
          }
        }
      }
      else
      {
        auto rand_int = [](size_t Min, size_t Max)
        {
          size_t sz = Max - Min + 1;
          return Min + (std::rand() % sz);
        };

        size_t x = rand_int(left, right);
        size_t y = -1;
        do
        {
          y = rand_int(left, right);
        } while (x == y);

        const vdim_t *vec_idx_left_p_x = GetDataByInternalID(idx_points[x]);
        const vdim_t *vec_idx_left_p_y = GetDataByInternalID(idx_points[y]);

        std::vector<id_t> ids_x_set, ids_y_set;
        ids_x_set.reserve(num_points);
        ids_y_set.reserve(num_points);

        for (size_t i = 0; i < num_points; i++)
        {
          const vdim_t *vec_idx_left_p_i = GetDataByInternalID(idx_points[left + i]);

          float dist_x = vec_L2sqr(vec_idx_left_p_x, vec_idx_left_p_i, D_);
          float dist_y = vec_L2sqr(vec_idx_left_p_y, vec_idx_left_p_i, D_);

          if (dist_x < dist_y)
          {
            ids_x_set.emplace_back(idx_points[left + i]);
          }
          else
          {
            ids_y_set.emplace_back(idx_points[left + i]);
          }
        }

        assert(ids_x_set.size() + ids_y_set.size() == num_points);

        // reorder idx_points
        size_t i = 0, j = 0;
        while (i < ids_x_set.size())
        {
          idx_points[left + i] = ids_x_set[i];
          i++;
        }
        while (j < ids_y_set.size())
        {
          idx_points[left + i] = ids_y_set[j];
          j++;
          i++;
        }

        CreateClusters(idx_points, left, left + ids_x_set.size() - 1, min_size_clusters, max_mst_degree);
        CreateClusters(idx_points, left + ids_x_set.size(), right, min_size_clusters, max_mst_degree);
      }
    }

    template <typename vdim_t>
    void HCNNG<vdim_t>::Search(
        const vdim_t *query,
        size_t k,
        size_t ef,
        std::priority_queue<std::pair<float, id_t>> & result)
    {
      assert(ef >= k);

      size_t comparison = 0;

      // auto vl = visited_list_pool_->GetFreeVisitedList();
      // auto visited_set = vl->mass_.data();
      // auto curr_visited = vl->curr_visited_;

      auto visited_set = std::make_unique<std::vector<bool>> (max_elements_, false);

      id_t ep = rand() % cur_element_count_;
      std::priority_queue<std::pair<float, id_t>> top_candidates;

      top_candidates.emplace(
        -vec_L2sqr(GetDataByInternalID(ep), query, D_),
        ep
      );
      comparison++;

      while (top_candidates.size())
      {
        auto [pstar_dist, pstar] = top_candidates.top();

        // visited_set[pstar] = curr_visited;          // update visited_set
        visited_set->at(pstar) = true;
        result.emplace(-pstar_dist, pstar);
        top_candidates.pop();

        {
          const auto & neighbors = adj_memory_[pstar];
          size_t num_neighbors = neighbors.size();
          
          for (size_t i = 0; i < num_neighbors; i++) {
            id_t neighbor_id = neighbors[i];
            if (visited_set->at(neighbor_id) == false) {
              top_candidates.emplace (
                - vec_L2sqr(GetDataByInternalID(neighbor_id), query, D_),
                neighbor_id
              );
              comparison++;
            }
          }
        }

        size_t candL = result.size() > ef ? 0 : ef - result.size();
        std::priority_queue<std::pair<float, id_t>> temp_candidates;
        while (candL--) {
          temp_candidates.emplace(top_candidates.top());
          top_candidates.pop();
        }
        top_candidates.swap(temp_candidates);
      }

      while (result.size() > k)
        result.pop();

      // visited_list_pool_->ReleaseVisitedList(vl);
      comparison_.fetch_add(comparison);
    }

    template <typename vdim_t>
    void HCNNG<vdim_t>::Search(
        const vdim_t *query,
        size_t k,
        size_t ef,
        id_t ep,
        std::priority_queue<std::pair<float, id_t>> &result)
    {
      
      assert(ef >= k);

      size_t comparison = 0;

      // auto vl = visited_list_pool_->GetFreeVisitedList();
      // auto visited_set = vl->mass_.data();
      // auto curr_visited = vl->curr_visited_;

      auto visited_set = std::make_unique<std::vector<bool>> (max_elements_, false);

      std::priority_queue<std::pair<float, id_t>> top_candidates;

      top_candidates.emplace(
        -vec_L2sqr(GetDataByInternalID(ep), query, D_),
        ep
      );
      comparison++;

      while (top_candidates.size())
      {
        auto [pstar_dist, pstar] = top_candidates.top();

        // visited_set[pstar] = curr_visited;          // update visited_set
        visited_set->at(pstar) = true;
        result.emplace(-pstar_dist, pstar);
        top_candidates.pop();

        {
          const auto & neighbors = adj_memory_[pstar];
          size_t num_neighbors = neighbors.size();
          
          for (size_t i = 0; i < num_neighbors; i++) {
            id_t neighbor_id = neighbors[i];
            if (visited_set->at(neighbor_id) == false) {
              top_candidates.emplace (
                - vec_L2sqr(GetDataByInternalID(neighbor_id), query, D_),
                neighbor_id
              );
              comparison++;
            }
          }
        }

        size_t candL = result.size() > ef ? 0 : ef - result.size();
        std::priority_queue<std::pair<float, id_t>> temp_candidates;
        while (candL--) {
          temp_candidates.emplace(top_candidates.top());
          top_candidates.pop();
        }
        top_candidates.swap(temp_candidates);
      }

      while (result.size() > k)
        result.pop();

      // visited_list_pool_->ReleaseVisitedList(vl);
      comparison_.fetch_add(comparison);
      
    }

    template <typename vdim_t>
    void HCNNG<vdim_t>::Search(
        const std::vector<std::vector<vdim_t>> &queries,
        size_t k,
        size_t ef,
        std::vector<std::vector<id_t>> &vids,
        std::vector<std::vector<float>> &dists)
    {
      const size_t nq = queries.size();
      vids.clear();
      dists.clear();
      vids.resize(nq);
      dists.resize(nq);

#pragma omp parallel for schedule(dynamic, 1) num_threads(num_threads_)
      for (size_t i = 0; i < nq; i++)
      {
        // std::cout << i << std::endl;
        const auto &query = queries[i];
        auto &vid = vids[i];
        auto &dist = dists[i];

        std::priority_queue<std::pair<float, id_t>> results;
        Search(query.data(), k, ef, results);
        // size_t actual_k = std::min(k, results.size());

        // while (results.size() > actual_k)
        //   results.pop();

        vid.reserve(k);
        dist.reserve(k);

        while (results.size())
        {
          const auto [d, id] = results.top();
          results.pop();

          vid.emplace_back(id);
          dist.emplace_back(-d);
        }
      }
    }

    template <typename vdim_t>
    void HCNNG<vdim_t>::PruneNeigh(size_t max_neigh)
    {
#pragma omp parallel for schedule(dynamic, 512) num_threads(num_threads_)
      for (id_t id = 0; id < cur_element_count_; id++)
      {
        const vdim_t *vec_curid = GetDataByInternalID(id);

        auto &neigh = adj_memory_[id];

        size_t new_size = std::min(neigh.size(), max_neigh);

        if (new_size == neigh.size())
          continue;

        std::vector<std::pair<float, id_t>> score;
        score.reserve(neigh.size());
        for (const auto &nid : neigh)
        {
          score.emplace_back(vec_L2sqr(GetDataByInternalID(nid), vec_curid, D_), nid);
        }

        std::sort(score.begin(), score.end());
        score.resize(new_size);
        score.shrink_to_fit();
        neigh.resize(new_size);
        neigh.shrink_to_fit();

        for (size_t i = 0; i < new_size; i++)
        {
          neigh[i] = score[i].second;
        }
      }
    }

    // template <typename vdim_t>
    // void HCNNG<vdim_t>::SaveHCNNG(
    //     const std::string &info_path, const std::string &edge_path, const std::string &data_path)
    // {

    //   SaveInfo(info_path);
    //   SaveData(data_path);
    //   SaveEdges(edge_path);
    // }

    template <typename vdim_t>
    size_t HCNNG<vdim_t>::GetComparisonAndClear()
    {
      return comparison_.exchange(0);
    }

    template <typename vdim_t>
    size_t HCNNG<vdim_t>::IndexSize() const
    {
      size_t sz = 0;
      // sz += cur_element_count_ * D_ * sizeof(vdim_t); // vector data
      for (id_t id = 0; id < cur_element_count_; id++)
      { // adj list
        sz += adj_memory_[id].size() * sizeof(id_t);
      }
      sz += data_memory_.size() * sizeof(vdim_t*);
      return sz;
    }

    template <typename vdim_t>
    id_t HCNNG<vdim_t>::GetClosestPoint(const vdim_t* data_point) {
      if (cur_element_count_ == 0)
      {
        throw std::runtime_error("empty graph");
      }
      size_t comparison = 0;
      id_t wander = 0;
      float dist = vec_L2sqr(data_point, GetDataByInternalID(wander), D_); 
      comparison++;
      bool moving = true;
      while (moving)
      {
        moving = false;
        const auto & adj = adj_memory_[wander];
        size_t n = adj.size();
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


    
    template <typename vdim_t> std::vector<float> 
    HCNNG<vdim_t>::GetSearchLength(
      const vdim_t *query,
      size_t k,
      size_t ef,
      std::priority_queue<std::pair<float, id_t>> &result)
    {
      assert(ef >= k);

      std::vector<float> length;

      size_t comparison = 0;

      // auto vl = visited_list_pool_->GetFreeVisitedList();
      // auto visited_set = vl->mass_.data();
      // auto curr_visited = vl->curr_visited_;

      auto visited_set = std::make_unique<std::vector<bool>> (max_elements_, false);

      id_t ep = 0;
      std::priority_queue<std::pair<float, id_t>> top_candidates;

      top_candidates.emplace(
        -vec_L2sqr(GetDataByInternalID(ep), query, D_),
        ep
      );
      comparison++;

      while (top_candidates.size())
      {
        auto [pstar_dist, pstar] = top_candidates.top();

        length.emplace_back(-pstar_dist);

        // visited_set[pstar] = curr_visited;          // update visited_set
        visited_set->at(pstar) = true;
        result.emplace(-pstar_dist, pstar);
        top_candidates.pop();

        {
          const auto & neighbors = adj_memory_[pstar];
          size_t num_neighbors = neighbors.size();
          
          for (size_t i = 0; i < num_neighbors; i++) {
            id_t neighbor_id = neighbors[i];
            if (visited_set->at(neighbor_id) == false) {
              top_candidates.emplace (
                - vec_L2sqr(GetDataByInternalID(neighbor_id), query, D_),
                neighbor_id
              );
              comparison++;
            }
          }
        }

        size_t candL = result.size() > ef ? 0 : ef - result.size();
        std::priority_queue<std::pair<float, id_t>> temp_candidates;
        while (candL--) {
          temp_candidates.emplace(top_candidates.top());
          top_candidates.pop();
        }
        top_candidates.swap(temp_candidates);
      }

      while (result.size() > k)
        result.pop();

      // visited_list_pool_->ReleaseVisitedList(vl);
      comparison_.fetch_add(comparison);

      return length;
    }

    template <typename vdim_t> std::vector<std::vector<float>>
    HCNNG<vdim_t>::GetSearchLength (
      const std::vector<std::vector<vdim_t>> &queries,
      size_t k,
      size_t ef,
      std::vector<std::vector<id_t>> &vids,
      std::vector<std::vector<float>> &dists
    )
    {
      const size_t nq = queries.size();
      vids.clear();
      dists.clear();
      vids.resize(nq);
      dists.resize(nq);

      std::vector<std::vector<float>> lengths(nq);
      
#pragma omp parallel for schedule(dynamic, 1) num_threads(num_threads_) 
      for (size_t i = 0; i < nq; i++)
      {
        // std::cout << i << std::endl;
        const auto &query = queries[i];
        auto &vid = vids[i];
        auto &dist = dists[i];

        std::priority_queue<std::pair<float, id_t>> results;
        lengths[i] = GetSearchLength(query.data(), k, ef, results);
        // size_t actual_k = std::min(k, results.size());

        // while (results.size() > actual_k)
        //   results.pop();

        vid.reserve(k);
        dist.reserve(k);

        while (results.size())
        {
          const auto [d, id] = results.top();
          results.pop();

          vid.emplace_back(id);
          dist.emplace_back(-d);
        }
      }

      return lengths;
    }

  } // namespace graph

} // namespace index
