#pragma once

#include <vector>
#include <algorithm>

#include <graph/diskann.hpp>
#include <graph/hnsw.hpp>
#include <graph/hcnng.hpp>
#include <graph/nsg.hpp>

#include <memory>

// #include <partition.hpp>
#include <utils/resize.hpp>
#include <utils/stimer.hpp>

// #include <any>

#include <cassert>

namespace anns
{

  namespace graph
  {

    template <typename vdim_t, class subgraph_t>
    class RandomPartitionGraph
    {
    public:
      std::vector<std::shared_ptr<DiskANN<vdim_t>>> diskann_s_;
      std::vector<std::shared_ptr<HNSW<vdim_t>>> hnsw_s_;
      std::vector<std::shared_ptr<HCNNG<vdim_t>>> hcnng_s_;
      std::vector<std::shared_ptr<NSG<vdim_t>>> nsg_s_;

      std::vector<std::vector<id_t>> refunction_;

      size_t dimension_{0};
      size_t num_partition_{0};
      size_t num_points_{0};

      size_t num_dedup_{0};

      std::atomic<size_t> comparison_{0};

      // std::shared_ptr<std::vector<vdim_t>> data_{nullptr};

      explicit RandomPartitionGraph(size_t dimension, size_t num_partition) : dimension_(dimension), num_partition_(num_partition)
      {
        assert(
            typeid(subgraph_t) == typeid(DiskANN<vdim_t>) ||
            typeid(subgraph_t) == typeid(HNSW<vdim_t>) ||
            typeid(subgraph_t) == typeid(HCNNG<vdim_t>) ||
            typeid(subgraph_t) == typeid(NSG<vdim_t>));
        assert(num_partition_);
      }

      /* Parameters pack for the different indexes
       * DiskANN {
       *   max_graph_degree, alpha, L
       *   [dedup]
       * }
       * HNSW {
       *   M, efc,
       *   [dedup]
       * }
       * HCNNG {
       *   num_random_clusters, min_size_clusters, max_mst_degree,
       *   [dedup]
       * }
       * NSG {
       *   max_graph_degree, L,
       *   [dedup]
       * }
       */
      void BuildIndex(const std::vector<vdim_t> &rawdata, size_t num_threads, const std::vector<float> &params)
      {
        // data random partition
        num_points_ = rawdata.size() / dimension_;
        // std::cout << "num_threads " << num_threads << std::endl;
        if (typeid(subgraph_t) == typeid(DiskANN<vdim_t>))
        {
          num_dedup_ = static_cast<size_t>(num_points_ * params[3]);
        }
        else if (typeid(subgraph_t) == typeid(HNSW<vdim_t>))
        {
          num_dedup_ = static_cast<size_t>(num_points_ * params[2]);
        }
        else if (typeid(subgraph_t) == typeid(HCNNG<vdim_t>))
        {
          num_dedup_ = static_cast<size_t>(num_points_ * params[3]);
        }
        else if (typeid(subgraph_t) == typeid(NSG<vdim_t>))
        {
          num_dedup_ = static_cast<size_t>(num_points_ * params[2]);
        }
        else
        {
          throw std::runtime_error("err index type");
        }
        refunction_.resize(num_partition_, std::vector<id_t>{});
        for (id_t id = 0; id < num_dedup_; id++)
        {
          for (size_t i = 0; i < num_partition_; i++)
            refunction_[i].emplace_back(id);
        }
        for (id_t id = num_dedup_; id < num_points_; id++)
        {
          refunction_[rand() % num_partition_].emplace_back(id);
        }
        auto pdata = std::make_unique<std::vector<std::vector<const vdim_t *>>>(num_partition_);
        std::for_each(pdata->begin(), pdata->end(), [=](auto &arr)
                      { arr.reserve(num_points_ / num_partition_); });
        for (size_t i = 0; i < num_partition_; i++)
        {
          for (id_t id : refunction_[i])
          {
            // pdata->at(i).insert(pdata->at(i).end(), rawdata.begin()+id*dimension_, rawdata.begin()+(id+1)*dimension_);
            pdata->at(i).emplace_back(rawdata.data() + id * dimension_);
          }
        }
        // new index
        diskann_s_.resize(0, nullptr);
        hnsw_s_.resize(0, nullptr);
        hcnng_s_.resize(0, nullptr);
        nsg_s_.resize(0, nullptr);
        for (size_t i = 0; i < num_partition_; i++)
        {
          if (typeid(subgraph_t) == typeid(DiskANN<vdim_t>))
          {
            
            diskann_s_.emplace_back(std::make_shared<DiskANN<vdim_t>>(
                dimension_, pdata->at(i).size(), static_cast<size_t>(params[0])));
          }
          else if (typeid(subgraph_t) == typeid(HNSW<vdim_t>))
          {
            hnsw_s_.emplace_back(std::make_shared<HNSW<vdim_t>>(
                dimension_, pdata->at(i).size(), static_cast<size_t>(params[0]), static_cast<size_t>(params[1])));
          }
          else if (typeid(subgraph_t) == typeid(HCNNG<vdim_t>))
          {
            hcnng_s_.emplace_back(std::make_shared<HCNNG<vdim_t>>(dimension_, pdata->at(i).size()));
          }
          else if (typeid(subgraph_t) == typeid(NSG<vdim_t>))
          {
            nsg_s_.emplace_back(std::make_shared<NSG<vdim_t>>(
                dimension_, pdata->at(i).size(), static_cast<size_t>(params[0])));
          }
          else
          {
            throw std::runtime_error("err index type");
          }
        }

        // std::cout << "Index Building" << std::endl;
        // build index
        for (size_t i = 0; i < num_partition_; i++)
        {
          if (typeid(subgraph_t) == typeid(DiskANN<vdim_t>))
          {
            diskann_s_[i]->SetNumThreads(num_threads);
            diskann_s_[i]->BuildIndex(pdata->at(i), static_cast<float>(params[1]), static_cast<size_t>(params[2]));
          }
          else if (typeid(subgraph_t) == typeid(HNSW<vdim_t>))
          {
            hnsw_s_[i]->SetNumThreads(num_threads);
            hnsw_s_[i]->Populate(pdata->at(i));
          }
          else if (typeid(subgraph_t) == typeid(HCNNG<vdim_t>))
          {
            hcnng_s_[i]->SetNumThreads(num_threads);
            hcnng_s_[i]->CreateHCNNG(pdata->at(i), static_cast<size_t>(params[0]), static_cast<size_t>(params[1]), static_cast<size_t>(params[2]));
          }
          else if (typeid(subgraph_t) == typeid(NSG<vdim_t>))
          {
            nsg_s_[i]->SetNumThreads(num_threads);
            nsg_s_[i]->BuildIndex(pdata->at(i), static_cast<size_t>(params[1]));
          }
          else
          {
            throw std::runtime_error("err index type");
          }
          // std::cout << "Subgraph_" << i << " was Built" << std::endl;
        }

        // std::cout << "Index finished" << std::endl;
      }

      void GetTopkNNParallel(const std::vector<std::vector<vdim_t>> &queries, size_t k,
                             size_t num_threads, size_t ef1, size_t ef2, std::vector<std::vector<id_t>> &knn)
      {
// fetch element from a given tuple
#define TUP_DIS(tup) (std::get<0>(tup))
#define TUP_GID(tup) (std::get<1>(tup))
#define TUP_VID(tup) (std::get<2>(tup))

        knn.assign(queries.size(), {});

        if (num_partition_ == 0)
          throw std::runtime_error("RPG index is not initialized");

        // searching process
        if (typeid(subgraph_t) == typeid(DiskANN<vdim_t>))
        {
#pragma omp parallel for schedule(dynamic, 64) num_threads(num_threads)
          for (size_t q = 0; q < queries.size(); q++)
          {
            size_t comparison = 0;
            const auto qv = queries[q].data();
            // tuple (distance, subgraph_id, vector_id_in_subgraph)
            std::priority_queue<std::tuple<float, id_t, id_t>> minheap;
            std::priority_queue<std::tuple<float, id_t, id_t>> maxheap;
            std::vector<std::vector<bool>> visited_points(num_partition_);
            for (size_t i = 0; i < num_partition_; i++)
            {
              visited_points[i].assign(diskann_s_[i]->cur_element_count_, false);
            }

            id_t closest_point;
            if (ef1 > 1)
            {
              std::vector<std::pair<float, id_t>> tmp;
              diskann_s_[0]->Search(qv, 1, ef1, tmp);
              closest_point = tmp.front().second;
            }
            else
            {
              closest_point = diskann_s_[0]->GetClosestPoint(qv);
            }

            float d = vec_L2sqr(qv, diskann_s_[0]->GetDataByInternalID(closest_point), dimension_);
            comparison++;
            visited_points[0][closest_point] = true;
            maxheap.emplace(d, 0, closest_point);
            minheap.emplace(-d, 0, closest_point);
            float bound = d;
            while (minheap.size())
            {
              auto [d, gid, vid] = minheap.top();
              minheap.pop();
              if (-d > bound && maxheap.size() >= ef2)
                break;
              auto &visited = visited_points[gid];
              auto &subgraph = diskann_s_[gid];
              size_t *ll = (size_t *)subgraph->GetLinkByInternalID(vid);
              size_t sz = *ll;
              id_t *adj = (id_t *)(ll + 1);
              for (size_t j = 0; j < sz; j++)
              {
                id_t nid = adj[j];
                if (visited[nid] == false)
                {
                  visited[nid] = true;
                  float dn = vec_L2sqr(qv, subgraph->GetDataByInternalID(nid), dimension_);
                  comparison++;
                  if (TUP_DIS(maxheap.top()) > dn || maxheap.size() < ef2)
                  {
                    maxheap.emplace(dn, gid, nid);
                    minheap.emplace(-dn, gid, nid);
                    // extend skip vectors
                    if (nid < num_dedup_)
                    {
                      for (size_t n = 0; n < gid; n++)
                      {
                        visited_points[n][nid] = true;
                        maxheap.emplace(dn, n, nid);
                        minheap.emplace(-dn, n, nid);
                      }
                      for (size_t n = gid + 1; n < num_partition_; n++)
                      {
                        visited_points[n][nid] = true;
                        maxheap.emplace(dn, n, nid);
                        minheap.emplace(-dn, n, nid);
                      }
                    }
                  }
                }
                while (maxheap.size() > ef2)
                  maxheap.pop();
                bound = TUP_DIS(maxheap.top());
              }
            }
            // pack results into `knn`
            while (maxheap.size() > k)
              maxheap.pop();
            while (maxheap.size())
            {
              const auto &[_, gid, vid] = maxheap.top();
              knn[q].emplace_back(refunction_[gid][vid]);
              maxheap.pop();
            }
            comparison_.fetch_add(comparison + diskann_s_[0]->GetComparisonAndClear());
          }
        }
        else if (typeid(subgraph_t) == typeid(HNSW<vdim_t>))
        {
#pragma omp parallel for schedule(dynamic, 64) num_threads(num_threads)
          for (size_t q = 0; q < queries.size(); q++)
          {
            size_t comparison = 0;
            const auto qv = queries[q].data();
            // tuple (distance, subgraph_id, vector_id_in_subgraph)
            std::priority_queue<std::tuple<float, id_t, id_t>> minheap;
            std::priority_queue<std::tuple<float, id_t, id_t>> maxheap;
            std::vector<std::vector<bool>> visited_points(num_partition_);
            for (size_t i = 0; i < num_partition_; i++)
            {
              visited_points[i].assign(hnsw_s_[i]->cur_element_count_, false);
            }

            id_t closest_point;
            if (ef1 > 1)
            {
              auto tmp = hnsw_s_[0]->Search(qv, 1, ef1);
              closest_point = tmp.top().second;
            }
            else
            {
              closest_point = hnsw_s_[0]->GetClosestPoint(qv);
            }

            float d = vec_L2sqr(qv, hnsw_s_[0]->GetDataByInternalID(closest_point), dimension_);
            comparison++;
            visited_points[0][closest_point] = true;
            maxheap.emplace(d, 0, closest_point);
            minheap.emplace(-d, 0, closest_point);
            float bound = d;
            while (minheap.size())
            {
              auto [d, gid, vid] = minheap.top();
              minheap.pop();
              if (-d > bound && maxheap.size() >= ef2)
                break;
              auto &visited = visited_points[gid];
              auto &subgraph = hnsw_s_[gid];
              size_t *ll = (size_t *)subgraph->GetLinkByInternalID(vid, 0);
              size_t sz = *ll;
              id_t *adj = (id_t *)(ll + 1);
              for (size_t j = 0; j < sz; j++)
              {
                id_t nid = adj[j];
                if (visited[nid] == false)
                {
                  visited[nid] = true;
                  float dn = vec_L2sqr(qv, subgraph->GetDataByInternalID(nid), dimension_);
                  comparison++;
                  if (TUP_DIS(maxheap.top()) > dn || maxheap.size() < ef2)
                  {
                    maxheap.emplace(dn, gid, nid);
                    minheap.emplace(-dn, gid, nid);
                    // extend skip vectors
                    if (nid < num_dedup_)
                    {
                      for (size_t n = 0; n < gid; n++)
                      {
                        visited_points[n][nid] = true;
                        maxheap.emplace(dn, n, nid);
                        minheap.emplace(-dn, n, nid);
                      }
                      for (size_t n = gid + 1; n < num_partition_; n++)
                      {
                        visited_points[n][nid] = true;
                        maxheap.emplace(dn, n, nid);
                        minheap.emplace(-dn, n, nid);
                      }
                    }
                  }
                }
                while (maxheap.size() > ef2)
                  maxheap.pop();
                bound = TUP_DIS(maxheap.top());
              }
            }
            // pack results into `knn`
            while (maxheap.size() > k)
              maxheap.pop();
            while (maxheap.size())
            {
              const auto &[_, gid, vid] = maxheap.top();
              knn[q].emplace_back(refunction_[gid][vid]);
              maxheap.pop();
            }
            comparison_.fetch_add(comparison + hnsw_s_[0]->GetComparisonAndClear());
          }
        }
        else if (typeid(subgraph_t) == typeid(HCNNG<vdim_t>))
        {
#pragma omp parallel for schedule(dynamic, 64) num_threads(num_threads)
          for (size_t q = 0; q < queries.size(); q++)
          {
            size_t comparison = 0;
            const auto qv = queries[q].data();
            // tuple (distance, subgraph_id, vector_id_in_subgraph)
            std::priority_queue<std::tuple<float, id_t, id_t>> minheap;
            std::priority_queue<std::tuple<float, id_t, id_t>> maxheap;
            std::vector<std::vector<bool>> visited_points(num_partition_);
            for (size_t i = 0; i < num_partition_; i++)
            {
              visited_points[i].assign(hcnng_s_[i]->cur_element_count_, false);
            }

            id_t closest_point;
            if (ef1 > 1)
            {
              std::priority_queue<std::pair<float, id_t>> tmp;
              hcnng_s_[0]->Search(qv, 1, ef1, tmp);
              closest_point = tmp.top().second;
            }
            else
            {
              closest_point = hcnng_s_[0]->GetClosestPoint(qv);
            }

            float d = vec_L2sqr(qv, hcnng_s_[0]->GetDataByInternalID(closest_point), dimension_);
            comparison++;
            visited_points[0][closest_point] = true;
            maxheap.emplace(d, 0, closest_point);
            minheap.emplace(-d, 0, closest_point);
            float bound = d;
            while (minheap.size())
            {
              auto [d, gid, vid] = minheap.top();
              minheap.pop();
              if (-d > bound && maxheap.size() >= ef2)
                break;
              auto &visited = visited_points[gid];
              auto &subgraph = hcnng_s_[gid];
              const auto &adj = subgraph->adj_memory_[vid];
              const size_t sz = adj.size();
              for (size_t j = 0; j < sz; j++)
              {
                id_t nid = adj[j];
                if (visited[nid] == false)
                {
                  visited[nid] = true;
                  float dn = vec_L2sqr(qv, subgraph->GetDataByInternalID(nid), dimension_);
                  comparison++;
                  if (TUP_DIS(maxheap.top()) > dn || maxheap.size() < ef2)
                  {
                    maxheap.emplace(dn, gid, nid);
                    minheap.emplace(-dn, gid, nid);
                    // extend skip vectors
                    if (nid < num_dedup_)
                    {
                      for (size_t n = 0; n < gid; n++)
                      {
                        visited_points[n][nid] = true;
                        maxheap.emplace(dn, n, nid);
                        minheap.emplace(-dn, n, nid);
                      }
                      for (size_t n = gid + 1; n < num_partition_; n++)
                      {
                        visited_points[n][nid] = true;
                        maxheap.emplace(dn, n, nid);
                        minheap.emplace(-dn, n, nid);
                      }
                    }
                  }
                }
                while (maxheap.size() > ef2)
                  maxheap.pop();
                bound = TUP_DIS(maxheap.top());
              }
            }
            // pack results into `knn`
            while (maxheap.size() > k)
              maxheap.pop();
            while (maxheap.size())
            {
              const auto &[_, gid, vid] = maxheap.top();
              knn[q].emplace_back(refunction_[gid][vid]);
              maxheap.pop();
            }
            comparison_.fetch_add(comparison + hcnng_s_[0]->GetComparisonAndClear());
          }
        }
        else if (typeid(subgraph_t) == typeid(NSG<vdim_t>))
        {
#pragma omp parallel for schedule(dynamic, 64) num_threads(num_threads)
          for (size_t q = 0; q < queries.size(); q++)
          {
            size_t comparison = 0;
            const auto qv = queries[q].data();
            // tuple (distance, subgraph_id, vector_id_in_subgraph)
            std::priority_queue<std::tuple<float, id_t, id_t>> minheap;
            std::priority_queue<std::tuple<float, id_t, id_t>> maxheap;
            std::vector<std::vector<bool>> visited_points(num_partition_);
            for (size_t i = 0; i < num_partition_; i++)
            {
              visited_points[i].assign(nsg_s_[i]->cur_element_count_, false);
            }

            id_t closest_point;
            if (ef1 > 1)
            {
              std::vector<std::pair<float, id_t>> tmp;
              nsg_s_[0]->Search(qv, 1, ef1, tmp);
              closest_point = tmp.front().second;
            }
            else
            {
              closest_point = nsg_s_[0]->GetClosestPoint(qv);
            }

            float d = vec_L2sqr(qv, nsg_s_[0]->GetDataByInternalID(closest_point), dimension_);
            comparison++;
            visited_points[0][closest_point] = true;
            maxheap.emplace(d, 0, closest_point);
            minheap.emplace(-d, 0, closest_point);
            float bound = d;
            while (minheap.size())
            {
              auto [d, gid, vid] = minheap.top();
              minheap.pop();
              if (-d > bound && maxheap.size() >= ef2)
                break;
              auto &visited = visited_points[gid];
              auto &subgraph = nsg_s_[gid];
              size_t *ll = (size_t *)subgraph->GetLinkByInternalID(vid);
              size_t sz = *ll;
              id_t *adj = (id_t *)(ll + 1);
              for (size_t j = 0; j < sz; j++)
              {
                id_t nid = adj[j];
                if (visited[nid] == false)
                {
                  visited[nid] = true;
                  float dn = vec_L2sqr(qv, subgraph->GetDataByInternalID(nid), dimension_);
                  comparison++;
                  if (TUP_DIS(maxheap.top()) > dn || maxheap.size() < ef2)
                  {
                    maxheap.emplace(dn, gid, nid);
                    minheap.emplace(-dn, gid, nid);
                    // extend skip vectors
                    if (nid < num_dedup_)
                    {
                      for (size_t n = 0; n < gid; n++)
                      {
                        visited_points[n][nid] = true;
                        maxheap.emplace(dn, n, nid);
                        minheap.emplace(-dn, n, nid);
                      }
                      for (size_t n = gid + 1; n < num_partition_; n++)
                      {
                        visited_points[n][nid] = true;
                        maxheap.emplace(dn, n, nid);
                        minheap.emplace(-dn, n, nid);
                      }
                    }
                  }
                }
                while (maxheap.size() > ef2)
                  maxheap.pop();
                bound = TUP_DIS(maxheap.top());
              }
            }
            // pack results into `knn`
            while (maxheap.size() > k)
              maxheap.pop();
            while (maxheap.size())
            {
              const auto &[_, gid, vid] = maxheap.top();
              knn[q].emplace_back(refunction_[gid][vid]);
              maxheap.pop();
            }
            comparison_.fetch_add(comparison + nsg_s_[0]->GetComparisonAndClear());
          }
        }
        else
        {
          throw std::runtime_error("err index type");
        }
        // finish all queries
      }

      std::vector<std::vector<id_t>> GetTopkNNParallel2(
        const std::vector<vdim_t> &queries, size_t k, size_t num_threads, size_t ef1, size_t ef2)
      {
        std::vector<std::vector<id_t>> knn;

// fetch element from a given tuple
#define TUP_DIS(tup) (std::get<0>(tup))
#define TUP_GID(tup) (std::get<1>(tup))
#define TUP_VID(tup) (std::get<2>(tup))

        size_t nq = queries.size() / dimension_;

        knn.assign(nq, {});

        if (num_partition_ == 0)
          throw std::runtime_error("RPG index is not initialized");

        // searching process
        if (typeid(subgraph_t) == typeid(DiskANN<vdim_t>))
        {
#pragma omp parallel for schedule(dynamic, 64) num_threads(num_threads)
          for (size_t q = 0; q < nq; q++)
          {
            size_t comparison = 0;
            const auto qv = queries.data() + q * dimension_;
            // tuple (distance, subgraph_id, vector_id_in_subgraph)
            std::priority_queue<std::tuple<float, id_t, id_t>> minheap;
            std::priority_queue<std::tuple<float, id_t, id_t>> maxheap;
            std::vector<std::vector<bool>> visited_points(num_partition_);
            for (size_t i = 0; i < num_partition_; i++)
            {
              visited_points[i].assign(diskann_s_[i]->cur_element_count_, false);
            }

            id_t closest_point;
            if (ef1 > 1)
            {
              std::vector<std::pair<float, id_t>> tmp;
              diskann_s_[0]->Search(qv, 1, ef1, tmp);
              closest_point = tmp.front().second;
            }
            else
            {
              closest_point = diskann_s_[0]->GetClosestPoint(qv);
            }

            float d = vec_L2sqr(qv, diskann_s_[0]->GetDataByInternalID(closest_point), dimension_);
            comparison++;
            visited_points[0][closest_point] = true;
            maxheap.emplace(d, 0, closest_point);
            minheap.emplace(-d, 0, closest_point);
            float bound = d;
            while (minheap.size())
            {
              auto [d, gid, vid] = minheap.top();
              minheap.pop();
              if (-d > bound && maxheap.size() >= ef2)
                break;
              auto &visited = visited_points[gid];
              auto &subgraph = diskann_s_[gid];
              size_t *ll = (size_t *)subgraph->GetLinkByInternalID(vid);
              size_t sz = *ll;
              id_t *adj = (id_t *)(ll + 1);
              for (size_t j = 0; j < sz; j++)
              {
                id_t nid = adj[j];
                if (visited[nid] == false)
                {
                  visited[nid] = true;
                  float dn = vec_L2sqr(qv, subgraph->GetDataByInternalID(nid), dimension_);
                  comparison++;
                  if (TUP_DIS(maxheap.top()) > dn || maxheap.size() < ef2)
                  {
                    maxheap.emplace(dn, gid, nid);
                    minheap.emplace(-dn, gid, nid);
                    // extend skip vectors
                    if (nid < num_dedup_)
                    {
                      for (size_t n = 0; n < gid; n++)
                      {
                        visited_points[n][nid] = true;
                        maxheap.emplace(dn, n, nid);
                        minheap.emplace(-dn, n, nid);
                      }
                      for (size_t n = gid + 1; n < num_partition_; n++)
                      {
                        visited_points[n][nid] = true;
                        maxheap.emplace(dn, n, nid);
                        minheap.emplace(-dn, n, nid);
                      }
                    }
                  }
                }
                while (maxheap.size() > ef2)
                  maxheap.pop();
                bound = TUP_DIS(maxheap.top());
              }
            }
            // pack results into `knn`
            while (maxheap.size() > k)
              maxheap.pop();
            while (maxheap.size())
            {
              const auto &[_, gid, vid] = maxheap.top();
              knn[q].emplace_back(refunction_[gid][vid]);
              maxheap.pop();
            }
            comparison_.fetch_add(comparison + diskann_s_[0]->GetComparisonAndClear());
          }
        }
        else if (typeid(subgraph_t) == typeid(HNSW<vdim_t>))
        {
#pragma omp parallel for schedule(dynamic, 64) num_threads(num_threads)
          for (size_t q = 0; q < nq; q++)
          {
            size_t comparison = 0;
            const auto qv = queries.data() + q * dimension_;
            // tuple (distance, subgraph_id, vector_id_in_subgraph)
            std::priority_queue<std::tuple<float, id_t, id_t>> minheap;
            std::priority_queue<std::tuple<float, id_t, id_t>> maxheap;
            std::vector<std::vector<bool>> visited_points(num_partition_);
            for (size_t i = 0; i < num_partition_; i++)
            {
              visited_points[i].assign(hnsw_s_[i]->cur_element_count_, false);
            }

            id_t closest_point;
            if (ef1 > 1)
            {
              auto tmp = hnsw_s_[0]->Search(qv, 1, ef1);
              closest_point = tmp.top().second;
            }
            else
            {
              closest_point = hnsw_s_[0]->GetClosestPoint(qv);
            }

            float d = vec_L2sqr(qv, hnsw_s_[0]->GetDataByInternalID(closest_point), dimension_);
            comparison++;
            visited_points[0][closest_point] = true;
            maxheap.emplace(d, 0, closest_point);
            minheap.emplace(-d, 0, closest_point);
            float bound = d;
            while (minheap.size())
            {
              auto [d, gid, vid] = minheap.top();
              minheap.pop();
              if (-d > bound && maxheap.size() >= ef2)
                break;
              auto &visited = visited_points[gid];
              auto &subgraph = hnsw_s_[gid];
              size_t *ll = (size_t *)subgraph->GetLinkByInternalID(vid, 0);
              size_t sz = *ll;
              id_t *adj = (id_t *)(ll + 1);
              for (size_t j = 0; j < sz; j++)
              {
                id_t nid = adj[j];
                if (visited[nid] == false)
                {
                  visited[nid] = true;
                  float dn = vec_L2sqr(qv, subgraph->GetDataByInternalID(nid), dimension_);
                  comparison++;
                  if (TUP_DIS(maxheap.top()) > dn || maxheap.size() < ef2)
                  {
                    maxheap.emplace(dn, gid, nid);
                    minheap.emplace(-dn, gid, nid);
                    // extend skip vectors
                    if (nid < num_dedup_)
                    {
                      for (size_t n = 0; n < gid; n++)
                      {
                        visited_points[n][nid] = true;
                        maxheap.emplace(dn, n, nid);
                        minheap.emplace(-dn, n, nid);
                      }
                      for (size_t n = gid + 1; n < num_partition_; n++)
                      {
                        visited_points[n][nid] = true;
                        maxheap.emplace(dn, n, nid);
                        minheap.emplace(-dn, n, nid);
                      }
                    }
                  }
                }
                while (maxheap.size() > ef2)
                  maxheap.pop();
                bound = TUP_DIS(maxheap.top());
              }
            }
            // pack results into `knn`
            while (maxheap.size() > k)
              maxheap.pop();
            while (maxheap.size())
            {
              const auto &[_, gid, vid] = maxheap.top();
              knn[q].emplace_back(refunction_[gid][vid]);
              maxheap.pop();
            }
            comparison_.fetch_add(comparison + hnsw_s_[0]->GetComparisonAndClear());
          }
        }
        else if (typeid(subgraph_t) == typeid(HCNNG<vdim_t>))
        {
#pragma omp parallel for schedule(dynamic, 64) num_threads(num_threads)
          for (size_t q = 0; q < nq; q++)
          {
            size_t comparison = 0;
            const auto qv = queries.data() + q * dimension_;
            // tuple (distance, subgraph_id, vector_id_in_subgraph)
            std::priority_queue<std::tuple<float, id_t, id_t>> minheap;
            std::priority_queue<std::tuple<float, id_t, id_t>> maxheap;
            std::vector<std::vector<bool>> visited_points(num_partition_);
            for (size_t i = 0; i < num_partition_; i++)
            {
              visited_points[i].assign(hcnng_s_[i]->cur_element_count_, false);
            }

            id_t closest_point;
            if (ef1 > 1)
            {
              std::priority_queue<std::pair<float, id_t>> tmp;
              hcnng_s_[0]->Search(qv, 1, ef1, tmp);
              closest_point = tmp.top().second;
            }
            else
            {
              closest_point = hcnng_s_[0]->GetClosestPoint(qv);
            }

            float d = vec_L2sqr(qv, hcnng_s_[0]->GetDataByInternalID(closest_point), dimension_);
            comparison++;
            visited_points[0][closest_point] = true;
            maxheap.emplace(d, 0, closest_point);
            minheap.emplace(-d, 0, closest_point);
            float bound = d;
            while (minheap.size())
            {
              auto [d, gid, vid] = minheap.top();
              minheap.pop();
              if (-d > bound && maxheap.size() >= ef2)
                break;
              auto &visited = visited_points[gid];
              auto &subgraph = hcnng_s_[gid];
              const auto &adj = subgraph->adj_memory_[vid];
              const size_t sz = adj.size();
              for (size_t j = 0; j < sz; j++)
              {
                id_t nid = adj[j];
                if (visited[nid] == false)
                {
                  visited[nid] = true;
                  float dn = vec_L2sqr(qv, subgraph->GetDataByInternalID(nid), dimension_);
                  comparison++;
                  if (TUP_DIS(maxheap.top()) > dn || maxheap.size() < ef2)
                  {
                    maxheap.emplace(dn, gid, nid);
                    minheap.emplace(-dn, gid, nid);
                    // extend skip vectors
                    if (nid < num_dedup_)
                    {
                      for (size_t n = 0; n < gid; n++)
                      {
                        visited_points[n][nid] = true;
                        maxheap.emplace(dn, n, nid);
                        minheap.emplace(-dn, n, nid);
                      }
                      for (size_t n = gid + 1; n < num_partition_; n++)
                      {
                        visited_points[n][nid] = true;
                        maxheap.emplace(dn, n, nid);
                        minheap.emplace(-dn, n, nid);
                      }
                    }
                  }
                }
                while (maxheap.size() > ef2)
                  maxheap.pop();
                bound = TUP_DIS(maxheap.top());
              }
            }
            // pack results into `knn`
            while (maxheap.size() > k)
              maxheap.pop();
            while (maxheap.size())
            {
              const auto &[_, gid, vid] = maxheap.top();
              knn[q].emplace_back(refunction_[gid][vid]);
              maxheap.pop();
            }
            comparison_.fetch_add(comparison + hcnng_s_[0]->GetComparisonAndClear());
          }
        }
        else if (typeid(subgraph_t) == typeid(NSG<vdim_t>))
        {
#pragma omp parallel for schedule(dynamic, 64) num_threads(num_threads)
          for (size_t q = 0; q < nq; q++)
          {
            size_t comparison = 0;
            const auto qv = queries.data() + q * dimension_;
            // tuple (distance, subgraph_id, vector_id_in_subgraph)
            std::priority_queue<std::tuple<float, id_t, id_t>> minheap;
            std::priority_queue<std::tuple<float, id_t, id_t>> maxheap;
            std::vector<std::vector<bool>> visited_points(num_partition_);
            for (size_t i = 0; i < num_partition_; i++)
            {
              visited_points[i].assign(nsg_s_[i]->cur_element_count_, false);
            }

            id_t closest_point;
            if (ef1 > 1)
            {
              std::vector<std::pair<float, id_t>> tmp;
              nsg_s_[0]->Search(qv, 1, ef1, tmp);
              closest_point = tmp.front().second;
            }
            else
            {
              closest_point = nsg_s_[0]->GetClosestPoint(qv);
            }

            float d = vec_L2sqr(qv, nsg_s_[0]->GetDataByInternalID(closest_point), dimension_);
            comparison++;
            visited_points[0][closest_point] = true;
            maxheap.emplace(d, 0, closest_point);
            minheap.emplace(-d, 0, closest_point);
            float bound = d;
            while (minheap.size())
            {
              auto [d, gid, vid] = minheap.top();
              minheap.pop();
              if (-d > bound && maxheap.size() >= ef2)
                break;
              auto &visited = visited_points[gid];
              auto &subgraph = nsg_s_[gid];
              size_t *ll = (size_t *)subgraph->GetLinkByInternalID(vid);
              size_t sz = *ll;
              id_t *adj = (id_t *)(ll + 1);
              for (size_t j = 0; j < sz; j++)
              {
                id_t nid = adj[j];
                if (visited[nid] == false)
                {
                  visited[nid] = true;
                  float dn = vec_L2sqr(qv, subgraph->GetDataByInternalID(nid), dimension_);
                  comparison++;
                  if (TUP_DIS(maxheap.top()) > dn || maxheap.size() < ef2)
                  {
                    maxheap.emplace(dn, gid, nid);
                    minheap.emplace(-dn, gid, nid);
                    // extend skip vectors
                    if (nid < num_dedup_)
                    {
                      for (size_t n = 0; n < gid; n++)
                      {
                        visited_points[n][nid] = true;
                        maxheap.emplace(dn, n, nid);
                        minheap.emplace(-dn, n, nid);
                      }
                      for (size_t n = gid + 1; n < num_partition_; n++)
                      {
                        visited_points[n][nid] = true;
                        maxheap.emplace(dn, n, nid);
                        minheap.emplace(-dn, n, nid);
                      }
                    }
                  }
                }
                while (maxheap.size() > ef2)
                  maxheap.pop();
                bound = TUP_DIS(maxheap.top());
              }
            }
            // pack results into `knn`
            while (maxheap.size() > k)
              maxheap.pop();
            while (maxheap.size())
            {
              const auto &[_, gid, vid] = maxheap.top();
              knn[q].emplace_back(refunction_[gid][vid]);
              maxheap.pop();
            }
            comparison_.fetch_add(comparison + nsg_s_[0]->GetComparisonAndClear());
          }
        }
        else
        {
          throw std::runtime_error("err index type");
        }
        // finish all queries

        return knn;
      }

      size_t IndexSize() const
      {
        size_t sz = 0;
        for (size_t i = 0; i < num_partition_; i++)
        {
          if (typeid(subgraph_t) == typeid(DiskANN<vdim_t>))
          {
            sz += diskann_s_[i]->IndexSize();
          }
          else if (typeid(subgraph_t) == typeid(HNSW<vdim_t>))
          {
            sz += hnsw_s_[i]->IndexSize();
          }
          else if (typeid(subgraph_t) == typeid(HCNNG<vdim_t>))
          {
            sz += hcnng_s_[i]->IndexSize();
          }
          else if (typeid(subgraph_t) == typeid(NSG<vdim_t>))
          {
            sz += nsg_s_[i]->IndexSize();
          }
          else
          {
            throw std::runtime_error("err index type");
          }
        }
        return sz;
      }

      size_t GetComparisonAndClear() { return comparison_.exchange(0); }
    };

  } // namespace graph

} // namespace anns
