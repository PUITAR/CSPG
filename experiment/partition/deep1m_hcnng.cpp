#include <iostream>
#include <fstream>
#include <graph/rpg.hpp>
#include <utils/binary_io.hpp>
#include <utils/stimer.hpp>
#include <utils/resize.hpp>
#include <utils/get_recall.hpp>

const std::string comma = ",";

using vdim_t = float;
using subgraph_t = puiann::graph::HCNNG<vdim_t>;

const size_t k = 10;
const size_t cases = 10;
const size_t num_threads = 24;

// Files
const std::string base_vectors_path = "/var/lib/docker/anns/dataset/deep1b/base.1M.fbin";
const std::string queries_vectors_path = "/var/lib/docker/anns/dataset/deep1b/query.public.10K.fbin";
const std::string groundtruth_path = "/var/lib/docker/anns/dataset/deep1b/gt_1M.ibin";

const std::string csv_path = "output/partition/deep1m_hcnng.csv";

struct params
{
  size_t num_random_clusters{10};
  size_t min_size_cluters{1000};
  size_t max_mst_degree{5};
  float dedup{0.5};
} default_params;

int main() {

  std::vector<vdim_t> base_vectors, queries_vectors;
  std::vector<id_t> groundtruth;
  auto [nb, d0] = utils::LoadFromFileBin(base_vectors, base_vectors_path);
  auto [nq, d1] = utils::LoadFromFileBin(queries_vectors, queries_vectors_path);
  auto [ng, d2] = utils::LoadFromFileBin(groundtruth, groundtruth_path);
  auto nest_queries_vectors = utils::Nest(std::move(queries_vectors), nq, d1);

  utils::STimer btimer;
  utils::STimer qtimer;

  std::ofstream csv(csv_path);

  csv << "index_type,num_partition,num_random_clusters,min_size_cluters,max_mst_degree,build_time,index_size,num_queries,efq,query_time,comparison,recall" << std::endl;

  for (size_t part = 1; part <= 16; part *= 2) {
    auto index = std::make_unique<puiann::graph::RandomPartitionGraph<vdim_t, subgraph_t>> (d0, part);
    btimer.Start();
    index->BuildIndex(base_vectors, num_threads, {
      default_params.num_random_clusters, default_params.min_size_cluters, default_params.max_mst_degree, default_params.dedup
    });
    btimer.Stop();
    for (size_t efq = 10; efq <= 300; efq += 10) {
      std::vector<std::vector<id_t>> knn;
      double qt = 0;
      size_t comparison;
      for (size_t t = 0; t < cases; t++) {
        qtimer.Reset(); 
        qtimer.Start();
        index->GetTopkNNParallel(nest_queries_vectors, k, num_threads, 1, efq, knn);
        qtimer.Stop();
        qt += qtimer.GetTime();
        comparison = index->GetComparisonAndClear();
      }
      double recall = utils::GetRecall(k, d2, groundtruth, knn);
      csv << "cspg" << comma << part << comma << default_params.num_random_clusters << comma << default_params.min_size_cluters << comma << default_params.max_mst_degree << comma
        << btimer.GetTime() << comma << index->IndexSize() << comma << nq << comma
        << 1 << "|" << efq << comma << qt/cases << comma << comparison << comma << recall << std::endl;
    }
  }

  return 0;
}
