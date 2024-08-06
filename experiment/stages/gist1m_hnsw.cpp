#include <iostream>
#include <fstream>
#include <graph/rpg.hpp>
#include <utils/binary_io.hpp>
#include <utils/stimer.hpp>
#include <utils/resize.hpp>
#include <utils/get_recall.hpp>

const std::string comma = ",";

using vdim_t = float;
using subgraph_t = anns::graph::HNSW<vdim_t>;

const size_t k = 10;
const size_t cases = 10;
const size_t num_threads = 24;

const size_t part = 2;

// Files
const std::string base_vectors_path = "/var/lib/docker/anns/dataset/gist1m/base.fvecs";
const std::string queries_vectors_path = "/var/lib/docker/anns/query/gist1m/query.fvecs";
const std::string groundtruth_path = "/var/lib/docker/anns/query/gist1m/gt.ivecs";

const std::string csv_path = "output/stages/gist1m_hnsw.csv";

struct params
{
  size_t max_degree{32};
  size_t efc{128};
  float dedup{0.5};
} default_params;

int main() {

  std::vector<vdim_t> base_vectors, queries_vectors;
  std::vector<id_t> groundtruth;
  auto [nb, d0] = utils::LoadFromFile(base_vectors, base_vectors_path);
  auto [nq, d1] = utils::LoadFromFile(queries_vectors, queries_vectors_path);
  auto [ng, d2] = utils::LoadFromFile(groundtruth, groundtruth_path);
  auto nest_queries_vectors = utils::Nest(std::move(queries_vectors), nq, d1);

  utils::STimer btimer;
  utils::STimer qtimer;

  std::ofstream csv(csv_path);

  csv << "index_type,num_partition,max_degree,efc,build_time,index_size,num_queries,efq1,efq2,query_time,comparison,recall" << std::endl;

  auto index = std::make_unique<anns::graph::RandomPartitionGraph<vdim_t, subgraph_t>> (d0, part);
  btimer.Start();
  index->BuildIndex(base_vectors, num_threads, {
    default_params.max_degree, default_params.efc, default_params.dedup
  });
  btimer.Stop();

  for (size_t efq1 = 1; efq1 <= 16; efq1 *= 2) {
    for (size_t efq2 = 10; efq2 <= 300; efq2 += 10) {
      std::vector<std::vector<id_t>> knn;
      double qt = 0;
      size_t comparison;
      for (size_t t = 0; t < cases; t++) {
        qtimer.Reset(); 
        qtimer.Start();
        index->GetTopkNNParallel(nest_queries_vectors, k, num_threads, efq1, efq2, knn);
        qtimer.Stop();
        qt += qtimer.GetTime();
        comparison = index->GetComparisonAndClear();
      }
      double recall = utils::GetRecall(k, d2, groundtruth, knn);
      csv << "cspg" << comma << part << comma << default_params.max_degree << comma
      << default_params.efc << comma << btimer.GetTime() << comma << index->IndexSize() << comma << nq << comma
      << efq1 << comma << efq2 << comma << qt/cases << comma << comparison << comma << recall << std::endl;
    }
  }

  return 0;
}
