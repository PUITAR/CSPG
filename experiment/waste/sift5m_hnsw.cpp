#include <iostream>
#include <fstream>
#include <graph/rpg.hpp>
#include <utils/binary_io.hpp>
#include <utils/stimer.hpp>
#include <utils/resize.hpp>
#include <utils/get_recall.hpp>

#include "get_waste.hpp"

const std::string comma = ",";

// Task Configures
using vdim_t = uint8_t;
using subgraph_t = anns::graph::HNSW<vdim_t>;
const size_t k = 10;
const size_t cases = 10;
const size_t num_threads = 24;

const std::string base_vectors_path = "/var/lib/docker/anns/dataset/sift10m/base5m.bvecs";
const std::string queries_vectors_path = "/var/lib/docker/anns/query/sift10m/query.bvecs";
const std::string groundtruth_path = "/var/lib/docker/anns/query/sift10m/gt5m.ivecs";

const std::string csv_path_baseline= "output/waste/sift5m_hnsw_baseline.csv";

// Parameters
struct params
{
  size_t max_degree{32};
  size_t efc{128};
  float dedup{0.5};
} default_params;

int main() {
  // std::cout << "Read Files" << std::endl;
  std::vector<vdim_t> base_vectors, queries_vectors;
  std::vector<id_t> groundtruth;
  auto [nb, d0] = utils::LoadFromFile(base_vectors, base_vectors_path);
  auto [nq, d1] = utils::LoadFromFile(queries_vectors, queries_vectors_path);
  auto [ng, d2] = utils::LoadFromFile(groundtruth, groundtruth_path);
  auto nest_queries_vectors = utils::Nest(std::move(queries_vectors), nq, d1);
  // std::cout << "Base vectors: " << nb << "x" << d0 << std::endl;
  // std::cout << "Queries vectors: " << nq << "x" << d1 << std::endl;
  // std::cout << "Groundtruth: " << ng << "x" << d2 << std::endl;
  
  utils::STimer build_timer;
  utils::STimer query_timer;
  
  std::ofstream csv_baseline(csv_path_baseline);
  csv_baseline << "index_type,num_partition,max_degree,efc,build_time,index_size,num_queries,efq,query_time,comparison,recall,w" << std::endl;

  auto hnsw = std::make_unique<anns::graph::HNSW<vdim_t>> (
    d0, nb, default_params.max_degree, default_params.efc);
  hnsw->SetNumThreads(num_threads);
  build_timer.Reset();
  build_timer.Start();
  hnsw->Populate(base_vectors);
  build_timer.Stop();
  for (size_t efq = 10; efq <= 300; efq += 10) {
    std::vector<std::vector<id_t>> knn;
    std::vector<std::vector<float>> dis;
    double qt = 0;
    size_t comparison;
    for (size_t t = 0; t < cases; t++) {
      query_timer.Reset();
      query_timer.Start();
      dis = hnsw->GetSearchLength(nest_queries_vectors, k, efq, knn, dis);
      query_timer.Stop();
      qt += query_timer.GetTime();
      comparison = hnsw->GetComparisonAndClear();
    }
    double recall = utils::GetRecall(k, d2, groundtruth, knn);
    double w = GetWasteFactor(dis);
    csv_baseline << "hnsw" << comma << 1 << comma << default_params.max_degree << comma << default_params.efc << comma
      << build_timer.GetTime() << comma << hnsw->IndexSize() << comma << nq << comma
      << efq << comma << qt/cases << comma << comparison << comma << recall << comma << w << std::endl;
  }
  
  // std::cout << "All Tasks Finished" << std::endl;
  return 0;
}
