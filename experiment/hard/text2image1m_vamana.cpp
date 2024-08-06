#include <iostream>
#include <fstream>
#include <graph/rpg.hpp>
#include <utils/binary_io.hpp>
#include <utils/stimer.hpp>
#include <utils/resize.hpp>
#include <utils/get_recall.hpp>

#define BASELINE
#define PARALLEL

const std::string comma = ",";

// Task Configures
using vdim_t = float;
using subgraph_t = puiann::graph::DiskANN<vdim_t>;
const size_t k = 10;
const size_t cases = 10;
const size_t num_threads = 24;

// Files
const std::string base_vectors_path = "/home/dbcloud/big-ann-benchmarks/data/text2image1B/base.1B.fbin.crop_nb_1000000";
const std::string queries_vectors_path = "/home/dbcloud/big-ann-benchmarks/data/text2image1B/query.heldout.30K.fbin";
const std::string groundtruth_path = "/home/dbcloud/big-ann-benchmarks/data/text2image1B/gt100-heldout.30K.ivecs";

#if defined (BASELINE)
const std::string csv_path_baseline = "output/hard/text2image1m_vamana_baseline.csv";
#endif
#if defined (PARALLEL)
const size_t part = 2;
const std::string csv_path_parallel = "output/hard/text2image1m_vamana_cspg_"+std::to_string(part)+".csv";
#endif

// Parameters
struct params
{
  size_t max_degree{32};
  size_t efc{128};
  float alpha{1.2};
  float dedup{0.5};
} default_params;

int main() {
  // std::cout << "Read Files" << std::endl;
  std::vector<vdim_t> base_vectors, queries_vectors;
  std::vector<id_t> groundtruth;
  auto [nb, d0] = utils::LoadFromFileBin(base_vectors, base_vectors_path);
  auto [nq, d1] = utils::LoadFromFileBin(queries_vectors, queries_vectors_path);
  auto [ng, d2] = utils::LoadFromFile(groundtruth, groundtruth_path);
  auto nest_queries_vectors = utils::Nest(std::move(queries_vectors), nq, d1);
  // std::cout << "Base vectors: " << nb << "x" << d0 << std::endl;
  // std::cout << "Queries vectors: " << nq << "x" << d1 << std::endl;
  // std::cout << "Groundtruth: " << ng << "x" << d2 << std::endl;
  
  utils::STimer build_timer;
  utils::STimer query_timer;

#if defined (BASELINE)
  std::ofstream csv_baseline(csv_path_baseline);
  csv_baseline << "index_type,num_partition,max_degree,alpha,efc,build_time,index_size,num_queries,efq,query_time,comparison,recall" << std::endl;
  auto diskann = std::make_unique<puiann::graph::DiskANN<vdim_t>> (d0, nb, default_params.max_degree);
  diskann->SetNumThreads(num_threads);
  build_timer.Reset();
  build_timer.Start();
  diskann->BuildIndex(base_vectors, default_params.alpha, default_params.efc);
  build_timer.Stop();
  for (size_t efq = 10; efq <= 300; efq += 10) {
    std::vector<std::vector<id_t>> knn;
    std::vector<std::vector<float>> dis;
    double qt = 0;
    size_t comparison;
    for (size_t t = 0; t < cases; t++) {
      query_timer.Reset();
      query_timer.Start();
      diskann->Search(nest_queries_vectors, k, efq, knn, dis);
      query_timer.Stop();
      qt += query_timer.GetTime();
      comparison = diskann->GetComparisonAndClear();
    }
    double recall = utils::GetRecall(k, d2, groundtruth, knn);
    csv_baseline << "vamana" << comma << 1 << comma << default_params.max_degree << comma << default_params.alpha << comma
      << default_params.efc << comma << build_timer.GetTime() << comma << diskann->IndexSize() << comma << nq << comma
      << efq << comma << qt/cases << comma << comparison << comma << recall << std::endl;
  }
#endif
  
#if defined (PARALLEL)
  std::ofstream csv_parallel(csv_path_parallel);
  csv_parallel << "index_type,num_partition,max_degree,alpha,efc,build_time,index_size,num_queries,efq,query_time,comparison,recall" << std::endl;
  build_timer.Reset();
  auto cspg = std::make_unique<puiann::graph::RandomPartitionGraph<vdim_t, subgraph_t>> (d0, part);
  build_timer.Start();
  cspg->BuildIndex(base_vectors, num_threads, {default_params.max_degree, default_params.alpha, default_params.efc, default_params.dedup});
  build_timer.Stop();

  for (size_t efq = 10; efq <= 300; efq += 10) {
    std::vector<std::vector<id_t>> knn;
    double qt = 0;
    size_t comparison;
    for (size_t t = 0; t < cases; t++) {
      query_timer.Reset();
      query_timer.Start();
      cspg->GetTopkNNParallel(nest_queries_vectors, k, num_threads, 1, efq, knn);
      query_timer.Stop();
      qt += query_timer.GetTime();
      comparison = cspg->GetComparisonAndClear();
    }
    double recall = utils::GetRecall(k, d2, groundtruth, knn);
    csv_parallel << "cspg" << comma << part << comma << default_params.max_degree << comma << default_params.alpha << comma
      << default_params.efc << comma << build_timer.GetTime() << comma << cspg->IndexSize() << comma << nq << comma
      << 1 << "|" << efq << comma << qt/cases << comma << comparison << comma << recall << std::endl;
  }
#endif

  // std::cout << "All Tasks Finished" << std::endl;
  return 0;
}
