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
using subgraph_t = puiann::graph::HCNNG<vdim_t>;
const size_t k = 10;
const size_t cases = 10;
const size_t num_threads = 24;

// Files
const std::string base_vectors_path = "/var/lib/docker/anns/dataset/sift1m/base.fvecs";
const std::string queries_vectors_path = "/var/lib/docker/anns/query/sift1m/query.fvecs";
const std::string groundtruth_path = "/var/lib/docker/anns/query/sift1m/gt.ivecs";

#if defined (BASELINE)
const std::string csv_path_baseline= "output/grid/sift1m_hcnng_cmp_baseline.csv";
#endif

#if defined (PARALLEL)
const size_t part = 2;
const std::string csv_path_parallel = "output/grid/sift1m_hcnng_cmp_cspg_"+std::to_string(part)+".csv";
#endif

// Parameters
struct params
{
  size_t num_random_clusters{15};
  size_t min_size_cluters{1000};
  size_t max_mst_degree{3};
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
  
#if defined (BASELINE)
  std::ofstream csv_baseline(csv_path_baseline);
  csv_baseline << "index_type,num_partition,num_random_clusters,min_size_cluters,max_mst_degree,build_time,index_size,num_queries,efq,query_time,comparison,recall" << std::endl;

  auto hcnng = std::make_unique<puiann::graph::HCNNG<vdim_t>> (d0, nb);
  hcnng->SetNumThreads(num_threads);
  build_timer.Reset();
  build_timer.Start();
  hcnng->CreateHCNNG(base_vectors, default_params.num_random_clusters, default_params.min_size_cluters, default_params.max_mst_degree);
  build_timer.Stop();
  for (size_t efq = 10; efq <= 300; efq += 10) {
    std::vector<std::vector<id_t>> knn;
    std::vector<std::vector<float>> dis;
    double qt = 0;
    size_t comparison;
    for (size_t t = 0; t < cases; t++) {
      query_timer.Reset();
      query_timer.Start();
      hcnng->Search(nest_queries_vectors, k, efq, knn, dis);
      query_timer.Stop();
      qt += query_timer.GetTime();
      comparison = hcnng->GetComparisonAndClear();
    }
    double recall = utils::GetRecall(k, d2, groundtruth, knn);
    csv_baseline << "hcnng" << comma << 1 << comma << default_params.num_random_clusters << comma << default_params.min_size_cluters << comma << default_params.max_mst_degree << comma
      << build_timer.GetTime() << comma << hcnng->IndexSize() << comma << nq << comma
      << efq << comma << qt/cases << comma << comparison << comma <<  recall << std::endl;
  }
#endif
  
#if defined (PARALLEL)
  std::ofstream csv_parallel(csv_path_parallel);
  csv_parallel << "index_type,num_partition,num_random_clusters,min_size_cluters,max_mst_degree,build_time,index_size,num_queries,efq,query_time,comparison,recall" << std::endl;

  build_timer.Reset();
  auto cspg = std::make_unique<puiann::graph::RandomPartitionGraph<vdim_t, subgraph_t>> (d0, part);
  build_timer.Start();
  cspg->BuildIndex(base_vectors, num_threads, 
    {default_params.num_random_clusters, default_params.min_size_cluters, default_params.max_mst_degree, default_params.dedup});
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
    csv_parallel << "cspg" << comma << part << comma << default_params.num_random_clusters << comma << default_params.min_size_cluters << comma << default_params.max_mst_degree << comma
      << build_timer.GetTime() << comma << cspg->IndexSize() << comma << nq << comma
      << 1 << "|" << efq << comma << qt/cases << comma << comparison << comma << recall << std::endl;
  }
#endif

  // std::cout << "All Tasks Finished" << std::endl;
  return 0;
}
