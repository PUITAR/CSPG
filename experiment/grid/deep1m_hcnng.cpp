#include <iostream>
#include <fstream>
#include <graph/rpg.hpp>
#include <utils/binary_io.hpp>
#include <utils/stimer.hpp>
#include <utils/resize.hpp>
#include <utils/get_recall.hpp>

const std::string comma = ",";

// Task Configures
using vdim_t = float;

const size_t k = 10;
const size_t cases = 10;
const size_t num_threads = 24;

// Files
const std::string base_vectors_path = "/var/lib/docker/anns/dataset/deep1b/base.1M.fbin";
const std::string queries_vectors_path = "/var/lib/docker/anns/dataset/deep1b/query.public.10K.fbin";
const std::string groundtruth_path = "/var/lib/docker/anns/dataset/deep1b/gt_1M.ibin";

const std::string csv_path_baseline= "output/grid/deep1m_hcnng.csv";

int main() {
  // std::cout << "Read Files" << std::endl;
  std::vector<vdim_t> base_vectors, queries_vectors;
  std::vector<id_t> groundtruth;
  auto [nb, d0] = utils::LoadFromFileBin(base_vectors, base_vectors_path);
  auto [nq, d1] = utils::LoadFromFileBin(queries_vectors, queries_vectors_path);
  auto [ng, d2] = utils::LoadFromFileBin(groundtruth, groundtruth_path);
  auto nest_queries_vectors = utils::Nest(std::move(queries_vectors), nq, d1);
  // std::cout << "Base vectors: " << nb << "x" << d0 << std::endl;
  // std::cout << "Queries vectors: " << nq << "x" << d1 << std::endl;
  // std::cout << "Groundtruth: " << ng << "x" << d2 << std::endl;
  
  utils::STimer build_timer;
  utils::STimer query_timer;
  
  std::ofstream csv_baseline(csv_path_baseline);
  csv_baseline << "index_type,num_partition,num_random_clusters,min_size_cluters,max_mst_degree,build_time,index_size,num_queries,efq,query_time,comparison,recall" << std::endl;

  auto test = [&] (size_t num_random_clusters, size_t min_size_cluters, size_t max_mst_degree) {
    auto hcnng = std::make_unique<anns::graph::HCNNG<vdim_t>> (d0, nb);
    hcnng->SetNumThreads(num_threads);
    build_timer.Reset();
    build_timer.Start();
    hcnng->CreateHCNNG(base_vectors, num_random_clusters, min_size_cluters, max_mst_degree);
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
      csv_baseline << "hcnng" << comma << 1 << comma << num_random_clusters << comma << min_size_cluters << comma << max_mst_degree << comma
        << build_timer.GetTime() << comma << hcnng->IndexSize() << comma << nq << comma
        << efq << comma << qt/cases << comma << comparison << comma <<  recall << std::endl;
    }
  };

  std::vector<size_t> num_random_clusters_list = {5, 10, 15};
  std::vector<size_t> min_size_cluters_list = {750, 1000, 1250};
  std::vector<size_t> max_mst_degree_list = {3, 5, 7};

  for (auto num_random_clusters : num_random_clusters_list) {
    for (auto min_size_cluters : min_size_cluters_list) {
      for (auto max_mst_degree : max_mst_degree_list) {
        test(num_random_clusters, min_size_cluters, max_mst_degree);
      }
    }
  }
  // std::cout << "All Tasks Finished" << std::endl;
  return 0;
}
