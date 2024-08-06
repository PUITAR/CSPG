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
const std::string base_vectors_path = "/var/lib/docker/anns/dataset/gist1m/base.fvecs";
const std::string queries_vectors_path = "/var/lib/docker/anns/query/gist1m/query.fvecs";
const std::string groundtruth_path = "/var/lib/docker/anns/query/gist1m/gt.ivecs";

const std::string csv_path_baseline= "output/grid/gist1m_hnsw.csv";


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
  csv_baseline << "index_type,num_partition,max_degree,efc,build_time,index_size,num_queries,efq,query_time,comparison,recall" << std::endl;

  std::vector<size_t> M_list = {16, 32, 64};
  std::vector<size_t> efc_list = {64, 128, 256};

  auto test = [&] (size_t M, size_t efc) {
    auto hnsw = std::make_unique<puiann::graph::HNSW<vdim_t>> (
      d0, nb, M, efc);
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
        hnsw->Search(nest_queries_vectors, k, efq, knn, dis);
        query_timer.Stop();
        qt += query_timer.GetTime();
        comparison = hnsw->GetComparisonAndClear();
      }
      double recall = utils::GetRecall(k, d2, groundtruth, knn);
      csv_baseline << "hnsw" << comma << 1 << comma << M << comma << efc << comma
        << build_timer.GetTime() << comma << hnsw->IndexSize() << comma << nq << comma
        << efq << comma << qt/cases << comma << comparison << comma << recall << std::endl;
    }
  };

  for (auto M : M_list) {
    for (auto efc : efc_list) {
      test(M, efc);
    }
  }

  // std::cout << "All Tasks Finished" << std::endl;
  return 0;
}
