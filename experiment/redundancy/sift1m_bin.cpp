#include <iostream>
#include <fstream>
#include <graph/rpg.hpp>
#include <graph/binhnsw.hpp>

#include <utils/binary_io.hpp>
#include <utils/stimer.hpp>
#include <utils/resize.hpp>
#include <utils/get_recall.hpp>

const std::string comma = ",";

using vdim_t = float;

const size_t k = 10;
const size_t cases = 10;
const size_t num_threads = 24;

const size_t part = 2;

// Files
const std::string base_vectors_path = "/var/lib/docker/anns/dataset/sift1m/base.fvecs";
const std::string queries_vectors_path = "/var/lib/docker/anns/query/sift1m/query.fvecs";
const std::string groundtruth_path = "/var/lib/docker/anns/query/sift1m/gt.ivecs";

const std::string path1 = "output/redundancy/sift1m_cspg_nsw_2.csv";
const std::string path2 = "output/redundancy/sift1m_binhnsw.csv";

struct params
{
  size_t max_degree{32};
  size_t efc{128};
  // float dedup{0.5};
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

  // std::ofstream csv1(path1);
  // csv1 << "index_type,num_partition,max_degree,efc,build_time,index_size,num_queries,efq,query_time,comparison,recall,dedup" << std::endl;

  // for (float dedup = 0.4; dedup <= 0.5; dedup += 0.1) {
  //   auto index = std::make_unique<anns::graph::RandomPartitionGraph<vdim_t, anns::graph::NSW<vdim_t>>> (d0, part);
  //   btimer.Start();
  //   index->BuildIndex(base_vectors, num_threads, {
  //     default_params.max_degree, default_params.efc, dedup});
  //   btimer.Stop();
  //   for (size_t efq = 10; efq <= 300; efq += 10) {
  //     std::vector<std::vector<id_t>> knn;
  //     double qt = 0;
  //     size_t comparison;
  //     for (size_t t = 0; t < cases; t++) {
  //       qtimer.Reset(); 
  //       qtimer.Start();
  //       index->GetTopkNNParallel(nest_queries_vectors, k, num_threads, 1, efq, knn);
  //       qtimer.Stop();
  //       qt += qtimer.GetTime();
  //       comparison = index->GetComparisonAndClear();
  //     }
  //     double recall = utils::GetRecall(k, d2, groundtruth, knn);
  //     csv1 << "cspg" << comma << part << comma << default_params.max_degree << comma
  //     << default_params.efc << comma << btimer.GetTime() << comma << index->IndexSize() << comma << nq << comma
  //     << 1 << "|" << efq << comma << qt/cases << comma << comparison << comma << recall << comma << dedup << std::endl;
  //   }
  // }

  std::ofstream csv2(path2);
  csv2 << "index_type,num_partition,max_degree,efc,build_time,index_size,num_queries,efq,query_time,comparison,recall,dedup" << std::endl;

  for (float dedup = 0.1; dedup <= 0.5; dedup += 0.1) {
    auto index = std::make_unique<anns::graph::BinHNSW<vdim_t>> (
      d0, nb, default_params.max_degree, default_params.efc
    );
    index->SetNumThreads(num_threads);
    btimer.Start();
    index->Populate(base_vectors, dedup);
    btimer.Stop();
    for (size_t efq = 10; efq <= 300; efq += 10) {
      std::vector<std::vector<id_t>> knn;
      std::vector<std::vector<float>> dists;
      double qt = 0;
      size_t comparison;
      for (size_t t = 0; t < cases; t++) {
        qtimer.Reset(); 
        qtimer.Start();
        index->Search(nest_queries_vectors, k, efq, knn, dists);
        qtimer.Stop();
        qt += qtimer.GetTime();
        comparison = index->GetComparisonAndClear();
      }
      double recall = utils::GetRecall(k, d2, groundtruth, knn);
      csv2 << "binhnsw" << comma << part << comma << default_params.max_degree << comma
      << default_params.efc << comma << btimer.GetTime() << comma << index->IndexSize() << comma << nq << comma
      << 1 << "|" << efq << comma << qt/cases << comma << comparison << comma << recall << comma << dedup << std::endl;
    }
  }

  return 0;
}
