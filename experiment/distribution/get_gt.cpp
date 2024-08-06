#include <vector_ops.hpp>
#include <utils/binary_io.hpp>
#include <iostream>
#include <queue>
#include <utility>

// Files
const std::string base_vectors_path = "/home/dbcloud/ym/CSPG/experiment/distribution/data/base.fvecs";
const std::string queries_vectors_path = "/home/dbcloud/ym/CSPG/experiment/distribution/data/query.fvecs";
const std::string groundtruth_path = "/home/dbcloud/ym/CSPG/experiment/distribution/data/gt.ivecs";

using vdim_t = float;

const size_t mx_gt = 100;

int main() {
  std::vector<vdim_t> base_vectors, queries_vectors;
  std::vector<id_t> groundtruth;
  auto [nb, d0] = utils::LoadFromFile(base_vectors, base_vectors_path);
  auto [nq, d1] = utils::LoadFromFile(queries_vectors, queries_vectors_path);
  std::cout << "Base vectors: " << nb << "x" << d0 << std::endl;
  std::cout << "Queries vectors: " << nq << "x" << d1 << std::endl;

  groundtruth.resize(nq * mx_gt);

#pragma omp parallel for
  for (size_t q = 0; q < nq; q++) {
    std::priority_queue<std::pair<float, id_t>> pq;
    const vdim_t * vq = queries_vectors.data() + q * d1;
    for (id_t id = 0; id < nb; id++) {
      // if (id == q) continue;
      pq.emplace(vec_L2sqr(vq, base_vectors.data() + id * d0, d0), id);
      if (pq.size() > mx_gt) pq.pop();
    }
    for (int i = mx_gt - 1; i >= 0; i--) {
      groundtruth[q * mx_gt + i] = pq.top().second;
      // std::cout << pq.top().first << std::endl;
      pq.pop();
    }
  }

  utils::WriteToFile(groundtruth, {nq, mx_gt}, groundtruth_path);

  return 0;
}