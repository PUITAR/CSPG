#include <utils/binary_io.hpp>
#include <cmath>
#include <omp.h>
#include <queue>
#include <vector_ops.hpp>

#include <string>

#include <utils/tqdm.hpp>

#include <iostream>

const std::string base10m_path = "/var/lib/docker/anns/dataset/sift10m/base.bvecs";
const std::string queries_path = "/var/lib/docker/anns/query/sift10m/query.bvecs";
const std::string output_path = "/var/lib/docker/anns/dataset/sift10m/";
const std::string gt_path = "/var/lib/docker/anns/query/sift10m/";

int main() {
  std::vector<uint8_t> base10m;
  std::vector<uint8_t> queries;
  auto [nb, d] = utils::LoadFromFile(base10m, base10m_path);
  auto [nq, dq] = utils::LoadFromFile(queries, queries_path);

  // std::cout << nb << " " << d << " " << nq << " " << dq << std::endl;

  float fn[5] = {0.1, 0.2, 0.5, 2, 5};

  std::string fn_str[5] = {"0.1", "0.2", "0.5", "2", "5"};

  size_t maxk = 100;

  utils::TQDM bar(5);

  for (size_t ii = 0; ii < 5; ii++) {
    float f = fn[ii];
    auto s = fn_str[ii];
    size_t nsample = static_cast<size_t> (std::floor(f*1e6));
    // std::cout << "nsample: " << nsample << std::endl;
    std::vector<uint8_t> sample_data(base10m.begin(), base10m.begin() + nsample * d);
    utils::WriteToFile(sample_data, {nsample, d}, 
      output_path + "base" + s + "m.bvecs");
    std::vector<id_t> groundtruth(nq*maxk);
    #pragma omp parallel for 
    for (size_t i = 0; i < nq; i++) {
      const uint8_t * qv = queries.data() + i * dq;
      std::priority_queue<std::pair<float, id_t>> pq;
      for (id_t j = 0; j < nsample; j++) {
        pq.emplace(
          vec_L2sqr(qv, sample_data.data() + j * d, d),
          j
        );
        if (pq.size() > maxk)
          pq.pop();
      }
      std::priority_queue<std::pair<float, id_t>> topk;
      for (size_t k = 0; k < maxk; k++) {
        const auto & t = pq.top();
        topk.emplace(-t.first, t.second);
        pq.pop();
      }
      for (size_t k = 0; k < maxk; k++) {
        groundtruth[i*maxk+k] = topk.top().second;
        topk.pop();
      }
    }
    utils::WriteToFile(groundtruth, {nq, maxk}, 
      gt_path + "gt" + s + "m.ivecs");
    bar.Next();
  }

  return 0;
}