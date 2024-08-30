# ANNS Library Usage

```c++
// ... ... ...
int main() {
  // std::cout << "Read Files" << std::endl;
  std::vector<vdim_t> base_vectors, queries_vectors;
  std::vector<id_t> groundtruth;
  auto [nb, d0] = utils::LoadFromFile(base_vectors, base_vectors_path);
  auto [nq, d1] = utils::LoadFromFile(queries_vectors, queries_vectors_path);
  auto [ng, d2] = utils::LoadFromFile(groundtruth, groundtruth_path);
  auto nest_queries_vectors = utils::Nest(std::move(queries_vectors), nq, d1);
  utils::STimer build_timer;
  utils::STimer query_timer;
  // ... ... ...
  std::ofstream csv_parallel(csv_path_parallel);
  csv_parallel << "index_type,num_partition,max_degree,efc,build_time,index_size,num_queries,efq,query_time,comparison,recall" << std::endl;

  build_timer.Reset();
  auto cspg = std::make_unique<anns::graph::RandomPartitionGraph<vdim_t, subgraph_t>> (d0, part);
  build_timer.Start();
  cspg->BuildIndex(base_vectors, num_threads, {default_params.max_degree, default_params.efc, default_params.dedup});
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
    csv_parallel << "cspg" << comma << part << comma << default_params.max_degree << comma
      << default_params.efc << comma << build_timer.GetTime() << comma << cspg->IndexSize() << comma << nq << comma
      << 1 << "|" << efq << comma << qt/cases << comma << comparison << comma << recall << std::endl;
  }
  // ... ... ...
  return 0;
}
```

This is an example of how to use the ANNS library to build and query an index. The `main` function takes in the path to the base vectors, queries vectors, and groundtruth file, as well as the number of partitions, maximum degree, and effective query count for the index. It then builds the index and performs queries on the queries vectors, and outputs the results to a CSV file.

To use the ANNS library simply, we wrap the CPP APIs with Python API, which you can use as follow. [Example for ANNS Python API](test_anns.ipynb)

```python
# Include the library modules
import sys
mpath = 'Path/To/CSPG/ANNS/modules'
if mpath not in sys.path:
  sys.path.append(mpath)

# Import important modules
import anns
import numpy as np
from binary_io import  *

# Data path
base = "/var/lib/docker/anns/dataset/sift1m/base.fvecs"
query = "/var/lib/docker/anns/query/sift1m/query.fvecs"
gt = "/var/lib/docker/anns/query/sift1m/gt.ivecs"

# Read data from files
base = fvecs_read(base)
query = fvecs_read(query)
gt = ivecs_read(gt)
nb, d = base.shape
nq, _ = query.shape
_, ngt = gt.shape

# Parameters
m = 2
threads = 24
k = 10

# Reshape the data to adjust to the library APIs
base = base.flatten().tolist()
query = query.flatten().tolist()
gt = gt.flatten().tolist()

# Initialize your index and build
index = anns.CSPG_HNSW_FLOAT(d, m)
index.build(base, threads, [32, 128, 0.5])

# Search k-NN
knn = index.search(query, k, 1, ef1 = 1, ef2 = 128)

# Get recall by groudtruth
print(anns.get_recall(k, ngt, gt, knn))
```
