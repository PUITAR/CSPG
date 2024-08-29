
import sys

mpath = '/home/dbcloud/ym/CSPG/ANNS/modules'

if mpath not in sys.path:
  sys.path.append(mpath)

import anns
import numpy as np


from binary_io import  *

base = "/var/lib/docker/anns/dataset/sift1m/base.fvecs"
query = "/var/lib/docker/anns/query/sift1m/query.fvecs"
gt = "/var/lib/docker/anns/query/sift1m/gt.ivecs"

base = fvecs_read(base)
query = fvecs_read(query)
gt = ivecs_read(gt)

base, query, gt

nb, d = base.shape
nq, _ = query.shape
_, ngt = gt.shape

nb, d, nq, ngt

m = 2
threads = 24
k = 10

base = base.flatten().tolist()
query = query.flatten().tolist()
gt = gt.flatten().tolist()


index = anns.CSPG_HNSW_FLOAT(d, m)

index.build(base, threads, [32, 128, 0.5])

knn = index.search(query, k, 1, ef1 = 1, ef2 = 128)

print(anns.get_recall(k, ngt, gt, knn))