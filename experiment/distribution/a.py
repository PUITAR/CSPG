# %%
import faiss
import sys

modules = '/home/dbcloud/ym/CSPG/PuiANN/modules'

if modules not in sys.path:
  sys.path.append(modules)

from binary_io import *
import puiann as pui

# %%
base = 'data/base.fvecs'
query = 'data/query.fvecs'
gt = 'data/gt.ivecs'

base = fvecs_read(base)[:10]
query = fvecs_read(query)
gt = ivecs_read(gt)

# %%
nb, d = base.shape
nq, ngt = gt.shape

# base = base.flatten().tolist()
gt = gt.flatten().tolist()
# query = [q.tolist() for q in query]

nb, d, nq, ngt

# %%
# nlist = 1204

# quantizer = faiss.IndexFlatL2(d) 
# index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
# # Train the index
# index.train(base)
# # Add the base vectors to the index
# index.add(base)

# %%
# Perform a search with query vectors
k = 10
threads = 1


# %%
# faiss.omp_set_num_threads(threads)

# out = '/home/dbcloud/ym/CSPG/experiment/output/distribution/rcd_ivfpq.csv'
# out = open(out, 'w')

# tm = pui.STimer()

# print('num_queries,nlist,nprobe,query_time,recall', file=out)
# for npb in range(10, 200+1, 10):
#   index.nprobe = npb
#   tm.reset()
#   tm.start()
#   D, I = index.search(query, k)
#   tm.stop()
#   query_time = tm.get_time()
#   recall = pui.get_recall(k, ngt, gt, I)
#   print(f"{nq},{nlist},{npb},{query_time},{recall}", file=out)

# out.close()

# %%
base = base.flatten().tolist()
query = [q.tolist() for q in query]

# %%
m = 2

index = pui.CSPG_DiskANN_FLOAT(d, m)

index.build(base, threads, [32, 1.2, 128, 0.5])


