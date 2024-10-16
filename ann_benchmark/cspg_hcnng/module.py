import anns
import numpy as np

from ..base.module import BaseANN


class CSPG_HCNNG(BaseANN):

  def __init__(self, params):
    self.m = params['m']
    self.index_args = params['index_args']
    self.name = 'cspg_hcnng'
    # print(f'm: {self.m}')
    # print(f'index_args: {self.index_args}')

  def set_query_arguments(self, ef2):
    self.ef1 = 1
    self.ef2 = ef2
    # print(f'ef1: {self.ef1}')
    # print(f'ef2: {self.ef2}')

  def fit(self, X):
    X = X.astype(np.float32)
    # print(f'X.shape: {X.shape}')
    _, self.d = X.shape
    self.index = anns.CSPG_HCNNG_FLOAT(self.d, self.m)
    X = X.flatten().tolist()
    self.index.build(X, 1, self.index_args)

  def query(self, q, n):
    q = q.tolist()
    knn = self.index.search(q, n, 1, self.ef1, self.ef2)
    return np.array(knn[0])
  
  def get_additional(self):
    return {"dist_comps": self.index.comparison()}
