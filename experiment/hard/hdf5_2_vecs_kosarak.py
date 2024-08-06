import h5py
import numpy as np
from typing import Callable, List, NamedTuple, Tuple, Union
import struct

dimension = 27983

def save_vecs(filename, data):
  with open(filename, 'wb') as fp:
    for y in data:
      d = struct.pack('I', y.size)
      fp.write(d)
      for x in y:
        a = struct.pack('f', x)
        fp.write(a)


def convert_sparse_to_list(data: np.ndarray, lengths: List[int]) -> List[np.ndarray]:
  """
  Converts sparse data into a list of arrays, where each array represents a separate data sample.

  Args:
      data (np.ndarray): The input sparse data represented as a numpy array.
      lengths (List[int]): List of lengths for each data sample in the sparse data.

  Returns:
      List[np.ndarray]: A list of arrays where each array is a data sample.
  """
  return [
      data[i - l : i] for i, l in zip(np.cumsum(lengths), lengths)
  ]

def dataset_transform(dataset: h5py.Dataset):
  """
  Transforms the dataset from the HDF5 format to conventional numpy format.

  If the dataset is dense, it's returned as a numpy array.
  If it's sparse, it's transformed into a list of numpy arrays, each representing a data sample.

  Args:
      dataset (h5py.Dataset): The input dataset in HDF5 format.

  Returns:
      Tuple[Union[np.ndarray, List[np.ndarray]], Union[np.ndarray, List[np.ndarray]]]: Tuple of training and testing data in conventional format.
  """
  if dataset.attrs.get("type", "dense") != "sparse":
      return np.array(dataset["train"]), np.array(dataset["test"])

  # we store the dataset as a list of integers, accompanied by a list of lengths in hdf5
  # so we transform it back to the format expected by the algorithms here (array of array of ints)
  return (
      convert_sparse_to_list(dataset["train"], dataset["size_train"]),
      convert_sparse_to_list(dataset["test"], dataset["size_test"])
  )

def extract_and_save_datasets(hdf5_file, train_output, test_output):
  # Open the HDF5 file
  with h5py.File(hdf5_file, 'r') as f:
    # Assuming datasets are named 'train' and 'test'
    train_data, test_data = dataset_transform(f)
    # print(len(train_data[1]), len(test_data[1]))
    mns = 1000000000000
    
    for vec in train_data:
      vec.resize(dimension, refcheck=False)
    for vec in test_data:
      vec.resize(dimension, refcheck=False)
    
    # Save the data to fvecs files
    save_vecs(train_output, np.vstack(train_data))
    save_vecs(test_output, np.vstack(test_data))

# Example usage:
hdf5_file = '/home/dbcloud/big-ann-benchmarks/data/kosarak/kosarak-jaccard.hdf5'
train_output = '/home/dbcloud/big-ann-benchmarks/data/kosarak/train_data.fvecs' 
test_output = '/home/dbcloud/big-ann-benchmarks/data/kosarak/test_data.fvecs' 

extract_and_save_datasets(hdf5_file, train_output, test_output)
