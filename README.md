# CSPG: Crossing Sparse Proximity Graphs for Approximate Nearest Neighbor Search

This is the official implementation of the paper [CSPG: Crossing Sparse Proximity Graphs for Approximate Nearest Neighbor Search](https://example.com/paper).

## Requirements

* C++17
* Python
* OpenMP

## Build

```shell
cd experiment
mkdir -p build
cd build
cmake ..
make -j
```

## Dataset
All dataset we used for the evaluation can be download at [ANN-Benchmark](https://github.com/erikbern/ann-benchmarks) or other public repo. You should replace the base data path, query data path and groundtruth data path with your data path by a global replacement (e.g. ctrl+H in vscode). All dataset should be in the format of `fvecs`, `ivecs`, `fbin` and `ibin`. You can use the functions in `ANNS/utils/binary_io.hpp` or use functions in `ANNS/modlues/binary_io.py`.

## Usage

## Run the Experiment

```shell
python experiment.py argument
```

where the argument is one of the following options:

* `overall`: overall performance
* `partition`: partitioning analysis
* `redundancy`: redundancy analysis
* `scale`: scale analysis
* `stages`: two-stages parameters' impact analysis
* `waste`: detour factor analysis
* `grid`: hyperparameters sweep
* `distribution`: comparison over different distribution
* `hard`: evaluations on harder datasets

Since our framework fully runs on the memory, the dataset may be large, please run the code with enough memory.
