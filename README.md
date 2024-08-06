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
