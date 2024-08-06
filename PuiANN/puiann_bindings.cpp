#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <graph/rpg.hpp>

#include <utils/get_recall.hpp>
#include <utils/stimer.hpp>

namespace py = pybind11;

using namespace puiann::graph;


PYBIND11_MODULE(puiann, m) {
  m.doc() = "CSPG's Python APIs";

  using rpg_diskann_float_t = RandomPartitionGraph<float, DiskANN<float>>;
  using rpg_hnsw_float_t = RandomPartitionGraph<float, HNSW<float>>;
  using rpg_hcnng_float_t = RandomPartitionGraph<float, HCNNG<float>>;
  using rpg_nsg_float_t = RandomPartitionGraph<float, NSG<float>>;

  using rpg_diskann_uint8_t = RandomPartitionGraph<uint8_t, DiskANN<uint8_t>>;
  using rpg_hnsw_uint8_t = RandomPartitionGraph<uint8_t, HNSW<uint8_t>>;
  using rpg_hcnng_uint8_t = RandomPartitionGraph<uint8_t, HCNNG<uint8_t>>;
  using rpg_nsg_uint8_t = RandomPartitionGraph<uint8_t, NSG<uint8_t>>;

  py::class_<rpg_hcnng_float_t> (m, "CSPG_HCNNG_FLOAT")
    .def(py::init<size_t, size_t>(), "CSPG constructor",
         py::arg("dim"), py::arg("m"))
    .def("build", &rpg_hcnng_float_t::BuildIndex, "Build index",
         py::arg("base vectors"), py::arg("threads"), py::arg("arg_list"))
    .def("search", &rpg_hcnng_float_t::GetTopkNNParallel2, "Search k approximate nearest neighbors",
         py::arg("query vectors"), py::arg("k"), py::arg("threads"), py::arg("ef1"), py::arg("ef2"))
    .def("size", &rpg_hcnng_float_t::IndexSize, "Get index size")
    .def("comparison", &rpg_hcnng_float_t::GetComparisonAndClear, "Get comparison and clear the counter");
  
  py::class_<rpg_hnsw_float_t> (m, "CSPG_HNSW_FLOAT")
    .def(py::init<size_t, size_t>(), "CSPG constructor",
         py::arg("dim"), py::arg("m"))
    .def("build", &rpg_hnsw_float_t::BuildIndex, "Build index",
         py::arg("base vectors"), py::arg("threads"), py::arg("arg_list"))
    .def("search", &rpg_hnsw_float_t::GetTopkNNParallel2, "Search k approximate nearest neighbors",
         py::arg("query vectors"), py::arg("k"), py::arg("threads"), py::arg("ef1"), py::arg("ef2"))
    .def("size", &rpg_hnsw_float_t::IndexSize, "Get index size")
    .def("comparison", &rpg_hnsw_float_t::GetComparisonAndClear, "Get comparison and clear the counter");
  
  py::class_<rpg_diskann_float_t> (m, "CSPG_DiskANN_FLOAT")
    .def(py::init<size_t, size_t>())
    .def("build", &rpg_diskann_float_t::BuildIndex, "Build index",
         py::arg("base vectors"), py::arg("threads"), py::arg("arg_list[R,alpha,L,lambda]"))
    .def("search", &rpg_diskann_float_t::GetTopkNNParallel2, "Search k approximate nearest neighbors",
         py::arg("query vectors"), py::arg("k"), py::arg("threads"), py::arg("ef1"), py::arg("ef2"))
    .def("size", &rpg_diskann_float_t::IndexSize, "Get index size")
    .def("comparison", &rpg_diskann_float_t::GetComparisonAndClear, "Get comparison and clear the counter");
  
  py::class_<rpg_nsg_float_t> (m, "CSPG_NSG_FLOAT")
    .def(py::init<size_t, size_t>())
    .def("build", &rpg_nsg_float_t::BuildIndex, "Build index",
         py::arg("base vectors"), py::arg("threads"), py::arg("arg_list"))
    .def("search", &rpg_nsg_float_t::GetTopkNNParallel2, "Search k approximate nearest neighbors",
         py::arg("query vectors"), py::arg("k"), py::arg("threads"), py::arg("ef1"), py::arg("ef2"))
    .def("size", &rpg_nsg_float_t::IndexSize, "Get index size")
    .def("comparison", &rpg_nsg_float_t::GetComparisonAndClear, "Get comparison and clear the counter");

  py::class_<rpg_hcnng_uint8_t> (m, "CSPG_HCNNG_UINT8")
    .def(py::init<size_t, size_t>(), "CSPG constructor",
         py::arg("dim"), py::arg("m"))
    .def("build", &rpg_hcnng_uint8_t::BuildIndex, "Build index",
         py::arg("base vectors"), py::arg("threads"), py::arg("arg_list"))
    .def("search", &rpg_hcnng_uint8_t::GetTopkNNParallel2, "Search k approximate nearest neighbors",
         py::arg("query vectors"), py::arg("k"), py::arg("threads"), py::arg("ef1"), py::arg("ef2"))
    .def("size", &rpg_hcnng_uint8_t::IndexSize, "Get index size")
    .def("comparison", &rpg_hcnng_uint8_t::GetComparisonAndClear, "Get comparison and clear the counter");

  py::class_<rpg_hnsw_uint8_t> (m, "CSPG_HNSW_UINT8")
    .def(py::init<size_t, size_t>(), "CSPG constructor", 
         py::arg("dim"), py::arg("m"))
    .def("build", &rpg_hnsw_uint8_t::BuildIndex, "Build index",
         py::arg("base vectors"), py::arg("threads"), py::arg("arg_list"))
    .def("search", &rpg_hnsw_uint8_t::GetTopkNNParallel2, "Search k approximate nearest neighbors",
         py::arg("query vectors"), py::arg("k"), py::arg("threads"), py::arg("ef1"), py::arg("ef2"))
    .def("size", &rpg_hnsw_uint8_t::IndexSize, "Get index size")
    .def("comparison", &rpg_hnsw_uint8_t::GetComparisonAndClear, "Get comparison and clear the counter");

  py::class_<rpg_diskann_uint8_t> (m, "CSPG_DiskANN_UINT8")
    .def(py::init<size_t, size_t>(), "CSPG constructor",
         py::arg("dim"), py::arg("m"))
    .def("build", &rpg_diskann_uint8_t::BuildIndex, "Build index",
         py::arg("base vectors"), py::arg("threads"), py::arg("arg_list[R,alpha,L,lambda]"))
    .def("search", &rpg_diskann_uint8_t::GetTopkNNParallel2, "Search k approximate nearest neighbors",
         py::arg("query vectors"), py::arg("k"), py::arg("threads"), py::arg("ef1"), py::arg("ef2"))
    .def("size", &rpg_diskann_uint8_t::IndexSize, "Get index size")
    .def("comparison", &rpg_diskann_uint8_t::GetComparisonAndClear, "Get comparison and clear the counter");

  py::class_<rpg_nsg_uint8_t> (m, "CSPG_NSG_UINT8")
    .def(py::init<size_t, size_t>(), "CSPG constructor",
         py::arg("dim"), py::arg("m"))
    .def("build", &rpg_nsg_uint8_t::BuildIndex, "Build index",
         py::arg("base vectors"), py::arg("threads"), py::arg("arg_list"))
    .def("search", &rpg_nsg_uint8_t::GetTopkNNParallel2, "Search k approximate nearest neighbors",
         py::arg("query vectors"), py::arg("k"), py::arg("threads"), py::arg("ef1"), py::arg("ef2"))
    .def("size", &rpg_nsg_uint8_t::IndexSize, "Get index size")
    .def("comparison", &rpg_nsg_uint8_t::GetComparisonAndClear, "Get comparison and clear the counter");
    
  m.def("get_recall", &utils::GetRecall, "Get the recall of the kNN results", 
         py::arg("k"), py::arg("ngt"), py::arg("groundtruth"), py::arg("knn_results"));

  py::class_<utils::STimer> (m, "STimer")
    .def(py::init<>())
    .def("start", &utils::STimer::Start)
    .def("stop", &utils::STimer::Stop)
    .def("reset", &utils::STimer::Reset)
    .def("get_time", &utils::STimer::GetTime);

}
