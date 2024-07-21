# `.\pytorch\torch\csrc\utils\verbose.cpp`

```py
#include <ATen/native/verbose_wrapper.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/csrc/utils/verbose.h>

namespace torch {

void initVerboseBindings(PyObject* module) {
  auto m = py::handle(module).cast<py::module>();

  // 定义名为 "_verbose" 的子模块，用于处理 MKL 和 MKLDNN 的详细输出
  auto verbose = m.def_submodule("_verbose", "MKL, MKLDNN verbose");
  
  // 将函数 "_mkl_set_verbose" 绑定到子模块 "_verbose" 中的 "mkl_set_verbose" 方法上
  verbose.def("mkl_set_verbose", torch::verbose::_mkl_set_verbose);
  
  // 将函数 "_mkldnn_set_verbose" 绑定到子模块 "_verbose" 中的 "mkldnn_set_verbose" 方法上
  verbose.def("mkldnn_set_verbose", torch::verbose::_mkldnn_set_verbose);
}

} // namespace torch
```