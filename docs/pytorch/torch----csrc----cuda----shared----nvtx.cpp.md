# `.\pytorch\torch\csrc\cuda\shared\nvtx.cpp`

```
#ifdef _WIN32
#include <wchar.h> // _wgetenv for nvtx
#endif
#include <nvToolsExt.h>  // 包含 NVIDIA NVTX 工具的头文件
#include <torch/csrc/utils/pybind.h>  // 包含 PyTorch 的 Python 绑定工具头文件

namespace torch::cuda::shared {

void initNvtxBindings(PyObject* module) {
  auto m = py::handle(module).cast<py::module>();  // 将传入的 Python 模块对象转换为 pybind11 的模块对象

  auto nvtx = m.def_submodule("_nvtx", "libNvToolsExt.so bindings");  // 定义名为 "_nvtx" 的子模块，用于绑定 libNvToolsExt.so 的函数
  nvtx.def("rangePushA", nvtxRangePushA);  // 绑定 NVTX 的 rangePushA 函数
  nvtx.def("rangePop", nvtxRangePop);  // 绑定 NVTX 的 rangePop 函数
  nvtx.def("rangeStartA", nvtxRangeStartA);  // 绑定 NVTX 的 rangeStartA 函数
  nvtx.def("rangeEnd", nvtxRangeEnd);  // 绑定 NVTX 的 rangeEnd 函数
  nvtx.def("markA", nvtxMarkA);  // 绑定 NVTX 的 markA 函数
}

} // namespace torch::cuda::shared
```