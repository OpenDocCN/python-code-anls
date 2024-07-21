# `.\pytorch\torch\csrc\itt.cpp`

```
#include <torch/csrc/itt_wrapper.h>
#include <torch/csrc/utils/pybind.h>

namespace torch::profiler {
void initIttBindings(PyObject* module) {
  // 将 PyObject* 转换为 py::module 类型，用于绑定函数和子模块
  auto m = py::handle(module).cast<py::module>();

  // 在主模块 m 下定义名为 "_itt" 的子模块，用于封装 VTune ITT 绑定
  auto itt = m.def_submodule("_itt", "VTune ITT bindings");

  // 给 _itt 子模块添加 Python 绑定函数，is_available 函数用于判断 ITT 是否可用
  itt.def("is_available", itt_is_available);

  // 给 _itt 子模块添加 Python 绑定函数，rangePush 函数用于开始一个新的 ITT 范围
  itt.def("rangePush", itt_range_push);

  // 给 _itt 子模块添加 Python 绑定函数，rangePop 函数用于结束当前 ITT 范围
  itt.def("rangePop", itt_range_pop);

  // 给 _itt 子模块添加 Python 绑定函数，mark 函数用于标记一个 ITT 事件
  itt.def("mark", itt_mark);
}
} // namespace torch::profiler
```