# `.\pytorch\torch\csrc\cpu\Module.cpp`

```
#include <ATen/cpu/Utils.h>
#include <torch/csrc/cpu/Module.h>
#include <torch/csrc/utils/pybind.h>

namespace torch::cpu {

// 初始化模块函数，将传入的 Python 模块对象转换为 py::module 类型
void initModule(PyObject* module) {
  auto m = py::handle(module).cast<py::module>();

  // 创建名为 "_cpu" 的子模块，描述为与 CPU 相关的 pybind
  auto cpu = m.def_submodule("_cpu", "cpu related pybind.");
  
  // 将 C++ 函数绑定为 Python 可调用对象，检查当前 CPU 是否支持 AVX2 指令集
  cpu.def("_is_cpu_support_avx2", at::cpu::is_cpu_support_avx2);
  
  // 将 C++ 函数绑定为 Python 可调用对象，检查当前 CPU 是否支持 AVX512 指令集
  cpu.def("_is_cpu_support_avx512", at::cpu::is_cpu_support_avx512);
  
  // 将 C++ 函数绑定为 Python 可调用对象，检查当前 CPU 是否支持 AVX512_VNNI 指令集
  cpu.def("_is_cpu_support_avx512_vnni", at::cpu::is_cpu_support_avx512_vnni);
  
  // 将 C++ 函数绑定为 Python 可调用对象，检查当前 CPU 是否支持 AMX_TILE 指令集
  cpu.def("_is_cpu_support_amx_tile", at::cpu::is_cpu_support_amx_tile);
  
  // 将 C++ 函数绑定为 Python 可调用对象，初始化 AMX 模块
  cpu.def("_init_amx", at::cpu::init_amx);
}

} // namespace torch::cpu
```