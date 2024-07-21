# `.\pytorch\functorch\csrc\init_dim_only.cpp`

```
// 包含 Torch 扩展的头文件
#include <torch/extension.h>
// 包含 functorch 库中的维度处理相关头文件
#include <functorch/csrc/dim/dim.h>

// 定义 at 命名空间下的 functorch 子命名空间
namespace at {
namespace functorch {

// 使用 PYBIND11_MODULE 宏定义 Python 模块的初始化函数
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // 初始化第一类维度，并将其作为 _C 的子模块安装
  auto dim = Dim_init();
  // 如果初始化失败，则抛出 Python 异常
  if (!dim) {
    throw py::error_already_set();
  }
  // 将 dim 对象设置为 m 模块的属性
  py::setattr(m, "dim", py::reinterpret_steal<py::object>(dim));
}

}} // namespace at::functorch
```