# `.\pytorch\torch\csrc\utils\python_symnode.cpp`

```py
#include <torch/csrc/utils/python_symnode.h>

namespace torch {

py::handle get_symint_class() {
  // NB: leak
  // 静态变量，存储 torch.SymInt 类的 Python 句柄，避免多次创建
  static py::handle symint_class =
      py::object(py::module::import("torch").attr("SymInt")).release();
  // 返回 torch.SymInt 类的 Python 句柄
  return symint_class;
}

py::handle get_symfloat_class() {
  // NB: leak
  // 静态变量，存储 torch.SymFloat 类的 Python 句柄，避免多次创建
  static py::handle symfloat_class =
      py::object(py::module::import("torch").attr("SymFloat")).release();
  // 返回 torch.SymFloat 类的 Python 句柄
  return symfloat_class;
}

py::handle get_symbool_class() {
  // NB: leak
  // 静态变量，存储 torch.SymBool 类的 Python 句柄，避免多次创建
  static py::handle symbool_class =
      py::object(py::module::import("torch").attr("SymBool")).release();
  // 返回 torch.SymBool 类的 Python 句柄
  return symbool_class;
}

} // namespace torch
```