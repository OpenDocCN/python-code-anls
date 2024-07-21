# `.\pytorch\torch\csrc\jit\python\module_python.h`

```py
#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/utils/pybind.h>

namespace py = pybind11;

namespace torch::jit {

// 将给定的 Python 对象转换为 torch::jit::Module（模块），如果对象是 ScriptModule 类型的实例
inline std::optional<Module> as_module(py::handle obj) {
  // 导入 torch.jit 模块并获取 ScriptModule 类型的句柄
  static py::handle ScriptModule =
      py::module::import("torch.jit").attr("ScriptModule");
  // 如果 obj 是 ScriptModule 类型的实例，则返回其底层的 Module 对象
  if (py::isinstance(obj, ScriptModule)) {
    return py::cast<Module>(obj.attr("_c"));
  }
  // 否则返回空 optional
  return c10::nullopt;
}

// 将给定的 Python 对象转换为 torch::jit::Object（对象），如果对象是 ScriptObject 或 RecursiveScriptClass 类型的实例
inline std::optional<Object> as_object(py::handle obj) {
  // 导入 torch 模块并获取 ScriptObject 类型的句柄
  static py::handle ScriptObject =
      py::module::import("torch").attr("ScriptObject");
  // 如果 obj 是 ScriptObject 类型的实例，则返回其底层的 Object 对象
  if (py::isinstance(obj, ScriptObject)) {
    return py::cast<Object>(obj);
  }

  // 否则，导入 torch.jit 模块并获取 RecursiveScriptClass 类型的句柄
  static py::handle RecursiveScriptClass =
      py::module::import("torch.jit").attr("RecursiveScriptClass");
  // 如果 obj 是 RecursiveScriptClass 类型的实例，则返回其底层的 Object 对象
  if (py::isinstance(obj, RecursiveScriptClass)) {
    return py::cast<Object>(obj.attr("_c"));
  }
  // 否则返回空 optional
  return c10::nullopt;
}

} // namespace torch::jit
```