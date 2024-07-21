# `.\pytorch\torch\csrc\api\include\torch\python.h`

```
/// 预处理指令，确保头文件只被包含一次
#pragma once

/// 包含 Torch 库的静态定义
#include <torch/detail/static.h>
/// 包含 Torch 的模块定义
#include <torch/nn/module.h>
/// 包含 Torch 的有序字典定义
#include <torch/ordered_dict.h>
/// 包含 Torch 的类型定义
#include <torch/types.h>

/// 包含 Torch C++ 实现中的设备定义
#include <torch/csrc/Device.h>
/// 包含 Torch C++ 实现中的数据类型定义
#include <torch/csrc/Dtype.h>
/// 包含 Torch C++ 实现中的动态类型定义
#include <torch/csrc/DynamicTypes.h>
/// 包含 Torch C++ 实现中的异常定义
#include <torch/csrc/Exceptions.h>
/// 包含 Torch C++ 实现中的自动求导变量定义
#include <torch/csrc/autograd/python_variable.h>
/// 包含 Torch C++ 实现中的 Python 头文件
#include <torch/csrc/python_headers.h>
/// 包含 Torch C++ 实现中的 Python 绑定工具函数
#include <torch/csrc/utils/pybind.h>
/// 包含 Torch C++ 实现中的 Python 数字处理工具函数
#include <torch/csrc/utils/python_numbers.h>
/// 包含 Torch C++ 实现中的 Python 元组处理工具函数
#include <torch/csrc/utils/python_tuples.h>

/// 包含 C++ 标准库中的迭代器定义
#include <iterator>
/// 包含 C++ 标准库中的字符串定义
#include <string>
/// 包含 C++ 标准库中的无序映射定义
#include <unordered_map>
/// 包含 C++ 标准库中的实用工具定义
#include <utility>
/// 包含 C++ 标准库中的向量定义
#include <vector>

/// Torch 命名空间
namespace torch {
/// Python 绑定命名空间
namespace python {
/// 内部实现细节命名空间
namespace detail {

/// 将 Python 对象转换为 Torch 设备对象
inline Device py_object_to_device(py::object object) {
  // 获取 Python 对象的指针
  PyObject* obj = object.ptr();
  // 检查是否为 Torch 设备对象
  if (THPDevice_Check(obj)) {
    // 将 Torch 设备对象转换为 Device 返回
    return reinterpret_cast<THPDevice*>(obj)->device;
  }
  // 抛出类型错误异常，期望得到设备对象
  throw TypeError("Expected device");
}

/// 将 Python 对象转换为 Torch 数据类型对象
inline Dtype py_object_to_dtype(py::object object) {
  // 获取 Python 对象的指针
  PyObject* obj = object.ptr();
  // 检查是否为 Torch 数据类型对象
  if (THPDtype_Check(obj)) {
    // 将 Torch 数据类型对象转换为 Dtype 返回
    return reinterpret_cast<THPDtype*>(obj)->scalar_type;
  }
  // 抛出类型错误异常，期望得到数据类型对象
  throw TypeError("Expected dtype");
}

/// 定义用于 Python 模块绑定的模板类型别名
template <typename ModuleType>
using PyModuleClass =
    py::class_<ModuleType, torch::nn::Module, std::shared_ptr<ModuleType>>;

/// 动态创建 `torch.nn.cpp.ModuleWrapper` 的子类，并且是 `torch.nn.Module` 的子类，
/// 将用户提供的 C++ 模块委托给它处理所有调用。
template <typename ModuleType>
void bind_cpp_module_wrapper(
    py::module module,
    PyModuleClass<ModuleType> cpp_class,
    const char* name) {
  // 获取 `torch.nn.cpp.ModuleWrapper` 类，我们将在下面动态创建一个子类
  py::object cpp_module =
      py::module::import("torch.nn.cpp").attr("ModuleWrapper");

  // 获取 `type` 类，我们将使用它作为元类来动态创建一个新的类
  py::object type_metaclass =
      py::reinterpret_borrow<py::object>((PyObject*)&PyType_Type);

  // `ModuleWrapper` 构造函数会在其构造函数中将所有函数复制到自己的 `__dict__` 中，
  // 但我们需要为我们的动态类提供一个构造函数。在构造函数内部，我们创建原始的 C++ 模块实例
  // （即我们要绑定的 `torch::nn::Module` 子类），然后将其传递给 `ModuleWrapper` 构造函数。
  py::dict attributes;

  // `type()` 函数始终需要一个 `str` 类型参数，但 pybind11 的 `str()` 方法总是创建一个 `unicode` 对象。
  py::object name_str = py::str(name);

  // 动态创建 `ModuleWrapper` 的子类，它是 `torch.nn.Module` 的子类，并将所有调用委托给我们要绑定的 C++ 模块。
  py::object wrapper_class =
      type_metaclass(name_str, py::make_tuple(cpp_module), attributes);

  // 动态类的构造函数调用 `ModuleWrapper.__init__()`，后者会用 C++ 模块的方法替换其自身的方法。
  wrapper_class.attr("__init__") = py::cpp_function(
      [cpp_module, cpp_class](
          py::object self, py::args args, py::kwargs kwargs) {
        cpp_module.attr("__init__")(self, cpp_class(*args, **kwargs));
      },
      py::is_method(wrapper_class));

  // 调用 `my_module.my_class` 现在意味着 `my_class` 是 `ModuleWrapper` 的子类，
  // 其方法调用进入我们要绑定的 C++ 模块。
  module.attr(name) = wrapper_class;
/// 结束 `detail` 命名空间
}
} // namespace detail

/// 为绑定 `nn::Module` 子类的 `pybind11 class_` 添加方法绑定。
///
/// 假设你已经使用 `py::class_<Net>(m, "Net")` 创建了一个 pybind11 类对象，
/// 这个函数将添加所有必要的 `.def()` 调用，以绑定 `nn::Module` 基类的方法，
/// 如 `train()`、`eval()` 等到 Python 中。
///
/// 如果可能，用户应该优先使用 `bind_module`。
template <typename ModuleType, typename... Extra>
py::class_<ModuleType, Extra...> add_module_bindings(
}

/// 创建一个 `nn::Module` 子类的 pybind11 类对象，并添加默认绑定。
///
/// 添加默认绑定后，返回类对象，你可以继续添加更多的绑定。
///
/// 示例用法：
/// \rst
/// .. code-block:: cpp
///
///   struct Net : torch::nn::Module {
///     Net(int in, int out) { }
///     torch::Tensor forward(torch::Tensor x) { return x; }
///   };
///
///   PYBIND11_MODULE(my_module, m) {
///     torch::python::bind_module<Net>(m, "Net")
///       .def(py::init<int, int>())
///       .def("forward", &Net::forward);
///  }
/// \endrst
template <typename ModuleType, bool force_enable = false>
std::enable_if_t<
    !torch::detail::has_forward<ModuleType>::value || force_enable,
    detail::PyModuleClass<ModuleType>>
bind_module(py::module module, const char* name) {
  // 创建一个名为 `cpp` 的子模块
  py::module cpp = module.def_submodule("cpp");
  // 调用 `add_module_bindings`，并将结果存储在 `cpp_class` 中
  auto cpp_class =
      add_module_bindings(detail::PyModuleClass<ModuleType>(cpp, name));
  // 调用 `bind_cpp_module_wrapper`，为 `module` 绑定 CPP 模块的包装器
  detail::bind_cpp_module_wrapper(module, cpp_class, name);
  // 返回 `cpp_class`
  return cpp_class;
}

/// 创建一个 `nn::Module` 子类的 pybind11 类对象，并添加默认绑定。
///
/// 添加默认绑定后，返回类对象，你可以继续添加更多的绑定。
///
/// 如果类有 `forward()` 方法，它将自动在 Python 中暴露为 `forward()` 和 `__call__`。
///
/// 示例用法：
/// \rst
/// .. code-block:: cpp
///
///   struct Net : torch::nn::Module {
///     Net(int in, int out) { }
///     torch::Tensor forward(torch::Tensor x) { return x; }
///   };
///
///   PYBIND11_MODULE(my_module, m) {
///     torch::python::bind_module<Net>(m, "Net")
///       .def(py::init<int, int>())
///       .def("forward", &Net::forward);
///  }
/// \endrst
template <
    typename ModuleType,
    typename = std::enable_if_t<torch::detail::has_forward<ModuleType>::value>>
detail::PyModuleClass<ModuleType> bind_module(
    py::module module,
    const char* name) {
  // 调用 `bind_module`，强制启用为真
  return bind_module<ModuleType, /*force_enable=*/true>(module, name)
      .def("forward", &ModuleType::forward) // 绑定 `forward()` 方法
      .def("__call__", &ModuleType::forward); // 绑定 `__call__` 方法到 `forward()`
}
} // namespace python
} // namespace torch
```