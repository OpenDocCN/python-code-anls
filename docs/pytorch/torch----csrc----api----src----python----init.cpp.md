# `.\pytorch\torch\csrc\api\src\python\init.cpp`

```
// 包含 Torch 的 Python 头文件
#include <torch/python.h>
// 包含 Torch 初始化相关头文件
#include <torch/python/init.h>

// 包含 Torch 的神经网络模块相关头文件
#include <torch/nn/module.h>
// 包含 Torch 的有序字典相关头文件
#include <torch/ordered_dict.h>

// 包含 Torch 的 Python 绑定工具相关头文件
#include <torch/csrc/utils/pybind.h>

// 包含标准库中的字符串和向量处理头文件
#include <string>
#include <vector>

// 定义命名空间别名 py 为 pybind11
namespace py = pybind11;

// 定义 pybind11 内部命名空间 detail
namespace pybind11 {
namespace detail {

// 定义宏 ITEM_TYPE_CASTER，用于创建类型转换器
#define ITEM_TYPE_CASTER(T, Name)                                             \
  template <>                                                                 \
  struct type_caster<typename torch::OrderedDict<std::string, T>::Item> {     \
   public:                                                                    \
    using Item = typename torch::OrderedDict<std::string, T>::Item;           \
    using PairCaster = make_caster<std::pair<std::string, T>>;                \
    // 设置类型转换器的类型名称
    PYBIND11_TYPE_CASTER(Item, _("Ordered" #Name "DictItem"));                \
    // 加载类型转换器，将 Python 对象转换为 C++ 对象
    bool load(handle src, bool convert) {                                     \
      return PairCaster().load(src, convert);                                 \
    }                                                                         \
    // 执行类型转换，将 C++ 对象转换为 Python 对象
    static handle cast(Item src, return_value_policy policy, handle parent) { \
      return PairCaster::cast(                                                \
          src.pair(), std::move(policy), std::move(parent));                  \
    }                                                                         \
  }

// 定义 ITEM_TYPE_CASTER 宏的实现，转换 torch::Tensor 类型
// NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
ITEM_TYPE_CASTER(torch::Tensor, Tensor);
// 定义 ITEM_TYPE_CASTER 宏的实现，转换 std::shared_ptr<torch::nn::Module> 类型
// NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
ITEM_TYPE_CASTER(std::shared_ptr<torch::nn::Module>, Module);
} // namespace detail
} // namespace pybind11

// 定义 Torch 命名空间
namespace torch {
// 定义 Torch Python 绑定命名空间
namespace python {
// 匿名命名空间，定义模板函数 bind_ordered_dict
namespace {
template <typename T>
void bind_ordered_dict(py::module module, const char* dict_name) {
  // 使用 OrderedDict 定义 ODict 类型
  using ODict = OrderedDict<std::string, T>;
  // clang-format off
  // 创建 Python 类型绑定，绑定名称为 dict_name
  py::class_<ODict>(module, dict_name)
      // 定义 items 方法，返回有序字典的键值对
      .def("items", &ODict::items)
      // 定义 keys 方法，返回有序字典的键
      .def("keys", &ODict::keys)
      // 定义 values 方法，返回有序字典的值
      .def("values", &ODict::values)
      // 定义 __iter__ 方法，使其可迭代
      .def("__iter__", [](const ODict& dict) {
            return py::make_iterator(dict.begin(), dict.end());
          }, py::keep_alive<0, 1>())
      // 定义 __len__ 方法，返回有序字典的大小
      .def("__len__", &ODict::size)
      // 定义 __contains__ 方法，判断键是否存在于有序字典中
      .def("__contains__", &ODict::contains)
      // 定义 __getitem__ 方法，根据键获取值
      .def("__getitem__", [](const ODict& dict, const std::string& key) {
        return dict[key];
      })
      // 定义 __getitem__ 方法，根据索引获取值
      .def("__getitem__", [](const ODict& dict, size_t index) {
        return dict[index];
      });
  // clang-format on
}
} // namespace

// 初始化 Torch Python 绑定函数
void init_bindings(PyObject* module) {
  // 将 PyObject 转换为 py::module
  py::module m = py::handle(module).cast<py::module>();
  // 在模块 m 中定义子模块 cpp
  py::module cpp = m.def_submodule("cpp");

  // 绑定 OrderedTensorDict 类型
  bind_ordered_dict<Tensor>(cpp, "OrderedTensorDict");
  // 绑定 OrderedModuleDict 类型
  bind_ordered_dict<std::shared_ptr<nn::Module>>(cpp, "OrderedModuleDict");

  // 在 cpp 模块中定义子模块 nn
  py::module nn = cpp.def_submodule("nn");
  // 添加 nn::Module 的绑定
  add_module_bindings(
      py::class_<nn::Module, std::shared_ptr<nn::Module>>(nn, "Module"));
}
} // namespace python
} // namespace torch
```