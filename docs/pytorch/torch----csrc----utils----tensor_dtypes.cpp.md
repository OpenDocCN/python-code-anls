# `.\pytorch\torch\csrc\utils\tensor_dtypes.cpp`

```
// 引入 Torch 的头文件，用于处理数据类型、异常和 Python 的头文件
#include <torch/csrc/Dtype.h>
#include <torch/csrc/DynamicTypes.h>
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/python_headers.h>
#include <torch/csrc/utils/object_ptr.h>
#include <torch/csrc/utils/tensor_dtypes.h>

// 定义 torch::utils 命名空间，用于包装相关工具函数
namespace torch::utils {

// 初始化数据类型函数
void initializeDtypes() {
  // 导入 Python 中的 torch 模块
  auto torch_module = THPObjectPtr(PyImport_ImportModule("torch"));
  // 如果导入失败，抛出 Python 异常
  if (!torch_module)
    throw python_error();

  // 使用宏展开定义所有标量数据类型的集合
#define DEFINE_SCALAR_TYPE(_1, n) at::ScalarType::n,
  auto all_scalar_types = {
      AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS(DEFINE_SCALAR_TYPE)};
#undef DEFINE_SCALAR_TYPE

  // 遍历所有标量数据类型
  for (at::ScalarType scalarType : all_scalar_types) {
    // 获取数据类型的主要名称和旧名称
    auto [primary_name, legacy_name] = c10::getDtypeNames(scalarType);
    
    // 创建新的 Python 数据类型对象
    PyObject* dtype = THPDtype_New(scalarType, primary_name);
    // 在 torch 模块中注册数据类型对象
    torch::registerDtypeObject((THPDtype*)dtype, scalarType);
    // 增加对象的引用计数
    Py_INCREF(dtype);
    
    // 将数据类型对象添加到 torch 模块中，使用主要名称
    if (PyModule_AddObject(torch_module.get(), primary_name.c_str(), dtype) != 0) {
      throw python_error();
    }

    // 如果存在旧名称，同样将对象添加到 torch 模块中，使用旧名称
    if (!legacy_name.empty()) {
      Py_INCREF(dtype);
      if (PyModule_AddObject(torch_module.get(), legacy_name.c_str(), dtype) != 0) {
        throw python_error();
      }
    }
  }
}

} // namespace torch::utils
```