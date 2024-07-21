# `.\pytorch\torch\csrc\Generator.h`

```py
#pragma once
// 声明了一个预处理指令，指示编译器只包含此头文件一次

#include <ATen/core/Generator.h>
// 包含了 ATen 库中的 Generator 头文件

#include <torch/csrc/Export.h>
// 包含了 Torch 库中的 Export 头文件

#include <torch/csrc/python_headers.h>
// 包含了 Torch 的 Python 相关头文件

// NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
// 禁止 lint 工具对下一行进行类型成员初始化的检查
struct THPGenerator {
  // 定义了一个结构体 THPGenerator，其内部包含一个 at::Generator 类型的 cdata 成员
  PyObject_HEAD at::Generator cdata;
};

// 创建一个新的 Python 对象，包装默认的 at::Generator。引用是借用的。
// 调用者应确保 at::Generator 对象的生命周期至少与 Python 包装器一样长。
TORCH_PYTHON_API PyObject* THPGenerator_initDefaultGenerator(
    at::Generator cdata);

// 检查给定的对象是否是 THPGeneratorClass 类型的实例
#define THPGenerator_Check(obj) PyObject_IsInstance(obj, THPGeneratorClass)

// THPGeneratorClass 的声明
TORCH_PYTHON_API extern PyObject* THPGeneratorClass;

// 初始化 THPGenerator 模块
bool THPGenerator_init(PyObject* module);

// 包装一个 at::Generator 对象，创建一个新的 Python 对象
TORCH_PYTHON_API PyObject* THPGenerator_Wrap(at::Generator gen);

// 解包一个 Python 对象，获取其内部的 at::Generator 对象
TORCH_PYTHON_API at::Generator THPGenerator_Unwrap(PyObject* state);

// 创建一个新的 Python 对象用于 Generator。Generator 对象必须尚未关联 PyObject*
PyObject* THPGenerator_NewWithVar(PyTypeObject* type, at::Generator gen);
```