# `.\pytorch\torch\csrc\DynamicTypes.h`

```py
#pragma once
// 声明此头文件为单次包含（once include），防止重复引用

// 提供 Python 张量对象与 at::Tensor 之间的转换

#include <torch/csrc/python_headers.h>
// 包含 Torch 的 Python 头文件，用于与 Python 交互

#include <ATen/Device.h>
#include <c10/core/Backend.h>
#include <c10/core/Layout.h>
#include <c10/core/ScalarType.h>
#include <c10/core/ScalarTypeToTypeMeta.h>
// 包含 ATen 和 c10 库的相关头文件

#include <torch/csrc/Export.h>
// 包含 Torch 的导出宏定义

#include <memory>
#include <string>
// 包含标准库的头文件

struct THPDtype;
struct THPLayout;
// 声明 THPDtype 和 THPLayout 结构体

namespace c10 {
struct Storage;
} // namespace c10
// 声明 c10 命名空间中的 Storage 结构体

namespace torch {
// Torch 命名空间开始

void registerDtypeObject(THPDtype* dtype, at::ScalarType scalarType);
// 注册 THPDtype 对象与 at::ScalarType 的关联

void registerLayoutObject(THPLayout* thp_layout, at::Layout layout);
// 注册 THPLayout 对象与 at::Layout 的关联

TORCH_PYTHON_API PyObject* createPyObject(const at::Storage& storage);
// 创建一个 Python 对象，表示给定的 at::Storage

at::Storage createStorage(PyObject* obj);
// 根据 Python 对象创建对应的 at::Storage

std::tuple<at::Storage, at::ScalarType, bool> createStorageGetType(
    PyObject* obj);
// 创建一个 at::Storage，并返回其类型和是否成功的元组

bool isStorage(PyObject* obj);
// 检查给定对象是否为 at::Storage 类型

// 以下两个方法返回一个借用的引用（borrowed reference）
TORCH_PYTHON_API THPDtype* getTHPDtype(at::ScalarType scalarType);
// 根据 at::ScalarType 获取对应的 THPDtype 对象

THPLayout* getTHPLayout(at::Layout layout);
// 根据 at::Layout 获取对应的 THPLayout 对象

} // namespace torch
// Torch 命名空间结束
```