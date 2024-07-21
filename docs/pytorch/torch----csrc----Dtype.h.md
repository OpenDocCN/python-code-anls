# `.\pytorch\torch\csrc\Dtype.h`

```py
#pragma once
// 预处理指令：指示编译器只包含本文件一次

#include <c10/core/ScalarType.h>
// 包含 C10 库中的 ScalarType 头文件

#include <torch/csrc/Export.h>
// 包含 Torch 导出宏定义的头文件

#include <torch/csrc/python_headers.h>
// 包含 Torch Python 头文件

constexpr int DTYPE_NAME_LEN = 64;
// 定义常量，表示数据类型名称的最大长度为 64

struct TORCH_API THPDtype {
  PyObject_HEAD at::ScalarType scalar_type;
  // 定义结构体 THPDtype，包含 PyObject_HEAD 和 at::ScalarType 类型的成员 scalar_type

  // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
  char name[DTYPE_NAME_LEN + 1];
  // 数据类型名称的字符数组成员，长度为 DTYPE_NAME_LEN + 1
};

TORCH_API extern PyTypeObject THPDtypeType;
// 声明 THPDtypeType 的外部 PyTypeObject 类型变量

inline bool THPDtype_Check(PyObject* obj) {
  // 内联函数：检查给定对象是否为 THPDtype 类型
  return Py_TYPE(obj) == &THPDtypeType;
}

inline bool THPPythonScalarType_Check(PyObject* obj) {
  // 内联函数：检查给定对象是否为 Python 中的标量类型
  return obj == (PyObject*)(&PyFloat_Type) ||
      obj == (PyObject*)(&PyComplex_Type) || obj == (PyObject*)(&PyBool_Type) ||
      obj == (PyObject*)(&PyLong_Type);
}

TORCH_API PyObject* THPDtype_New(
    at::ScalarType scalar_type,
    const std::string& name);
// 声明 THPDtype_New 函数：创建新的 THPDtype 对象

void THPDtype_init(PyObject* module);
// 声明 THPDtype_init 函数：初始化 THPDtype 相关的 Python 模块
```