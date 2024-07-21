# `.\pytorch\torch\csrc\QScheme.h`

```py
#pragma once
// 防止头文件被多次包含的预处理指令

#include <torch/csrc/python_headers.h>
// 包含 Torch 的 Python 头文件

#include <c10/core/QScheme.h>
// 包含 C10 库中的 QScheme 头文件

#include <string>
// 包含标准 C++ 字符串头文件

constexpr int QSCHEME_NAME_LEN = 64;
// 定义常量，表示 QScheme 名称的最大长度为 64

struct THPQScheme {
  PyObject_HEAD
  at::QScheme qscheme;
  // 定义结构体 THPQScheme，包含一个 Torch 的 QScheme 对象
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
  char name[QSCHEME_NAME_LEN + 1];
  // 结构体中的字符数组，用于存储 QScheme 的名称，长度为 QSCHEME_NAME_LEN + 1
};

extern PyTypeObject THPQSchemeType;
// 声明一个外部的 PyTypeObject，表示 THPQScheme 的 Python 类型

inline bool THPQScheme_Check(PyObject* obj) {
  return Py_TYPE(obj) == &THPQSchemeType;
}
// 内联函数，检查给定的 Python 对象是否为 THPQScheme 类型的实例

PyObject* THPQScheme_New(at::QScheme qscheme, const std::string& name);
// 声明一个函数原型，用于创建新的 THPQScheme 对象的 Python 包装

void THPQScheme_init(PyObject* module);
// 声明一个函数原型，用于在 Python 模块中初始化 THPQScheme 相关内容
```