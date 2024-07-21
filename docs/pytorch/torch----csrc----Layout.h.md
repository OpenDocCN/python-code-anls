# `.\pytorch\torch\csrc\Layout.h`

```py
#pragma once
// 预处理指令，确保本头文件只被编译一次

#include <torch/csrc/python_headers.h>
// 引入 Torch 的 Python 头文件

#include <ATen/Layout.h>
// 引入 ATen 库的布局相关头文件

#include <string>
// 引入标准库中的字符串处理功能

const int LAYOUT_NAME_LEN = 64;
// 定义常量，布局名称的最大长度为 64

struct THPLayout {
  PyObject_HEAD
  at::Layout layout;
  // Python 对象头部，表示此结构体可以作为 Python 对象
  // ATen 库的布局对象

  // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
  char name[LAYOUT_NAME_LEN + 1];
  // 布局名称的字符数组，额外预留一个字符存放字符串结尾的空字符
};

extern PyTypeObject THPLayoutType;
// 外部声明，表示 THPLayoutType 是一个 PyTypeObject 类型的变量

inline bool THPLayout_Check(PyObject* obj) {
  return Py_TYPE(obj) == &THPLayoutType;
}
// 内联函数，检查给定的 Python 对象是否为 THPLayoutType 类型

PyObject* THPLayout_New(at::Layout layout, const std::string& name);
// 函数声明，创建一个新的 THPLayout 对象并返回对应的 Python 对象

void THPLayout_init(PyObject* module);
// 函数声明，初始化 THPLayout 模块
```