# `.\pytorch\torch\csrc\MemoryFormat.h`

```
#pragma once
// 预处理指令：确保头文件只被编译一次

#include <torch/csrc/python_headers.h>
// 包含 Torch 的 Python 头文件

#include <c10/core/MemoryFormat.h>
// 包含 C10 的内存格式定义

#include <string>
// 包含 C++ 标准库中的 string 头文件

const int MEMORY_FORMAT_NAME_LEN = 64;
// 声明常量，指定内存格式名称的最大长度为 64

struct THPMemoryFormat {
  PyObject_HEAD 
  // 定义 Python 对象的头部

  at::MemoryFormat memory_format;
  // 在结构体中定义 ATen 的内存格式对象

  // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
  char name[MEMORY_FORMAT_NAME_LEN + 1];
  // 字符数组，用于存储内存格式的名称，长度为常量 MEMORY_FORMAT_NAME_LEN + 1
};

extern PyTypeObject THPMemoryFormatType;
// 声明一个外部可见的 Python 类型对象 THPMemoryFormatType

inline bool THPMemoryFormat_Check(PyObject* obj) {
  return Py_TYPE(obj) == &THPMemoryFormatType;
}
// 内联函数：检查给定对象是否为 THPMemoryFormat 类型的实例

PyObject* THPMemoryFormat_New(
    at::MemoryFormat memory_format,
    const std::string& name);
// 函数声明：创建一个新的 THPMemoryFormat 对象

void THPMemoryFormat_init(PyObject* module);
// 函数声明：初始化模块，定义 THPMemoryFormat 类型和相关操作
```