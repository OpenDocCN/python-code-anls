# `.\pytorch\torch\csrc\Size.h`

```py
#pragma once


// 预处理指令：确保头文件只被包含一次，避免重复定义
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/python_headers.h>
#include <cstdint>

// 声明 Python 中的 THPSizeType 类型对象
extern PyTypeObject THPSizeType;

// 宏定义：检查对象是否为 THPSizeType 类型
#define THPSize_Check(obj) (Py_TYPE(obj) == &THPSizeType)

// 函数声明：根据 torch::autograd::Variable 创建 Python 对象 PyObject*
PyObject* THPSize_New(const torch::autograd::Variable& t);

// 函数声明：根据维度 dim 和大小数组 sizes 创建 Python 对象 PyObject*
PyObject* THPSize_NewFromSizes(int64_t dim, const int64_t* sizes);

// 函数声明：根据 at::Tensor t 的大小创建 Python 对象 PyObject*
PyObject* THPSize_NewFromSymSizes(const at::Tensor& t);

// 函数声明：初始化 THPSize 模块，参数为 Python 模块对象
void THPSize_init(PyObject* module);


这些注释按照要求分别解释了每行代码的作用，包括预处理指令、头文件引入、外部类型声明、宏定义和函数声明。
```