# `.\pytorch\torch\csrc\fx\node.h`

```
#pragma once

指示预处理器仅包含此头文件一次，用于防止多次包含同一头文件造成的重复定义错误。


#include <torch/csrc/python_headers.h>

包含名为`torch/csrc/python_headers.h`的头文件，用于导入必要的Python头文件以支持与Python的接口交互。


bool NodeBase_init(PyObject* module);

声明一个名为`NodeBase_init`的函数，该函数接受一个名为`module`的PyObject指针参数，并返回一个布尔类型的值。该函数用于初始化NodeBase模块。


bool NodeIter_init(PyObject* module);

声明一个名为`NodeIter_init`的函数，该函数接受一个名为`module`的PyObject指针参数，并返回一个布尔类型的值。该函数用于初始化NodeIter模块。
```