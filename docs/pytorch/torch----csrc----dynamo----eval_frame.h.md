# `.\pytorch\torch\csrc\dynamo\eval_frame.h`

```
#pragma once
#include <Python.h>


// 告诉编译器只包含此头文件一次，防止重复定义
// 包含 Python.h 头文件，该头文件提供了使用 Python C API 的功能



extern "C" {
PyObject* torch_c_dynamo_eval_frame_init(void);
}


// 声明一个 extern "C" 的 C 函数接口，以便在 C++ 代码中使用 C 的方式来调用
// 声明了一个名为 torch_c_dynamo_eval_frame_init 的函数，返回类型为 PyObject*
// 函数没有参数
```