# `.\pytorch\torch\csrc\utils\python_stub.h`

```
#pragma once

#pragma once 是一个预处理指令，用于在编译时确保头文件只包含一次，防止多次包含同一头文件导致的重定义错误。


struct _object;

定义了一个名为 _object 的结构体，但没有给出具体的结构体定义。这种声明通常用于引用结构体而不需要详细的结构体定义。


using PyObject = _object;

使用别名声明 `PyObject`，将 `_object` 结构体命名为 `PyObject`，从而可以通过 `PyObject` 使用 `_object` 结构体的功能和属性。
```