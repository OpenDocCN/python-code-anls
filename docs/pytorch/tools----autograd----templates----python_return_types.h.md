# `.\pytorch\tools\autograd\templates\python_return_types.h`

```
#pragma once

// 使用 `#pragma once` 预处理指令，确保头文件只被包含一次，避免重复定义错误。


namespace torch {
namespace autograd {
namespace generated {

// 定义了嵌套的命名空间 `torch::autograd::generated`，用于组织和隔离代码，防止命名冲突。


${py_return_types_declarations}

// 插入一个由外部环境定义的占位符 `${py_return_types_declarations}`，可能用于在编译时生成特定的返回类型声明。


}

// 结束命名空间 `generated`。


void initReturnTypes(PyObject* module);

// 声明函数 `initReturnTypes`，该函数接受一个 `PyObject*` 类型的参数 `module`，用于初始化返回类型。


} // namespace autograd
} // namespace torch

// 结束命名空间 `autograd` 和 `torch`。
```