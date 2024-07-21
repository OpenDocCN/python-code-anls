# `.\pytorch\torch\csrc\jit\frontend\builtin_functions.h`

```py
#pragma once
// 使用预处理指令#pragma once，确保头文件只被编译一次

#include <torch/csrc/Export.h>
// 包含torch库中的Export.h头文件，用于导出符号的宏定义

#include <torch/csrc/jit/api/module.h>
// 包含torch库中jit模块的API接口中module.h头文件

namespace torch {
namespace jit {

TORCH_API const std::vector<Function*>& getAllBuiltinFunctionsFor(Symbol name);
// torch命名空间下的jit命名空间中声明函数getAllBuiltinFunctionsFor，返回一个指向函数指针向量的常量引用，
// 该函数接受一个Symbol类型的参数name作为输入参数

} // namespace jit
} // namespace torch
// 结束torch命名空间和jit命名空间的定义
```