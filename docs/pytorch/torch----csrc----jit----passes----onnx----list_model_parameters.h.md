# `.\pytorch\torch\csrc\jit\passes\onnx\list_model_parameters.h`

```py
// 使用#pragma once确保头文件只被编译一次，避免重复包含
#pragma once

// 包含Torch库的模块定义和IR定义的头文件
#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/ir/ir.h>

// torch命名空间包含了所有Torch相关的代码
namespace torch {
// jit命名空间包含了Torch JIT编译器的功能
namespace jit {

// TORCH_API声明了一个公共接口函数，用于列出模块的参数
// 返回一个Module对象和一个IValue向量的pair
TORCH_API std::pair<Module, std::vector<IValue>> list_module_parameters(
    const Module& module);

} // namespace jit
} // namespace torch
```