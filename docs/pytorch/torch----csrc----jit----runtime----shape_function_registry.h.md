# `.\pytorch\torch\csrc\jit\runtime\shape_function_registry.h`

```
#pragma once
// 使用 #pragma once 防止头文件的多重包含

#include <torch/csrc/Export.h>
// 包含 torch 库的导出头文件

#include <torch/csrc/jit/ir/ir.h>
// 包含 torch 的 JIT 模块中 IR 相关的头文件

namespace torch::jit {

TORCH_API const std::string& GetSerializedFuncs();
// 声明一个返回常量引用的函数 GetSerializedFuncs()，用于获取序列化函数的字符串

TORCH_API const OperatorMap<std::string>& GetFuncMapping();
// 声明一个返回常量引用的 OperatorMap<std::string> 类型对象的函数 GetFuncMapping()，用于获取函数映射

} // namespace torch::jit
// 声明 torch::jit 命名空间，包含了与 JIT 相关的函数和类
```