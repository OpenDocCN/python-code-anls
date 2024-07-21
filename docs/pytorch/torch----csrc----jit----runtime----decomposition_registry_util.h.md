# `.\pytorch\torch\csrc\jit\runtime\decomposition_registry_util.h`

```py
#pragma once
// 预处理指令，确保头文件只被编译一次

#include <torch/csrc/Export.h>
// 包含 Torch 库的导出头文件，用于声明导出符号

#include <torch/csrc/jit/ir/ir.h>
// 包含 Torch 的 JIT 模块中的 IR 头文件，用于声明 IR 相关的类和函数

namespace torch::jit {

TORCH_API const std::string& GetSerializedDecompositions();
// 声明一个函数 GetSerializedDecompositions()，返回一个常量引用的 std::string
// 该函数用于获取序列化的分解信息

TORCH_API const OperatorMap<std::string>& GetDecompositionMapping();
// 声明一个函数 GetDecompositionMapping()，返回一个常量引用的 OperatorMap<std::string>
// 该函数用于获取操作符到分解字符串的映射

} // namespace torch::jit
// 命名空间结束
```