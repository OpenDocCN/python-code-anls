# `.\pytorch\torch\csrc\jit\passes\onnx\naming.h`

```
#pragma once

#include <torch/csrc/jit/ir/ir.h>  // 包含 Torch 的 IR 模块头文件

namespace torch {
namespace jit {
namespace onnx {

namespace ONNXScopeName {

// 创建完整的作用域名称，结合类名和变量名
std::string createFullScopeName(
    const std::string& class_name,       // 类名
    const std::string& variable_name);   // 变量名

// 根据作用域获取变量名
std::string variableName(torch::jit::ScopePtr scope);

// 从根节点开始获取变量名，使用指定的层级分隔符
std::string variableNameFromRoot(
    torch::jit::ScopePtr scope,          // 根作用域
    const std::string& layer_separator); // 层级分隔符

// 获取类名
std::string className(torch::jit::ScopePtr scope);

// 从根节点开始获取类名，使用指定的层级分隔符
std::string classNameFromRoot(
    torch::jit::ScopePtr scope,          // 根作用域
    const std::string& layer_separator); // 层级分隔符

// 检查作用域是否兼容
bool isCompatibleScope(torch::jit::ScopePtr scope);

} // namespace ONNXScopeName

// 为图中的节点和值分配作用域名称
TORCH_API void AssignScopedNamesForNodeAndValue(std::shared_ptr<Graph>& graph);

} // namespace onnx
} // namespace jit
} // namespace torch
```