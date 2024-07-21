# `.\pytorch\torch\csrc\jit\passes\onnx\function_extraction.h`

```py
// 预处理指令，表示此头文件只需包含一次
#pragma once

// 包含Torch的IR相关头文件
#include <torch/csrc/jit/ir/ir.h>

// 定义torch命名空间
namespace torch {
// 定义jit命名空间，包含了Torch JIT的功能
namespace jit {

// onnx命名空间，用于导出和序列化ONNX模型相关功能
namespace onnx {

// NodeAttrNameMap用于跟踪无法通过Torch IR追踪的函数属性信息
// NodeAttrNameMap跟踪从函数子图内部的IR节点属性名称到函数属性名称的映射
// 示例中展示了CELU和LayerNorm的导出情况
using NodeAttrNameMap = std::
    unordered_map<const Node*, std::unordered_map<std::string, std::string>>;

// 函数声明：用于在图上提取函数属性信息并进行转换
// 返回NodeAttrNameMap类型的对象，包含函数属性的映射信息
TORCH_API NodeAttrNameMap ONNXFunctionExtraction(
    std::shared_ptr<Graph>& graph,
    const std::unordered_set<std::string>& module_names,
    const std::vector<std::string>& param_names);

// 函数声明：清除ONNX导出中的作用域记录
TORCH_API void ONNXClearScopeRecords();

// 函数声明：跟踪作用域内的属性信息，以便导出到ONNX模型
// attributes参数用于记录作用域内的属性信息
TORCH_API void ONNXTrackScopeAttributes(
    std::shared_ptr<Graph>& graph,
    std::map<std::string, IValue>& attributes);

} // namespace onnx

} // namespace jit
} // namespace torch
```