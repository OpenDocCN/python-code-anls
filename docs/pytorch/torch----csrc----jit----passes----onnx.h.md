# `.\pytorch\torch\csrc\jit\passes\onnx.h`

```
#pragma once
// 预处理指令，确保头文件只被编译一次

#include <torch/csrc/jit/ir/ir.h>
// 引入 Torch 库中的 IR 头文件

#include <torch/csrc/onnx/onnx.h>
// 引入 Torch 库中的 ONNX 头文件

#include <torch/csrc/utils/pybind.h>
// 引入 Torch 库中的 Python 绑定工具头文件

namespace torch {
namespace jit {

TORCH_API std::shared_ptr<Graph> ToONNX(
    std::shared_ptr<Graph>& state,
    ::torch::onnx::OperatorExportTypes operator_export_type);
// 将 Torch 图形转换为 ONNX 格式的图形

TORCH_API py::dict BlockToONNX(
    Block* old_block,
    Block* new_block,
    ::torch::onnx::OperatorExportTypes operator_export_type,
    py::dict& env,
    py::set& values_in_env,
    bool is_sub_block = false);
// 将 Torch 基本块转换为 ONNX 格式的字典

TORCH_API void NodeToONNX(
    Node* old_node,
    Block* new_block,
    ::torch::onnx::OperatorExportTypes operator_export_type,
    py::dict& env,
    py::set& values_in_env);
// 将 Torch 节点转换为 ONNX 格式

TORCH_API void RemovePrintOps(std::shared_ptr<Graph>& graph);
// 从 Torch 图中移除打印操作

TORCH_API void PreprocessCaffe2Ops(std::shared_ptr<Graph>& graph);
// 预处理 Torch 图中的 Caffe2 操作

} // namespace jit
} // namespace torch
// Torch JIT（即时编译）命名空间结束
```