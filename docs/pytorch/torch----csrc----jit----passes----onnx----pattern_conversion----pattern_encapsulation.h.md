# `.\pytorch\torch\csrc\jit\passes\onnx\pattern_conversion\pattern_encapsulation.h`

```py
#pragma once

#include <torch/csrc/jit/ir/ir.h>  // 包含 Torch 的 IR 相关头文件

namespace torch {
namespace jit {

// 引言
//
// 这部分封装会查找模式的节点，类似于其他预先 ONNX 转换的方式编写的方式。但不同于转换节点，
// 它会将它们封装成新占位符节点的子块。此部分在 ONNX 转换之前运行，因此在调用符号函数之前运行。
//
// 注意：为什么将功能分为两部分
//
// 目的是支持依赖于形状和类型信息的转换。形状和类型信息仅在 _jit_pass_onnx 之后才可用，
// 后者将 aten 节点转换为 onnx 节点。因此存在相互依赖问题。_jit_pass_onnx 依赖于预处理
// 传递，将 aten 节点转换为可转换条件，并且预处理传递依赖于 _jit_pass_onnx 来转换上游节点
// 并应用 onnx 形状推断。将传递分成两部分可以打破这种相互依赖关系。
//
// 注意：编辑模式封装
//
// 封装步骤识别模式，并将节点复制到新占位符节点的子块中。新占位符节点的输出用于替代原始节点。
// 模式的类别存储为 attr::name。
TORCH_API std::optional<Node*> EncapsulatePatternIntoSubblock(Node* n);

} // namespace jit
} // namespace torch
```