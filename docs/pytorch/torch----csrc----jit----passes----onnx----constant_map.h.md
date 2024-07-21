# `.\pytorch\torch\csrc\jit\passes\onnx\constant_map.h`

```
#pragma once
// 使用预处理器指令 #pragma once，确保头文件只被包含一次

#include <c10/macros/Macros.h>
// 包含 c10 库中的宏定义文件 Macros.h

C10_DIAGNOSTIC_PUSH_AND_IGNORED_IF_DEFINED("-Wsuggest-override")
// 开启诊断并忽略编译警告 "-Wsuggest-override"

C10_DIAGNOSTIC_PUSH_AND_IGNORED_IF_DEFINED("-Wnewline-eof")
// 开启诊断并忽略编译警告 "-Wnewline-eof"

#include <onnx/shape_inference/implementation.h>
// 包含 ONNX 库中形状推断实现的头文件 implementation.h

C10_DIAGNOSTIC_POP()
// 恢复之前的诊断设置

C10_DIAGNOSTIC_POP()
// 恢复之前的诊断设置

#include <torch/csrc/jit/ir/ir.h>
// 包含 Torch 库中的 IR 相关头文件 ir.h

#include <torch/csrc/jit/serialization/export.h>
// 包含 Torch 库中的序列化导出头文件 export.h

#include <mutex>
// 包含 C++ 标准库中的互斥量头文件

#include <unordered_map>
// 包含 C++ 标准库中的无序映射头文件

namespace torch {
namespace jit {
// 进入 torch 和 jit 命名空间

using ShapeDataMap =
    std::unordered_map<std::string, ::ONNX_NAMESPACE::TensorShapeProto>;
// 定义 ShapeDataMap 类型为 std::unordered_map<std::string, ::ONNX_NAMESPACE::TensorShapeProto>，用于映射字符串到 ONNX 的张量形状结构体

}; // namespace jit
}; // namespace torch
// 结束 jit 和 torch 命名空间
```