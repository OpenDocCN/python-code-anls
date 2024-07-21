# `.\pytorch\torch\csrc\jit\serialization\onnx.h`

```
#pragma once
// 防止头文件被多次包含

#include <onnx/onnx_pb.h>
// 包含 ONNX 的 Protocol Buffer 头文件

#include <torch/csrc/jit/ir/ir.h>
// 包含 Torch 的 JIT IR 头文件

namespace torch::jit {
// 进入 torch::jit 命名空间

TORCH_API std::string prettyPrint(const ::ONNX_NAMESPACE::ModelProto& model);
// 声明一个名为 prettyPrint 的函数，接受 ONNX 的 ModelProto 对象作为参数，并返回一个字符串

} // namespace torch::jit
// 退出 torch::jit 命名空间
```