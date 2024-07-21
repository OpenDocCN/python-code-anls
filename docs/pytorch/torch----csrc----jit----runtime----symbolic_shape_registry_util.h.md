# `.\pytorch\torch\csrc\jit\runtime\symbolic_shape_registry_util.h`

```
#pragma once
// 声明该文件仅被包含一次，直到 native_functions.yaml 和 derivatives.yaml 合并为止。理想情况下，这些内容应全部迁移到 native_functions.yaml 中

#include <torch/csrc/Export.h>
// 包含 Torch 的导出宏定义文件

#include <torch/csrc/jit/ir/ir.h>
// 包含 Torch 的 JIT 模块中的 IR 相关头文件

namespace torch::jit {

TORCH_API const OperatorMap<std::string>& get_tensorexpr_elementwise_set();
// 声明一个函数 get_tensorexpr_elementwise_set()，返回一个常引用的 OperatorMap<std::string> 对象，作为 Torch API 的一部分

} // namespace torch::jit
// 结束 torch::jit 命名空间
```