# `.\pytorch\torch\csrc\jit\runtime\symbolic_script.h`

```
#pragma once
// 这个文件是临时的，直到 native_functions.yaml 和 derivatives.yaml 合并。
// 理想情况下，这些内容应该全部移入 native_functions.yaml

#include <c10/util/Optional.h>
#include <c10/util/StringUtil.h>
#include <torch/csrc/jit/api/module.h>

namespace torch::jit {
// 表示一个梯度对，包含前向图和反向图的共享指针
struct GradientPair {
  std::shared_ptr<Graph> forward;
  std::shared_ptr<Graph> backward;
};

// 获取给定函数模式的梯度信息，返回可选的梯度对
TORCH_API std::optional<GradientPair> gradientInfoForSchema(
    const FunctionSchema& schema);

// 检查给定函数模式是否有梯度信息
TORCH_API bool hasGradientInfoForSchema(const FunctionSchema& schema);
} // namespace torch::jit
```