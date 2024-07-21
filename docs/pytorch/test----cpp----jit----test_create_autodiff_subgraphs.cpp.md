# `.\pytorch\test\cpp\jit\test_create_autodiff_subgraphs.cpp`

```py
#include <gtest/gtest.h>

#include "test/cpp/jit/test_utils.h"

#include "torch/csrc/jit/passes/create_autodiff_subgraphs.h"

// 命名空间 torch::jit 中的代码
namespace torch {
namespace jit {

// 定义单元测试 CreateAutodiffSubgraphsTest.Basic
TEST(CreateAutodiffSubgraphsTest, Basic) {
  // 构建 LSTM 图形
  auto graph = build_lstm();
  // 调用 CreateAutodiffSubgraphs 函数，设定阈值为 2
  CreateAutodiffSubgraphs(graph, /*threshold=*/2);
  // 使用 testing::FileCheck 进行断言
  testing::FileCheck()
      // 检查 DifferentiableGraph 中不包含指定操作
      .check_not("aten::mm")
      ->check_not("aten::sigmoid")
      ->check_not("aten::tanh")
      ->check_not("aten::mul")
      // 检查是否包含 DifferentiableGraph 和 return
      ->check("DifferentiableGraph")
      ->check_next("return")
      ->run(*graph);
}

} // namespace jit
} // namespace torch
```