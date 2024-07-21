# `.\pytorch\torch\csrc\jit\passes\utils\op_registry.cpp`

```py
// 包含 Torch JIT passes 中的 op_registry.h 头文件
#include <torch/csrc/jit/passes/utils/op_registry.h>

// 定义命名空间 torch::jit，用于 Torch JIT 功能
namespace torch {
namespace jit {

// 定义函数 ops_one_tensor_in_shape_transform，返回一个指向 OperatorSet 的共享指针
std::shared_ptr<OperatorSet> ops_one_tensor_in_shape_transform() {
  // 创建一个 OperatorSet 对象 ops，并使用列表初始化器初始化
  std::shared_ptr<OperatorSet> ops = std::make_shared<OperatorSet>(OperatorSet{
      "aten::flatten(Tensor self, int start_dim, int end_dim) -> Tensor",  // 注册 aten::flatten 操作
  });
  // 返回 ops 对象的共享指针
  return ops;
};

} // namespace jit
} // namespace torch
```