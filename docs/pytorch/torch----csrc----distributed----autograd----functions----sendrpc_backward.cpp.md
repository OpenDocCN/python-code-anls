# `.\pytorch\torch\csrc\distributed\autograd\functions\sendrpc_backward.cpp`

```py
// 引入 sendrpc_backward.h 头文件，包含了 SendRpcBackward 类的声明
#include <torch/csrc/distributed/autograd/functions/sendrpc_backward.h>

// 命名空间 torch 开始
namespace torch {
// 命名空间 distributed 开始
namespace distributed {
// 命名空间 autograd 开始
namespace autograd {

// SendRpcBackward 类的成员函数 apply 的实现，接受一个右值引用的变量列表 inputs
torch::autograd::variable_list SendRpcBackward::apply(
    torch::autograd::variable_list&& inputs) {
  // 断言 inputs 应为空，否则输出错误信息 "SendRpcBackward should receive no inputs"
  TORCH_INTERNAL_ASSERT(
      inputs.empty(), "SendRpcBackward should receive no inputs");

  // 检查每个梯度变量是否已定义，如果未定义则输出错误信息 "BUG!: SendRpcBackward didn't receive valid gradients"
  for (const auto& grad : grads_) {
    TORCH_INTERNAL_ASSERT(
        grad.defined(), "BUG!: SendRpcBackward didn't receive valid gradients");
  }

  // 返回移动语义的 grads_ 变量列表，即将其所有权移交给调用者
  return std::move(grads_);
}

// 设置 grads_ 成员变量的值
void SendRpcBackward::setGrads(const torch::autograd::variable_list& grads) {
  grads_ = grads;
}

// 获取 grads_ 成员变量的值的常引用
const torch::autograd::variable_list& SendRpcBackward::getGrads() const {
  return grads_;
}

} // namespace autograd
} // namespace distributed
} // namespace torch
// 命名空间 torch 结束
```