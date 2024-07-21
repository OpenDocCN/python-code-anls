# `.\pytorch\aten\src\ATen\native\VariableMethodStubs.cpp`

```py
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_backward_native.h>
#include <ATen/ops/_fw_primal_native.h>
#include <ATen/ops/_version_native.h>
#include <ATen/ops/alias.h>
#include <ATen/ops/data_native.h>
#include <ATen/ops/is_leaf_native.h>
#include <ATen/ops/output_nr_native.h>
#include <ATen/ops/requires_grad_native.h>
#include <ATen/ops/retain_grad_native.h>
#include <ATen/ops/retains_grad_native.h>
#include <ATen/ops/set_data_native.h>
#include <ATen/ops/zeros_like_ops.h>
#endif

// The stubs in here are used by dynamic dispatch. It just redirects everything
// to the Tensor method we manually bind in TensorBody.h.

namespace at::native {

// 实现了 _backward 方法，用于张量的反向传播
void _backward(const Tensor& self, TensorList inputs, const std::optional<Tensor>& gradient_opt, std::optional<bool> keep_graph, bool create_graph) {
  return self._backward(inputs, gradient_opt, keep_graph, create_graph);
}

// 实现了 set_data 方法，用于设置张量的数据
void set_data(Tensor& self, const Tensor& new_data) {
  return self.set_data(new_data);
}

// 实现了 data 方法，用于获取张量的数据
Tensor data(const Tensor& self) {
  return self.data();
}

// 实现了 is_leaf 方法，用于检查张量是否为叶子节点
bool is_leaf(const Tensor& self) {
  return self.is_leaf();
}

// 实现了 output_nr 方法，用于获取张量的输出编号
int64_t output_nr(const Tensor& self) {
  return self.output_nr();
}

// 实现了 _version 方法，用于获取张量的版本号
int64_t _version(const Tensor& self) {
  return self._version();
}

// 实现了 requires_grad_ 方法，用于设置张量是否需要梯度
Tensor& requires_grad_(Tensor& self, bool _requires_grad) {
  self.requires_grad_(_requires_grad);
  return self;
}

// 实现了 retain_grad 方法，用于保留张量的梯度
void retain_grad(Tensor& self) {
  return self.retain_grad();
}

// 实现了 retains_grad 方法，用于检查张量是否保留梯度
bool retains_grad(const Tensor& self) {
  return self.retains_grad();
}

// 下面的函数用于前向传播原语（forward primal），预期仅在推理模式下以及所有输入都是推理张量时调用
// 返回张量的别名，仅在满足特定条件时调用，否则抛出错误信息
Tensor _fw_primal(const Tensor& self, int64_t level) {
  TORCH_INTERNAL_ASSERT(
    InferenceMode::is_enabled() && self.is_inference(),
    "Expected this method to only be reached in inference mode and when all the "
    "inputs are inference tensors. You should NOT call this method directly as "
    "native::_fw_primal. Please use the dispatcher, i.e., at::_fw_primal. Please "
    "file an issue if you come across this error otherwise.");
  return at::alias(self);
}

} // namespace at::native
```