# `.\pytorch\aten\src\ATen\native\GatedLinearUnit.cpp`

```
// 定义 TORCH_ASSERT_ONLY_METHOD_OPERATORS 宏，用于包含必要的头文件
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>  // 引入张量的核心功能头文件
#include <ATen/TensorIterator.h>  // 引入张量迭代器头文件
#include <ATen/TensorOperators.h>  // 引入张量操作符头文件
#include <ATen/native/Activation.h>  // 引入原生激活函数头文件

// 根据条件选择是否包含 AT_PER_OPERATOR_HEADERS 宏定义的头文件
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/cat.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/glu_backward_native.h>
#include <ATen/ops/glu_backward_jvp_native.h>
#include <ATen/ops/glu_jvp_native.h>
#include <ATen/ops/glu_native.h>
#include <ATen/ops/sigmoid.h>
#endif

namespace at::meta {

// 定义 TORCH_META_FUNC 宏，用于声明 glu 函数的元数据
TORCH_META_FUNC(glu) (
    const Tensor& self, int64_t dim
) {
  // 检查张量维度是否大于0，否则输出错误信息
  TORCH_CHECK(self.dim() > 0, "glu does not support 0-dimensional tensors");
  // 获取包装后的维度索引，确保在有效范围内
  auto wrap_dim = maybe_wrap_dim(dim, self.dim());
  // 获取指定维度上的尺寸
  const int64_t nIn = self.size(wrap_dim);
  // 检查尺寸是否为偶数，否则输出错误信息
  TORCH_CHECK(nIn % 2 == 0, "Halving dimension must be even, but dimension ",
              wrap_dim, " is size ", nIn);

  // 计算输出张量的大小为输入的一半
  const int64_t selfSize = nIn / 2;
  // 获取输入张量的前半部分和后半部分
  Tensor firstHalf = self.narrow(wrap_dim, 0, selfSize);
  Tensor secondHalf = self.narrow(wrap_dim, selfSize, selfSize);
  // 构建使用二进制操作的函数
  build_borrowing_binary_op(maybe_get_output(), firstHalf, secondHalf);
}
} // namespace at::meta

namespace at::native {

// 定义 DISPATCH 宏，用于声明 glu_stub 函数的分发机制
DEFINE_DISPATCH(glu_stub);
// 定义 DISPATCH 宏，用于声明 glu_backward_stub 函数的分发机制
DEFINE_DISPATCH(glu_backward_stub);
// 定义 DISPATCH 宏，用于声明 glu_jvp_stub 函数的分发机制
DEFINE_DISPATCH(glu_jvp_stub);

// 定义 TORCH_IMPL_FUNC 宏，实现 glu_out 函数，接收输入张量和输出张量
TORCH_IMPL_FUNC(glu_out) (const Tensor& self, int64_t dim, const Tensor& out) {
  // 调用 glu_stub 分发函数处理设备类型和当前实例
  glu_stub(device_type(), *this);
}

// 定义 glu_backward_cpu_out 函数，计算梯度的反向传播
Tensor& glu_backward_cpu_out(const Tensor& grad_output, const Tensor& input,
                             int64_t dim, Tensor& grad_input) {
  // 检查输入张量的维度是否大于0，否则输出错误信息
  TORCH_CHECK(input.dim() > 0, "glu does not support 0-dimensional tensors");
  // 获取包装后的维度索引，确保在有效范围内
  auto wrap_dim = maybe_wrap_dim(dim, input.dim());
  // 获取指定维度上的尺寸
  const int64_t nIn = input.size(wrap_dim);
  // 检查尺寸是否为偶数，否则输出错误信息
  TORCH_CHECK(nIn % 2 == 0, "Halving dimension must be even, but dimension ",
              wrap_dim, " is size ", nIn);

  // 调整梯度输入张量的大小与输入张量相同
  grad_input.resize_as_(input);
  // 计算输入张量的大小为一半
  const int64_t inputSize = nIn / 2;
  // 获取输入张量的前半部分和后半部分
  Tensor firstHalf = input.narrow(wrap_dim, 0, inputSize);
  Tensor secondHalf = input.narrow(wrap_dim, inputSize, inputSize);
  // 获取梯度输入张量的前半部分和后半部分
  Tensor gradInputfirstHalf = grad_input.narrow(wrap_dim, 0, inputSize);
  Tensor gradInputsecondHalf = grad_input.narrow(wrap_dim, inputSize, inputSize);

  // 对第一半部分进行 sigmoid 操作
  at::sigmoid_out(gradInputfirstHalf, secondHalf);
  // 通过张量迭代器配置优化第二个梯度输入部分
  auto iter = at::TensorIteratorConfig()
    .add_output(gradInputsecondHalf)
    .add_const_input(gradInputfirstHalf)
    .add_const_input(firstHalf)
    .add_const_input(grad_output)
    .build();
    # 调用某个对象的build方法，执行特定的构建操作

  glu_backward_stub(iter.device_type(), iter);
  # 调用glu_backward_stub函数，传入迭代器的设备类型和迭代器本身作为参数

  gradInputfirstHalf.mul_(grad_output);
  # 使用grad_output乘以gradInputfirstHalf的前半部分，结果存回gradInputfirstHalf

  return grad_input;
  # 返回变量grad_input作为函数的输出结果
} // namespace at::native
```