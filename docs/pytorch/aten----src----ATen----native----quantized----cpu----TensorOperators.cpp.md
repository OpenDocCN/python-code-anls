# `.\pytorch\aten\src\ATen\native\quantized\cpu\TensorOperators.cpp`

```
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
// 定义宏，用于限定仅使用方法操作符

#include <ATen/core/Tensor.h>
#include <ATen/ExpandUtils.h>
#include <ATen/native/Resize.h>
#include <ATen/quantized/Quantizer.h>
#include <c10/core/QScheme.h>
// 引入所需的头文件，包括张量操作、扩展工具、调整大小、量化器和量化方案

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/eq.h>
#include <ATen/ops/eq_native.h>
#include <ATen/ops/ge.h>
#include <ATen/ops/ge_native.h>
#include <ATen/ops/gt.h>
#include <ATen/ops/gt_native.h>
#include <ATen/ops/le.h>
#include <ATen/ops/le_native.h>
#include <ATen/ops/lt.h>
#include <ATen/ops/lt_native.h>
#include <ATen/ops/ne.h>
#include <ATen/ops/ne_native.h>
#include <ATen/ops/resize_native.h>
#endif
// 根据条件引入不同的运算符头文件

namespace at {
namespace native {

/*
All comparator operators will be named "<aten op name>_quantized_cpu".
'_out' will be appended for the 'out' variant of the op.

TODO: This is an inefficient implementation that uses `.dequantize`.
      Need a more efficient implementation.
*/
// 定义量化 CPU 下的比较操作符及其输出变体的命名规则和实现方法

#define DEFINE_COMPARATOR(at_op) \
Tensor& at_op##_out_quantized_cpu(const Tensor& self, \
                                const Scalar& other, Tensor& out) { \
  TORCH_CHECK(out.dtype() == at::ScalarType::Bool, \
              "The 'out' tensor must have dtype 'torch.bool'"); \
  auto self_dq = self.dequantize(); \
  return at:: at_op##_out(out, self_dq, other); \
} \
Tensor at_op##_quantized_cpu(const Tensor& self, const Scalar& other) { \
  auto self_dq = self.dequantize(); \
  return at:: at_op(self_dq, other); \
} \
Tensor& at_op##_out_quantized_cpu(const Tensor& self, \
                                const Tensor& other, Tensor& out) { \
  /* We infer size to make sure the tensors are compatible. */\
  infer_size_dimvector(self.sizes(), other.sizes()); \
  TORCH_CHECK(out.dtype() == at::ScalarType::Bool, \
              "The 'out' tensor must have dtype 'torch.bool'"); \
  auto self_dq = self.dequantize(); \
  auto other_dq = other.dequantize(); \
  return at:: at_op##_out(out, self_dq, other_dq); \
} \
Tensor at_op##_quantized_cpu(const Tensor& self, const Tensor& other) { \
  /* We infer size to make sure the tensors are compatible. */\
  infer_size_dimvector(self.sizes(), other.sizes()); \
  auto self_dq = self.dequantize(); \
  auto other_dq = other.dequantize(); \
  return at:: at_op(self_dq, other_dq); \
}
// 定义宏，用于生成量化 CPU 下的比较操作符和输出变体的实现函数

#define AT_FORALL_OPERATORS(_) \
_(ne)                          \
_(eq)                          \
_(ge)                          \
_(le)                          \
_(gt)                          \
_(lt)                          \

AT_FORALL_OPERATORS(DEFINE_COMPARATOR)
// 对所有的比较操作符应用宏，生成相应的量化 CPU 下的操作函数

#undef AT_FORALL_OPERATORS
#undef DEFINE_COMPARATOR

const Tensor& quantized_resize_cpu_(
    const Tensor& self,
    IntArrayRef size,
  // 根据“Writing Nondeterministic Operations”注释
  // 这是一个非确定性操作，因为如果存储被重新调整大小，新元素将未初始化
  globalContext().alertNotDeterministic("quantized_resize_cpu_");

  // 检查是否指定了内存格式，如果有，抛出错误信息
  TORCH_CHECK(
      !optional_memory_format.has_value(),
      "Unsupported memory format for quantized tensor resize ",
      optional_memory_format.value());

  // 获取当前量化器的量化模式
  auto qscheme = self.quantizer()->qscheme();

  // 检查量化模式是否为每张量仿射或每张量对称，否则抛出错误信息
  TORCH_CHECK(
      qscheme == QScheme::PER_TENSOR_AFFINE ||
          qscheme == QScheme::PER_TENSOR_SYMMETRIC,
      "Can only resize quantized tensors with per-tensor schemes!");

  // 获取当前张量的不安全实现指针
  auto* self_ = self.unsafeGetTensorImpl();

  // 调用 CPU 上的 resize_impl_cpu_ 函数来实现张量的调整大小
  // NOLINTNEXTLINE(bugprone-argument-comment)
  resize_impl_cpu_(self_, size, /*strides=*/c10::nullopt);

  // 返回调整大小后的张量对象
  return self;
}

}}  // at::native
```