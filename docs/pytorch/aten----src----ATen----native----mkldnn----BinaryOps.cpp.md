# `.\pytorch\aten\src\ATen\native\mkldnn\BinaryOps.cpp`

```
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
// 定义宏，用于指定仅包含方法运算符

#include <ATen/core/Tensor.h>
#include <ATen/Config.h>
#include <ATen/ExpandUtils.h>
// 引入 ATen 库的头文件

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/add_native.h>
#include <ATen/ops/empty_native.h>
#include <ATen/ops/mul_native.h>
#endif
// 根据编译选项选择引入特定的 ATen 头文件

#if !AT_MKLDNN_ENABLED()
// 如果未启用 MKLDNN 加速

namespace at {
namespace native {

Tensor& mkldnn_add_out(
    const Tensor& self,
    const Tensor& other,
    const Scalar& alpha,
    Tensor& result
    ) {
  TORCH_CHECK(false, "mkldnn_add_out: ATen not compiled with MKLDNN support");
  // 抛出错误，表示 ATen 未编译支持 MKLDNN
}

Tensor mkldnn_add(const Tensor& self, const Tensor& other, const Scalar& alpha) {
  TORCH_CHECK(false, "mkldnn_add: ATen not compiled with MKLDNN support");
  // 抛出错误，表示 ATen 未编译支持 MKLDNN
}

Tensor& mkldnn_add_(Tensor& self, const Tensor& other, const Scalar& alpha) {
  TORCH_CHECK(false, "mkldnn_add_: ATen not compiled with MKLDNN support");
  // 抛出错误，表示 ATen 未编译支持 MKLDNN
}

Tensor& mkldnn_mul_out(const Tensor& self, const Tensor& other, Tensor& result) {
  TORCH_CHECK(false, "mkldnn_mul_out: ATen not compiled with MKLDNN support");
  // 抛出错误，表示 ATen 未编译支持 MKLDNN
}

Tensor mkldnn_mul(const Tensor& self, const Tensor& other) {
  TORCH_CHECK(false, "mkldnn_mul: ATen not compiled with MKLDNN support");
  // 抛出错误，表示 ATen 未编译支持 MKLDNN
}

Tensor& mkldnn_mul_(Tensor& self, const Tensor& other) {
  TORCH_CHECK(false, "mkldnn_mul_: ATen not compiled with MKLDNN support");
  // 抛出错误，表示 ATen 未编译支持 MKLDNN
}

} // namespace native
} // namespace at

#else // AT_MKLDNN_ENABLED

#include <ATen/native/mkldnn/MKLDNNCommon.h>
// 引入 MKLDNN 相关的头文件

namespace at {
namespace native {

static Tensor emptyBinaryOp(const Tensor& self, const Tensor& other) {
  if (!self.requires_grad() && !other.requires_grad()) {
    auto out_size = infer_size(self.sizes(), other.sizes());
    auto out_dtype = promoteTypes(
        c10::typeMetaToScalarType(self.dtype()),
        c10::typeMetaToScalarType(other.dtype()));
    TORCH_CHECK(
        self.device() == other.device(),
        "Expected same device for binary mkldnn op");
    return empty_mkldnn(
        out_size,
        out_dtype,
        self.options().layout_opt(),
        self.options().device_opt(),
        self.options().pinned_memory_opt());
    // 创建一个空的 MKLDNN 张量以进行二元操作
  } else {
    TORCH_CHECK(
        false,
        "MKLDNN does not support Binary Ops with a 0-dimension Tensor in training");
    // 抛出错误，表示在训练中 MKLDNN 不支持与零维张量的二元操作
  }
}

Tensor& mkldnn_add_out(
    const Tensor& self,
    const Tensor& other,
    const Scalar& alpha,
    Tensor& result
    ) {
  ideep::tensor& x = itensor_from_mkldnn(self);
  ideep::tensor& y = itensor_from_mkldnn(other);

  ideep::tensor& z = itensor_from_mkldnn(result);
  if (result.is_same(other)) {
    const std::vector<float> scales{alpha.to<float>(), 1.0};
    ideep::sum::compute(scales, {y, x}, z);
    // 如果结果张量与另一个操作数相同，则按比例对应相加
  } else {
    const std::vector<float> scales{1.0, alpha.to<float>()};
    ideep::sum::compute(scales, {x, y}, z);
    // 否则按比例对应相加
  }

  return result;
}

Tensor mkldnn_add(const Tensor& self, const Tensor& other, const Scalar& alpha) {
  if (self.numel() == 0 || other.numel() == 0) {
    // 如果任何一个操作数为空
    // 调用函数 `emptyBinaryOp`，传入 `self` 和 `other` 作为参数，并返回结果
    return emptyBinaryOp(self, other);
  }

  // 从 PyTorch Tensor `self` 创建对应的 MKL-DNN ideep::tensor `x`
  ideep::tensor& x = itensor_from_mkldnn(self);
  // 从 PyTorch Tensor `other` 创建对应的 MKL-DNN ideep::tensor `y`
  ideep::tensor& y = itensor_from_mkldnn(other);

  // 创建一个新的 MKL-DNN ideep::tensor `z` 作为计算结果的存储
  ideep::tensor z;
  // 定义包含两个浮点数的向量 `scales`，分别是 1.0 和 `alpha` 的浮点表示
  const std::vector<float> scales{1.0, alpha.to<float>()};
  // 调用 MKL-DNN 的 sum::compute 函数，使用指定的缩放因子 `scales`，对输入的 ideep::tensor `x` 和 `y` 进行求和，结果存储在 `z` 中
  ideep::sum::compute(scales, {x, y}, z);

  // 使用 MKL-DNN 的 ideep::tensor `z`，创建一个新的 PyTorch Tensor，并使用指定的类型和设备选项进行初始化
  return new_with_itensor_mkldnn(std::move(z), optTypeMetaToScalarType(self.options().dtype_opt()),
                                 self.options().device_opt());
}

Tensor& mkldnn_add_(Tensor& self, const Tensor& other, const Scalar& alpha) {
  return native::mkldnn_add_out(self, other, alpha, self);
}

Tensor& mkldnn_mul_out(const Tensor& self, const Tensor& other, Tensor& result) {
  // 检查输出张量的大小是否与输入张量相同
  TORCH_CHECK(result.sizes() == self.sizes(),
             "mkldnn_mul_out: the output size should be same as input size");
  // 将输出张量转换为 MKL-DNN 张量
  ideep::tensor& z = itensor_from_mkldnn(result);
  // 将输入张量转换为 MKL-DNN 张量
  ideep::tensor& x = itensor_from_mkldnn(self);

  // 处理零维张量的情况
  if (other.ndimension() == 0) {
    // 对零维张量执行元素级线性操作
    ideep::eltwise_forward::compute(
      x, z, ideep::algorithm::eltwise_linear,
      ideep::prop_kind::forward_inference, /*alpha*/ other.item().to<float>());

    return result;
  } else {
    // 检查输入张量的大小是否与其他张量相同
    TORCH_CHECK(self.sizes() == other.sizes(),
               "mkldnn_mul_out: currently mkldnn not support broadcasting");
    // 将其他张量转换为 MKL-DNN 张量
    ideep::tensor y = itensor_from_mkldnn(other);
    // 执行二进制乘法操作
    ideep::binary::compute(x, y, z, dnnl::algorithm::binary_mul);

    return result;
  }
}

Tensor mkldnn_mul(const Tensor& self, const Tensor& other) {
  // 如果任一输入张量的元素个数为0，则返回一个空的张量
  if (self.numel() == 0 || other.numel() == 0) {
    return emptyBinaryOp(self, other);
  }
  // 创建一个空的 MKL-DNN 张量，用于存储乘法操作的结果
  Tensor result = empty_mkldnn(self.sizes(), optTypeMetaToScalarType(self.options().dtype_opt()),
                               self.options().layout_opt(), self.options().device_opt(),
                               self.options().pinned_memory_opt());
  // 调用 mkldnn_mul_out 函数执行乘法操作，并返回结果张量
  return native::mkldnn_mul_out(self, other, result);
}

Tensor& mkldnn_mul_(Tensor& self, const Tensor& other) {
  // 调用 mkldnn_mul_out 函数执行乘法操作，结果保存在输入张量 self 中
  return native::mkldnn_mul_out(self, other, self);
}

} // namespace native
} // namespace at

#endif // AT_MKLDNN_ENABLED
```