# `.\pytorch\aten\src\ATen\native\PointwiseOps.cpp`

```
// 定义宏，用于在编译时仅包含方法运算符
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
// 包含点对点操作的头文件
#include <ATen/native/PointwiseOps.h>

// 包含张量核心和元数据的头文件
#include <ATen/core/Tensor.h>
#include <ATen/TensorMeta.h>

// 如果未定义 AT_PER_OPERATOR_HEADERS，则包含通用函数头文件，否则分别包含特定的加法、除法通用函数头文件
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/addcdiv_native.h>
#include <ATen/ops/addcmul_native.h>
#endif

// at 命名空间下的 meta 命名空间
namespace at::meta {

// 定义函数 addcmul 的元数据函数
TORCH_META_FUNC(addcmul)
(const Tensor& self,
 const Tensor& tensor1,
 const Tensor& tensor2,
 const Scalar& value) {
  // 构建三元操作，可能获取输出
  build_ternary_op(maybe_get_output(), self, tensor1, tensor2);
}

// 定义函数 addcdiv 的元数据函数
TORCH_META_FUNC(addcdiv)
(const Tensor& self,
 const Tensor& tensor1,
 const Tensor& tensor2,
 const Scalar& value) {
  // 如果 tensor1 和 tensor2 的数据类型为整数类型（包括布尔类型）
  if (isIntegralType(tensor1.scalar_type(), /*includeBool=*/true) &&
      isIntegralType(tensor2.scalar_type(), /*includeBool=*/true)) {
    // 抛出错误，不再支持使用 addcdiv 进行整数除法，未来版本将对 tensor1 和 tensor2 执行真正的除法
    TORCH_CHECK(
        false,
        "Integer division with addcdiv is no longer supported, and in a future  ",
        "release addcdiv will perform a true division of tensor1 and tensor2. ",
        "The historic addcdiv behavior can be implemented as ",
        "(input + value * torch.trunc(tensor1 / tensor2)).to(input.dtype) ",
        "for integer inputs and as ",
        "(input + value * tensor1 / tensor2) for float inputs. ",
        "The future addcdiv behavior is just the latter implementation: ",
        "(input + value * tensor1 / tensor2), for all dtypes.");
  }
  // 构建三元操作，可能获取输出
  build_ternary_op(maybe_get_output(), self, tensor1, tensor2);
}

} // namespace at::meta

// at 命名空间下的 native 命名空间
namespace at::native {

// 定义函数 addcmul_out 的实现函数
TORCH_IMPL_FUNC(addcmul_out)
(const Tensor& self,
 const Tensor& tensor1,
 const Tensor& tensor2,
 const Scalar& value,
 const Tensor& result) {
  // 调用 addcmul_stub，根据设备类型和传入参数执行具体的加乘运算
  addcmul_stub(device_type(), *this, value);
}

// 定义函数 addcdiv_out 的实现函数
TORCH_IMPL_FUNC(addcdiv_out)
(const Tensor& self,
 const Tensor& tensor1,
 const Tensor& tensor2,
 const Scalar& value,
 const Tensor& result) {
  // 调用 addcdiv_stub，根据设备类型和传入参数执行具体的加除运算
  addcdiv_stub(device_type(), *this, value);
}

// 定义 addcmul_stub 的分发函数
DEFINE_DISPATCH(addcmul_stub);
// 定义 addcdiv_stub 的分发函数
DEFINE_DISPATCH(addcdiv_stub);

} // namespace at::native
```