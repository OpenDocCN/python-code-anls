# `.\pytorch\aten\src\ATen\native\ReduceAllOps.cpp`

```py
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/ReduceAllOps.h>
#include <ATen/native/Resize.h>

#include <ATen/core/Tensor.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_aminmax_native.h>
#include <ATen/ops/aminmax.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/max.h>
#include <ATen/ops/max_native.h>
#include <ATen/ops/min.h>
#include <ATen/ops/min_native.h>
#endif

namespace at::native {

// 定义分发函数的声明
DEFINE_DISPATCH(min_all_stub);
DEFINE_DISPATCH(max_all_stub);

// 计算张量的最小值
Tensor min(const Tensor &self) {
  // 检查张量元素数量大于零
  TORCH_CHECK(self.numel() > 0,
              "min(): Expected reduction dim to be specified for input.numel() == 0. Specify the reduction dim with the 'dim' argument.");
  // 创建一个与输入张量相同类型的空张量作为结果
  Tensor result = at::empty({}, self.options());
  // 调用最小值分发函数，计算张量的最小值并存储在结果张量中
  min_all_stub(self.device().type(), result, self.contiguous());
  // 返回计算得到的最小值张量
  return result;
}

// 计算张量的最小值并将结果存储在预先分配的输出张量中
Tensor& min_unary_out(const Tensor &self, Tensor& out) {
  // 首先检查设备是否匹配（CPU vs GPU）
  TORCH_CHECK(self.device() == out.device());

  // 检查是否可以进行类型转换
  TORCH_CHECK(canCast(
      typeMetaToScalarType(self.dtype()),
      typeMetaToScalarType(out.dtype())));

  // 调整输出张量的大小为标量
  at::native::resize_output(out, {});

  // 调用最小值分发函数，计算张量的最小值并存储在输出张量中
  min_all_stub(self.device().type(), out, self.contiguous());
  // 返回计算得到的最小值张量的引用
  return out;
}

// 计算张量的最大值
Tensor max(const Tensor &self) {
  // 检查张量元素数量大于零
  TORCH_CHECK(self.numel() > 0,
              "max(): Expected reduction dim to be specified for input.numel() == 0. Specify the reduction dim with the 'dim' argument.");
  // 创建一个与输入张量相同类型的空张量作为结果
  Tensor result = at::empty({}, self.options());
  // 调用最大值分发函数，计算张量的最大值并存储在结果张量中
  max_all_stub(self.device().type(), result, self.contiguous());
  // 返回计算得到的最大值张量
  return result;
}

// 计算张量的最大值并将结果存储在预先分配的输出张量中
Tensor& max_unary_out(const Tensor &self, Tensor& out) {
  // 首先检查设备是否匹配（CPU vs GPU）
  TORCH_CHECK(self.device() == out.device());

  // 检查是否可以进行类型转换
  TORCH_CHECK(canCast(
      typeMetaToScalarType(self.dtype()),
      typeMetaToScalarType(out.dtype())));

  // 调整输出张量的大小为标量
  at::native::resize_output(out, {});

  // 调用最大值分发函数，计算张量的最大值并存储在输出张量中
  max_all_stub(self.device().type(), out, self.contiguous());
  // 返回计算得到的最大值张量的引用
  return out;
}

// 弃用函数：使用 at::aminmax 替代
std::tuple<Tensor, Tensor> _aminmax_all(const Tensor &self) {
  // 发出一次性警告，提醒用户函数即将弃用
  TORCH_WARN_ONCE("_aminmax is deprecated as of PyTorch 1.11 and will be removed in a future release. Use aminmax instead."
                  " This warning will only appear once per process.");
  // 调用 at::aminmax 函数计算张量的最小值和最大值并返回结果
  return at::aminmax(self);
}

} // namespace at::native
```