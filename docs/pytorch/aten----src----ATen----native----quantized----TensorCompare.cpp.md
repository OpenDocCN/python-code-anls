# `.\pytorch\aten\src\ATen\native\quantized\TensorCompare.cpp`

```
// 包含 ATen 库中的各种头文件，用于张量操作和计算
#include <ATen/ATen.h>
#include <ATen/CPUApplyUtils.h>
#include <ATen/ExpandUtils.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <ATen/native/TensorCompare.h>
#include <c10/util/Exception.h>

// 将代码置于 ATen 的命名空间内
namespace at {
namespace native {

// 在 CPU 上实现的量化最大值计算函数，接受一个张量作为输入，返回一个张量
Tensor max_quantized_cpu(const Tensor& self) {
  // 调用张量的 reshape 方法，将其视图转换为一维，计算最大值并返回
  return std::get<0>(self.reshape({-1}).max(/*dim=*/0));
}

// 在指定输出张量上计算量化最大值的函数，接受输入张量和输出张量引用作为参数，返回输出张量引用
Tensor& max_quantized_unary_out(const Tensor& self, Tensor& out) {
  // 检查输入张量和输出张量是否在相同的设备上
  TORCH_CHECK(self.device() == out.device());

  // 检查数据类型是否可以转换，以确保计算的正确性
  TORCH_CHECK(canCast(
      typeMetaToScalarType(self.dtype()),
      typeMetaToScalarType(out.dtype())));
  
  // 调用 max_quantized_cpu 函数计算最大值，并将结果复制到输出张量上
  Tensor temp = max_quantized_cpu(self);
  at::native::resize_output(out, temp.sizes());
  out.copy_(temp);
  
  // 返回输出张量的引用
  return out;
}

// 在 CPU 上实现的量化最小值计算函数，接受一个张量作为输入，返回一个张量
Tensor min_quantized_cpu(const Tensor& self) {
  // 调用张量的 reshape 方法，将其视图转换为一维，计算最小值并返回
  return std::get<0>(self.reshape({-1}).min(/*dim=*/0));
}

// 在指定输出张量上计算量化最小值的函数，接受输入张量和输出张量引用作为参数，返回输出张量引用
Tensor& min_quantized_unary_out(const Tensor& self, Tensor& out) {
  // 检查输入张量和输出张量是否在相同的设备上
  TORCH_CHECK(self.device() == out.device());

  // 检查数据类型是否可以转换，以确保计算的正确性
  TORCH_CHECK(canCast(
      typeMetaToScalarType(self.dtype()),
      typeMetaToScalarType(out.dtype())));
  
  // 调用 min_quantized_cpu 函数计算最小值，并将结果复制到输出张量上
  Tensor temp = min_quantized_cpu(self);
  at::native::resize_output(out, temp.sizes());
  out.copy_(temp);
  
  // 返回输出张量的引用
  return out;
}

// TODO: move to TensorMath.cpp
// 在 CPU 上实现的稳定排序函数，接受一个张量和一些可选参数，返回排序后的张量和索引张量的元组
std::tuple<Tensor, Tensor> sort_quantized_cpu_stable(
    const Tensor& self,
    std::optional<bool> stable,
    int64_t dim,
    bool descending) {
  // 调用 ATen 库中的 sort 函数，对张量的整数表示进行排序，并返回排序后的结果和索引
  auto [sort_int, sort_indices] =
      at::sort(self.int_repr(), stable, dim, descending);
  
  // 使用 _make_per_tensor_quantized_tensor 函数创建一个新的量化张量，基于排序后的整数张量及原始张量的量化参数
  return std::forward_as_tuple(
      at::_make_per_tensor_quantized_tensor(
          sort_int, self.q_scale(), self.q_zero_point()),
      sort_indices);
}

} // namespace native
} // namespace at
```