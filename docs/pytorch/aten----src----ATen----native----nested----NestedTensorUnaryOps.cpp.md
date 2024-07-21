# `.\pytorch\aten\src\ATen\native\nested\NestedTensorUnaryOps.cpp`

```
// 包含头文件，用于NestedTensorMath的函数实现
#include <ATen/native/nested/NestedTensorMath.h>

// 包含各种必要的ATen头文件
#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#include <ATen/NestedTensorImpl.h>
#include <ATen/ScalarOps.h>
#include <ATen/TensorIndexing.h>
#include <ATen/TensorOperators.h>
#include <ATen/TensorUtils.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/layer_norm.h>
#include <ATen/native/nested/NestedTensorUtils.h>

// 使用at命名空间
namespace at {
namespace native {

// 返回一个NestedTensor，其元素为输入Tensor每个元素的绝对值
Tensor NestedTensor_abs(const Tensor& self) {
  return map_nt(self, at::abs);
}

// 原地计算输入NestedTensor每个元素的绝对值
Tensor& NestedTensor_abs_(Tensor& self) {
  auto self_ptr = get_nested_tensor_impl(self);
  check_numel_equals_buffer_size(self_ptr);
  auto buffer = self_ptr->get_buffer();
  at::abs_(buffer);
  return self;
}

// 返回一个NestedTensor，其元素为输入Tensor每个元素的符号函数值
Tensor NestedTensor_sgn(const Tensor& self) {
  return map_nt(self, at::sgn);
}

// 原地计算输入NestedTensor每个元素的符号函数值
Tensor& NestedTensor_sgn_(Tensor& self) {
  auto self_ptr = get_nested_tensor_impl(self);
  check_numel_equals_buffer_size(self_ptr);
  auto buffer = self_ptr->get_buffer();
  buffer.sgn_();
  return self;
}

// 原地计算输入NestedTensor每个元素的逻辑非（取反）
Tensor& NestedTensor_logical_not_(Tensor& self){
  auto self_ptr = get_nested_tensor_impl(self);
  check_numel_equals_buffer_size(self_ptr);
  auto buffer = self_ptr->get_buffer();
  buffer.logical_not_();
  return self;
}

// 返回一个NestedTensor，其元素为输入Tensor每个元素的逻辑非值
Tensor NestedTensor_logical_not(const Tensor& self) {
  return map_nt(self, at::logical_not);
}

// 原地计算输入NestedTensor每个元素的ReLU函数值
Tensor& NestedTensor_relu_(Tensor& self) {
  auto self_ptr = get_nested_tensor_impl(self);
  check_numel_equals_buffer_size(self_ptr);
  auto buffer = self_ptr->get_buffer();
  at::relu_(buffer);
  return self;
}

// 返回一个NestedTensor，其元素为输入Tensor每个元素的ReLU函数值
Tensor NestedTensor_relu(const Tensor& self) {
  return map_nt(self, at::relu);
}

// 原地计算输入NestedTensor每个元素的GELU函数值，支持近似计算
Tensor& NestedTensor_gelu_(Tensor& self, c10::string_view approximate) {
  auto self_ptr = get_nested_tensor_impl(self);
  check_numel_equals_buffer_size(self_ptr);
  auto buffer = self_ptr->get_buffer();
  at::gelu_(buffer, approximate);
  return self;
}

// 返回一个NestedTensor，其元素为输入Tensor每个元素的GELU函数值，支持近似计算
Tensor NestedTensor_gelu(const Tensor& self, c10::string_view approximate) {
  return map_nt(
      self,
      [approximate](const Tensor& buffer) {
        return at::gelu(buffer, approximate);
      });
}

// 原地计算输入NestedTensor每个元素的双曲正切函数值
Tensor& NestedTensor_tanh_(Tensor& self) {
  auto self_ptr = get_nested_tensor_impl(self);
  check_numel_equals_buffer_size(self_ptr);
  auto buffer = self_ptr->get_buffer();
  at::tanh_(buffer);
  return self;
}

// 返回一个NestedTensor，其元素为输入Tensor每个元素的双曲正切函数值
Tensor NestedTensor_tanh(const Tensor& self) {
  return map_nt(self, at::tanh);
}

// 原地计算输入NestedTensor每个元素的负数
Tensor& NestedTensor_neg_(Tensor& self) {
  auto self_ptr = get_nested_tensor_impl(self);
  check_numel_equals_buffer_size(self_ptr);
  auto buffer = self_ptr->get_buffer();
  at::neg_(buffer);
  return self;
}

// 返回一个NestedTensor，其元素为输入Tensor每个元素的负数
Tensor NestedTensor_neg(const Tensor& self) {
  return map_nt(self, at::neg);
}

// 原地将输入NestedTensor的所有元素置为0
Tensor& zero_nested_(Tensor& self) {
  const auto& self_buf = get_nested_tensor_impl(self)->get_buffer();
  self_buf.fill_(0);
  return self;
}

// 返回一个NestedTensor，其元素为输入Tensor每个元素的SiLU函数值
Tensor NestedTensor_silu(const Tensor& self){
  return map_nt(self, at::silu);
}

// 结束native命名空间和at命名空间
} // namespace native
} // namespace at
// 计算 NestedTensor 中每个元素的自适应线性单元激活（SiLU）函数，并返回自身的引用
Tensor& NestedTensor_silu_(Tensor& self){
  // 获取 NestedTensor 的实现指针
  auto self_ptr = get_nested_tensor_impl(self);
  // 检查 NestedTensor 的元素数量是否等于其缓冲区大小
  check_numel_equals_buffer_size(self_ptr);
  // 获取 NestedTensor 的缓冲区
  auto buffer = self_ptr->get_buffer();
  // 在缓冲区上应用自适应线性单元激活函数（SiLU）
  at::silu_(buffer);
  // 返回自身的引用
  return self;
}

// 对 NestedTensor 中每个元素应用正弦函数，并返回结果
Tensor sin_nested(const Tensor& self) {
  return map_nt(self, at::sin);
}

// 对 NestedTensor 中每个元素应用余弦函数，并返回结果
Tensor cos_nested(const Tensor& self) {
  return map_nt(self, at::cos);
}

// 将输入的 NestedTensor 数据钉入内存，可选指定设备
Tensor _pin_memory_nested(const Tensor& self, std::optional<Device> device) {
  // 获取 NestedTensor 的实现指针
  auto* nt_input = get_nested_tensor_impl(self);
  // 获取 NestedTensor 内部不安全的存储作为张量
  const auto& input_buffer = nt_input->get_unsafe_storage_as_tensor();
  // 将内部缓冲区数据钉入内存，可选指定设备
  return wrap_buffer(
      at::_pin_memory(input_buffer, device),
      nt_input->get_nested_sizes(),
      nt_input->get_nested_strides(),
      nt_input->get_storage_offsets());
}

} // namespace native
} // namespace at
```