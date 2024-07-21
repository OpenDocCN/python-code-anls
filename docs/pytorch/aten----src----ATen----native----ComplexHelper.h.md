# `.\pytorch\aten\src\ATen\native\ComplexHelper.h`

```
#pragma once

#include <ATen/core/Tensor.h>
#include <c10/util/irange.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/view_as_real_native.h>
#include <ATen/ops/view_as_complex_native.h>

#include <utility>
#endif

// WARNING: this header contains non-inline functions and should be only
// included from ONE cpp file

namespace at::native {

// View tensor with new dtype, storage offset, sizes and strides
inline Tensor view_tensor(
    const Tensor &tensor, ScalarType dtype,
    c10::SymInt offset, SymIntArrayRef sizes, SymIntArrayRef strides) {
  // 获取输入张量的存储
  Storage storage = tensor.storage();
  // 移除Conjugate标记，复制关键集合
  auto key_set = tensor.key_set().remove(DispatchKey::Conjugate);
  // 创建一个新的张量
  auto new_tensor = detail::make_tensor<TensorImpl>(
      c10::TensorImpl::VIEW, std::move(storage), key_set, scalarTypeToTypeMeta(dtype));
  // 获取新张量的实现指针
  auto * impl = new_tensor.unsafeGetTensorImpl();
  // 设置新张量的大小、步长和偏移量
  impl->set_sizes_and_strides(sizes, strides, offset);
  // 返回新张量
  return new_tensor;
}

// 计算用于view_as_real的步长
inline SymDimVector computeStrideForViewAsReal(SymIntArrayRef oldstride) {
  // 创建一个新的步长向量，比原来的向量多一维
  SymDimVector res(oldstride.size() + 1);
  // 根据原步长计算新的步长向量
  for (const auto i : c10::irange(oldstride.size())) {
    res[i] = oldstride[i] * 2;
  }
  // 最后一维步长为1
  res.back() = 1;
  return res;
}

// 将复数张量视为实数张量的物理实现
inline Tensor _view_as_real_physical(const Tensor& self) {
  // 检查输入张量是否为复数张量
  TORCH_CHECK(self.is_complex(), "view_as_real is only supported for complex tensors");
  // 获取输入张量的大小
  auto old_sizes = self.sym_sizes();
  // 创建一个新的大小向量，比原来的向量多一维
  SymDimVector new_sizes(old_sizes.size() + 1);
  std::copy(old_sizes.begin(), old_sizes.end(), new_sizes.begin());
  // 最后一维包含实部和虚部两个值
  new_sizes.back() = 2;
  // 计算新的步长向量
  auto new_strides = computeStrideForViewAsReal(self.sym_strides());
  // 计算新的存储偏移量
  auto new_storage_offset = self.sym_storage_offset() * 2;
  // 将输入张量视为实数张量并返回
  const auto float_type = c10::toRealValueType(self.scalar_type());
  auto real_tensor = view_tensor(self, float_type, std::move(new_storage_offset), new_sizes, new_strides);
  return real_tensor;
}

// 期望输入为复数张量，并返回一个相应的实数dtype的张量，
// 最后两个维度包含复数值
Tensor view_as_real(const Tensor& self) {
  // 检查张量是否有共轭标记
  TORCH_CHECK(!self.is_conj(), "view_as_real doesn't work on unresolved conjugated tensors.  To resolve the conjugate tensor so you can view it as real, use self.resolve_conj(); however, be warned that the resulting tensor will NOT alias the original.");
  // 调用_view_as_real_physical进行实际的视图操作
  return _view_as_real_physical(self);
}

// 计算用于view_as_complex的步长
inline SymDimVector computeStrideForViewAsComplex(SymIntArrayRef oldstride) {
  // 获取维度数
  const int64_t dim = oldstride.size();
  // 检查最后一维是否为步长1
  TORCH_CHECK(oldstride[dim-1] == 1, "Tensor must have a last dimension with stride 1");
  
  // 创建一个新的步长向量，少了一维
  SymDimVector res(dim - 1);
  // 对于除最后一维以外的所有维度，检查步长是否能被2整除，然后计算新的步长
  for (const auto i : c10::irange(res.size())) {
    TORCH_CHECK(oldstride[i] % 2 == 0, "Tensor must have a stride divisible by 2 for all but last dimension");
    res[i] = oldstride[i] / 2;
  }
  return res;
}
// 返回一个与原张量相应复数类型的张量
Tensor view_as_complex(const Tensor& self) {
  // 检查张量的标量类型是否为 float、double 或 half
  TORCH_CHECK(
    self.scalar_type() == kFloat || self.scalar_type() == kDouble || self.scalar_type() == kHalf,
    "view_as_complex 仅支持 half、float 和 double 类型的张量，但当前张量类型为: ", self.scalar_type());

  // 获取张量的旧尺寸
  auto old_sizes = self.sym_sizes();
  // 检查张量是否至少有一个维度
  TORCH_CHECK(!old_sizes.empty(), "输入张量必须至少有一个维度");
  // 检查张量的最后一个维度是否为2
  TORCH_CHECK(old_sizes[old_sizes.size()-1] == 2, "张量的最后一个维度必须为2");
  // 创建新尺寸，不包括最后一个维度
  SymDimVector new_sizes(old_sizes.begin(), old_sizes.end() - 1);

  // 根据当前的符号步长计算新的步长用于视图转换为复数
  const auto new_strides = computeStrideForViewAsComplex(self.sym_strides());
  // 获取复数类型对应的数据类型
  const auto complex_type = c10::toComplexType(self.scalar_type());

  // 检查张量的存储偏移是否为2的倍数
  TORCH_CHECK(self.sym_storage_offset() % 2 == 0, "张量的存储偏移必须是2的倍数");
  // 计算新的存储偏移，使其除以2
  const auto new_storage_offset = self.sym_storage_offset() / 2;

  // 返回基于视图转换的新张量
  return view_tensor(self, complex_type, new_storage_offset, new_sizes, new_strides);
}
```