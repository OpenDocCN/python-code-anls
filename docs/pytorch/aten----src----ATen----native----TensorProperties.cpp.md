# `.\pytorch\aten\src\ATen\native\TensorProperties.cpp`

```py
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
// 定义宏，仅在 Torch 中使用方法操作符

#include <ATen/core/Tensor.h>
#include <ATen/Context.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/detail/CUDAHooksInterface.h>
#include <ATen/native/TensorProperties.h>
// 包含 ATen 库的头文件，用于张量操作和属性

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_nested_tensor_size_native.h>
#include <ATen/ops/contiguous_native.h>
#include <ATen/ops/cudnn_is_acceptable_native.h>
#include <ATen/ops/detach_native.h>
#include <ATen/ops/equal.h>
#include <ATen/ops/is_same_size_native.h>
#include <ATen/ops/is_set_to_native.h>
#include <ATen/ops/size_native.h>
#include <ATen/ops/stride_native.h>
#include <ATen/ops/sym_numel_native.h>
#include <ATen/ops/sym_size_native.h>
#include <ATen/ops/sym_storage_offset_native.h>
#include <ATen/ops/sym_stride_native.h>
#endif
// 根据条件包含不同的 ATen 操作和函数头文件

#include <c10/util/irange.h>
// 包含 C10 库的范围工具

namespace at::native {

bool is_same_size(const Tensor& self, const Tensor& other) {
  return self.sym_sizes().equals(other.sym_sizes());
}
// 检查两个张量是否具有相同的符号大小

bool nested_is_same_size(const Tensor& self, const Tensor& other) {
  TORCH_CHECK(
      self.is_nested() && other.is_nested(),
      "Expected both self and other to be nested tensors. ",
      "Self ", self.is_nested()? "is " : "is not ",
      "nested. While Other ",
      other.is_nested()? "is " : "is not ",
      "nested.")
  const auto self_nt_size = _nested_tensor_size(self);
  const auto other_nt_size = _nested_tensor_size(other);
  return at::equal(self_nt_size, other_nt_size);
}
// 检查两个嵌套张量是否具有相同的大小

int64_t size(const Tensor& self, int64_t dim) {
  return self.size(dim);
}
// 返回张量在给定维度上的大小

int64_t stride(const Tensor& self, int64_t dim) {
  return self.stride(dim);
}
// 返回张量在给定维度上的步长

c10::SymInt sym_size(const Tensor& self, int64_t dim) {
  return self.sym_size(dim);
}
// 返回张量在给定维度上的符号大小

c10::SymInt sym_stride(const Tensor& self, int64_t dim) {
  return self.sym_stride(dim);
}
// 返回张量在给定维度上的符号步长

c10::SymInt sym_numel(const Tensor& self) {
  return self.sym_numel();
}
// 返回张量的符号元素个数

c10::SymInt sym_storage_offset(const Tensor& self) {
  return self.sym_storage_offset();
}
// 返回张量的符号存储偏移量

int64_t size(const Tensor& self, Dimname dim) {
  size_t pos_dim = dimname_to_position(self, dim);
  return self.sizes()[pos_dim];
}
// 返回张量在指定维度名称上的大小

int64_t stride(const Tensor& self, Dimname dim) {
  size_t pos_dim = dimname_to_position(self, dim);
  return self.strides()[pos_dim];
}
// 返回张量在指定维度名称上的步长

} // namespace at::native
// 结束 at::native 命名空间
// 检查是否可以使用 cuDNN 进行加速的条件判断函数，参数为 TensorBase 引用
bool cudnn_is_acceptable(const TensorBase& self) {
  // 如果全局上下文未启用 cuDNN，返回 false
  if (!globalContext().userEnabledCuDNN()) return false;
  // 如果张量不在 GPU 上，返回 false
  if (!self.is_cuda()) return false;
  // 获取张量的标量类型
  auto st = self.scalar_type();
  // 如果标量类型不是 kDouble、kFloat 或 kHalf，返回 false
  if (!(st == kDouble || st == kFloat || st == kHalf)) return false;
  // 如果当前的 cuDNN 没有与 PyTorch 编译兼容，返回 false
  if (!detail::getCUDAHooks().compiledWithCuDNN()) return false;
  // 如果张量的符号元素数为 0，返回 false
  // 注意：在旧的 Python 代码中，还有一个检查 cuDNN 库是否动态链接的测试，不过目前不能确定是否可以进行这种测试。
  if (self.sym_numel() == 0) return false;
  // 如果以上条件都满足，返回 true，表示可以使用 cuDNN 加速
  return true;
}

// 重载的 cudnn_is_acceptable 函数，参数为 Tensor 引用，调用上述函数进行判断
bool cudnn_is_acceptable(const Tensor& self) {
  return cudnn_is_acceptable(static_cast<const TensorBase&>(self));
}

// Tensor 类的 detach_ 函数，返回自身引用，用于 VariableType 的钩子和 Declarations.yaml 的条目
Tensor & detach_(Tensor & self) {
  // 这里仅为 VariableType 的钩子函数和 Declarations.yaml 中的入口而存在
  //AT_ERROR("detach_ is not implemented for Tensor");
  return self;
}

// contiguous 函数返回一个连续的张量，根据指定的内存格式 memory_format
Tensor contiguous(const Tensor& self, MemoryFormat memory_format) {
  // 如果张量已经是指定的内存格式连续，则直接返回自身
  if (self.is_contiguous(memory_format)) {
    return self;
  }
  // 如果要求保留内存格式，抛出异常，因为 contiguous 操作不支持保留内存格式
  TORCH_CHECK(
      memory_format != MemoryFormat::Preserve,
      "preserve memory format is unsupported by the contiguous operator");

  // 否则，返回一个按照指定内存格式克隆的张量
  return self.clone(memory_format);
}

// 检查一个张量是否与另一个张量相同
bool is_set_to(const Tensor& self, const Tensor& src) {
  // 如果两个张量的存储指针、存储偏移和维度都相同
  if (self.storage().unsafeGetStorageImpl() == src.storage().unsafeGetStorageImpl() &&
      self.storage_offset() == src.storage_offset() &&
      self.dim() == src.dim()) {
    // 检查每个维度的大小和步长是否一致
    for (const auto d : c10::irange(self.dim())) {
      if (self.size(d) != src.size(d) || self.stride(d) != src.stride(d)) {
        return false;
      }
    }
    // 如果所有维度都匹配，返回 true
    return true;
  }
  // 否则，返回 false
  return false;
}
```