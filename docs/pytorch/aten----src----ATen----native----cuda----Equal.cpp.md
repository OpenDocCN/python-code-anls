# `.\pytorch\aten\src\ATen\native\cuda\Equal.cpp`

```py
// 定义宏 TORCH_ASSERT_ONLY_METHOD_OPERATORS
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
// 包含 ATen 库中的 Tensor 类定义
#include <ATen/core/Tensor.h>
// 包含 ATen 库中的 NamedTensorUtils 头文件
#include <ATen/NamedTensorUtils.h>

// 根据不同的宏定义条件包含不同的头文件
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/NativeFunctions.h>
#include <ATen/CUDAFunctions.h>
#else
#include <ATen/ops/eq_cuda_dispatch.h>
#include <ATen/ops/equal_native.h>
#endif

// 定义 ATen 命名空间下的 native 命名空间
namespace at::native {

// 定义函数 cuda_equal，用于比较两个 Tensor 是否相等（CUDA 版本）
bool cuda_equal(const Tensor& self, const Tensor &src) {
  // 如果两个 Tensor 的命名不同，直接返回 false
  if (!at::namedinference::are_names_equal(
          self.unsafeGetTensorImpl(), src.unsafeGetTensorImpl())) {
    return false;
  }
  // 使用 NoNamesGuard 来确保在函数作用域内没有命名信息
  at::NoNamesGuard guard;
  // 检查两个 Tensor 是否在相同设备上，不同设备则抛出异常
  TORCH_CHECK(self.device() == src.device(), "Cannot compare two tensors on "
              "different devices. Got: ", self.device(), " and ", src.device());
  // 检查两个 Tensor 的尺寸是否相同，不同则返回 false
  if (self.sizes() != src.sizes()) {
    return false;
  }
  // 如果其中一个 Tensor 的元素个数为 0，则直接返回 true
  if (self.numel() == 0) {
    return true;
  }

  // 执行快速路径优化：确保存储和步幅完全相同，包括额外的标志位检查，用于优化比较
  if (self.is_alias_of(src)
      && self.storage_offset() == src.storage_offset()
      && self.dtype() == src.dtype()
      && self.is_contiguous() == src.is_contiguous()
      && self.strides().equals(src.strides())
      // 额外检查确保安全性，以防直接在 C++ 中调用 cuda_equal
      && self.layout() == src.layout()
      && self.is_neg() == src.is_neg()
      && self.is_conj() == src.is_conj()) {
    return true;
  }

  // 调用 CUDA 版本的 eq 函数，比较两个 Tensor 的内容是否完全相等，并返回比较结果
  return at::cuda::eq(self, src).all().item().to<bool>();
}

} // namespace at::native
```