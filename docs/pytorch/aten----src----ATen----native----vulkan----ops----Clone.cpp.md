# `.\pytorch\aten\src\ATen\native\vulkan\ops\Clone.cpp`

```
// 引入 Vulkan 相关的头文件
#include <ATen/native/vulkan/ops/Common.h>
#include <torch/library.h>

// 根据是否定义了 AT_PER_OPERATOR_HEADERS 来选择包含对应的头文件
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/empty_like.h>
#include <ATen/ops/empty_strided.h>
#endif

// 定义命名空间 at::native::vulkan::ops
namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace {

// clone 函数的实现，复制给定的 Tensor，并按照可选的内存格式复制
Tensor clone(
    const Tensor& src,  // 源 Tensor
    std::optional<c10::MemoryFormat> optional_memory_format) {  // 可选的内存格式
  auto memory_format = optional_memory_format.value_or(MemoryFormat::Preserve);  // 获取内存格式，默认为 Preserve
  // 检查内存格式是否为 Preserve 或 Contiguous
  TORCH_CHECK(
      (c10::MemoryFormat::Preserve == memory_format) ||
          (c10::MemoryFormat::Contiguous == memory_format),
      "Vulkan supports Preserve and Contiguous memory formats");

  Tensor self;  // 新建的目标 Tensor
  if (memory_format == MemoryFormat::Preserve) {
    if (src.is_non_overlapping_and_dense()) {
      // 如果源 Tensor 是非重叠且密集的，直接复制所有步长，比调用 empty_like 稍快
      self = at::empty_strided(src.sizes(), src.strides(), src.options());
    } else {
      self = at::empty_like(src);  // 否则调用 empty_like 复制源 Tensor 的属性
    }
  } else {
    self = at::empty_like(src, src.options(), memory_format);  // 使用指定的内存格式复制
  }

  self.copy_(src);  // 将源 Tensor 的数据复制到目标 Tensor
  return self;  // 返回复制后的目标 Tensor
}

#ifdef USE_VULKAN_API

// 注册 Vulkan 版本的 clone 实现到 aten 库中
TORCH_LIBRARY_IMPL(aten, Vulkan, m) {
  m.impl(TORCH_SELECTIVE_NAME("aten::clone"), TORCH_FN(clone));
}

#endif /* USE_VULKAN_API */

} // namespace
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
```