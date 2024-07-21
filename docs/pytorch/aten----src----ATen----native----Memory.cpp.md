# `.\pytorch\aten\src\ATen\native\Memory.cpp`

```
// 定义宏 TORCH_ASSERT_ONLY_METHOD_OPERATORS，用于仅包含方法操作符
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS

// 包含 ATen 核心 Tensor 类头文件
#include <ATen/core/Tensor.h>
// 包含 ATen 内存重叠检查头文件
#include <ATen/MemoryOverlap.h>

// 如果未定义 AT_PER_OPERATOR_HEADERS，则包含以下标准功能和本地功能头文件
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
// 如果定义了 AT_PER_OPERATOR_HEADERS，则包含以下特定功能的内部操作头文件
#else
#include <ATen/ops/_debug_has_internal_overlap_native.h>
#include <ATen/ops/_pin_memory.h>
#include <ATen/ops/is_pinned_native.h>
#include <ATen/ops/pin_memory_native.h>
#endif

// 定义 ATen::native 命名空间
namespace at::native {

// 将 at::has_internal_overlap 暴露为一个操作符，供测试目的使用
int64_t _debug_has_internal_overlap(const Tensor& self) {
  return static_cast<int64_t>(at::has_internal_overlap(self));
}

// 默认情况下，检查张量是否被固定在内存中，如果未加载后端扩展或不支持，返回 false
bool is_pinned_default(const Tensor& self, std::optional<Device> device) {
  return false;
}

// 将张量固定在内存中，如果已经固定则返回自身，否则调用 at::_pin_memory 进行固定
Tensor pin_memory(const Tensor& self, std::optional<Device> device) {
  // 这里有点烦，必须进行两次动态分派，相当让人恼火
  if (self.is_pinned(device)) {
    return self;
  }
  return at::_pin_memory(self, device);
}

} // namespace at::native
```