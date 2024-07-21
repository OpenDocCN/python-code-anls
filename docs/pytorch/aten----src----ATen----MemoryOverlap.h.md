# `.\pytorch\aten\src\ATen\MemoryOverlap.h`

```py
#pragma once

#include <c10/macros/Export.h>

namespace c10 {
struct TensorImpl;
}

namespace at {
class TensorBase;

// MemOverlap: Whether or not there is memory overlap
//
// No: Absolutely no memory overlap
// Yes: Absolutely yes memory overlap
// TooHard: There might be memory overlap, but it was too expensive to compute.
//
// NB: Please update the python test for these if you renumber them.
// 定义枚举类型 MemOverlap，表示内存重叠情况
enum class MemOverlap { No, Yes, TooHard };

// MemOverlapStatus: Status of memory overlap between tensors
//
// Full: Full memory overlap
// Partial: Partial memory overlap
// No: No memory overlap
// TooHard: Expensive to determine overlap status
//
// NB: Please update the python test for these if you renumber them.
// 定义枚举类型 MemOverlapStatus，表示张量之间的内存重叠状态
enum class MemOverlapStatus { Full, Partial, No, TooHard };

// TORCH_API: Declaration for TorchScript API visibility
// 获取张量对象 t 的内部重叠情况，并返回 MemOverlap 枚举
TORCH_API MemOverlap has_internal_overlap(const TensorBase& t);
// 获取张量实现对象 t 的内部重叠情况，并返回 MemOverlap 枚举
TORCH_API MemOverlap has_internal_overlap(c10::TensorImpl* t);

// TORCH_API: Declaration for TorchScript API visibility
// 断言张量对象 t 没有内部重叠
TORCH_API void assert_no_internal_overlap(const TensorBase& t);
// 断言张量实现对象 t 没有内部重叠
TORCH_API void assert_no_internal_overlap(c10::TensorImpl* t);

// TORCH_API: Declaration for TorchScript API visibility
// 获取张量对象 a 和 b 之间的重叠状态，并返回 MemOverlapStatus 枚举
TORCH_API MemOverlapStatus get_overlap_status(const TensorBase& a, const TensorBase& b);
// 获取张量实现对象 a 和 b 之间的重叠状态，并返回 MemOverlapStatus 枚举
TORCH_API MemOverlapStatus get_overlap_status(const c10::TensorImpl* a, const c10::TensorImpl* b);

// TORCH_API: Declaration for TorchScript API visibility
// 断言张量对象 a 和 b 没有部分重叠
TORCH_API void assert_no_partial_overlap(const TensorBase& a, const TensorBase& b);
// 断言张量实现对象 a 和 b 没有部分重叠
void assert_no_partial_overlap(c10::TensorImpl* a, c10::TensorImpl* b);

// TORCH_API: Declaration for TorchScript API visibility
// 断言张量对象 a 和 b 没有重叠
TORCH_API void assert_no_overlap(const TensorBase& a, const TensorBase& b);
// 断言张量实现对象 a 和 b 没有重叠
TORCH_API void assert_no_overlap(c10::TensorImpl* a, c10::TensorImpl* b);

} // namespace at
```