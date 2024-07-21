# `.\pytorch\aten\src\ATen\MemoryOverlap.cpp`

```py
// ATen 库头文件，用于张量内存重叠检测
#include <ATen/MemoryOverlap.h>
// ATen 核心张量基类头文件
#include <ATen/core/TensorBase.h>
// 张量布局定义头文件
#include <c10/core/Layout.h>
// C10 实用工具，用于范围迭代
#include <c10/util/irange.h>

// ATen 命名空间
namespace at {

// 检查张量是否存在内部重叠，基于张量基类的引用
MemOverlap has_internal_overlap(const TensorBase& tensor) {
  // 调用底层实现的内部重叠检测函数
  return has_internal_overlap(tensor.unsafeGetTensorImpl());
}

// 检查张量是否存在内部重叠，基于张量实现指针
MemOverlap has_internal_overlap(TensorImpl* t) {
  // 断言张量布局为 kStrided（步长张量）
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(t->layout() == kStrided);

  // 如果张量被标记为非重叠且密集，返回无内部重叠
  if (t->is_non_overlapping_and_dense()) {
    return MemOverlap::No;
  }

  // 获取张量的符号步长和符号大小
  auto strides = t->sym_strides();
  auto sizes = t->sym_sizes();
  
  // 遍历张量的各个维度
  for (const auto i : c10::irange(strides.size())) {
    // 进行大小不敏感的测试，确保在某些未支持的符号整数设置下，正确报告内存重叠可能性
    if (TORCH_GUARD_SIZE_OBLIVIOUS(sizes[i].sym_gt(1)) && strides[i] == 0) {
      return MemOverlap::Yes;
    }
  }

  // 如果以上条件都未满足，则返回过于复杂，无法确定内部重叠状态
  return MemOverlap::TooHard;
}

// 断言张量不存在内部重叠，基于张量基类的引用
void assert_no_internal_overlap(const TensorBase& t) {
  // 调用底层实现的内部重叠断言函数
  assert_no_internal_overlap(t.unsafeGetTensorImpl());
}

// 断言张量不存在内部重叠，基于张量实现指针
void assert_no_internal_overlap(TensorImpl* t) {
  // 检查张量是否存在内部重叠，如果存在，则抛出异常
  TORCH_CHECK(has_internal_overlap(t) != MemOverlap::Yes,
    "unsupported operation: more than one element of the written-to tensor "
    "refers to a single memory location. Please clone() the tensor before "
    "performing the operation.");
}

// 获取两个张量之间的内存重叠状态，基于张量基类的引用
MemOverlapStatus get_overlap_status(const TensorBase& a, const TensorBase& b) {
  // 调用底层实现的张量重叠状态获取函数
  return get_overlap_status(a.unsafeGetTensorImpl(), b.unsafeGetTensorImpl());
}

// 获取两个张量之间的内存重叠状态，基于张量实现指针
MemOverlapStatus get_overlap_status(const TensorImpl* a, const TensorImpl* b) {
  // 如果两个张量指针相同，则完全重叠
  if (a == b) return MemOverlapStatus::Full;
  // 如果任一张量的元素数为零，则不存在重叠
  if (a->numel() == 0 || b->numel() == 0) {
    return MemOverlapStatus::No;
  }
  // 如果任一张量不是非重叠且密集，则重叠状态复杂，无法确定
  if (!a->is_non_overlapping_and_dense() || !b->is_non_overlapping_and_dense()) {
    return MemOverlapStatus::TooHard;
  }

  // 检查张量的存储是否相等，而不仅仅是指针相等
  auto a_storage = a->unsafe_storage();
  if (a_storage && a_storage.is_alias_of(b->unsafe_storage())) {
    const auto a_begin = static_cast<const char*>(a->data());
    const auto a_end = a_begin + a->numel() * a->itemsize();
    const auto b_begin = static_cast<const char*>(b->data());
    const auto b_end = b_begin + b->numel() * b->itemsize();

    // 如果起始地址和结束地址完全一致，并且步长相等，则完全重叠，否则部分重叠
    if (a_begin == b_begin && a_end == b_end) {
      return (a->strides() == b->strides()) ?
          MemOverlapStatus::Full : MemOverlapStatus::Partial;
    }
    // 如果起始地址交叉，则部分重叠
    if (a_begin < b_end && b_begin < a_end) {
      return MemOverlapStatus::Partial;
    }
  }

  // 若以上条件都未满足，则认为不存在内存重叠
  return MemOverlapStatus::No;
}

} // namespace at
# 确保两个张量对象没有部分重叠的内存区域
void assert_no_partial_overlap(const TensorBase& a, const TensorBase& b) {
  # 调用底层函数，检查两个张量实现对象之间是否存在部分重叠
  assert_no_partial_overlap(a.unsafeGetTensorImpl(), b.unsafeGetTensorImpl());
}

# 确保两个张量实现对象没有部分重叠的内存区域
void assert_no_partial_overlap(TensorImpl* a, TensorImpl* b) {
  # 获取两个张量实现对象之间的重叠状态
  const auto lap = get_overlap_status(a, b);
  # 使用 Torch 的检查机制确保不存在部分重叠或完全重叠
  TORCH_CHECK(lap != MemOverlapStatus::Partial,
    "unsupported operation: some elements of the input tensor and "
    "the written-to tensor refer to a single memory location. "
    "Please clone() the tensor before performing the operation.");
}

# 确保两个张量对象没有完全或部分重叠的内存区域
void assert_no_overlap(const TensorBase& a, const TensorBase& b) {
  # 调用底层函数，检查两个张量实现对象之间是否存在完全或部分重叠
  assert_no_overlap(a.unsafeGetTensorImpl(), b.unsafeGetTensorImpl());
}

# 确保两个张量实现对象没有完全或部分重叠的内存区域
void assert_no_overlap(TensorImpl* a, TensorImpl* b) {
  # 获取两个张量实现对象之间的重叠状态
  const auto lap = get_overlap_status(a, b);
  # 使用 Torch 的检查机制确保不存在部分重叠或完全重叠
  TORCH_CHECK(lap != MemOverlapStatus::Partial && lap != MemOverlapStatus::Full,
    "unsupported operation: some elements of the input tensor and "
    "the written-to tensor refer to a single memory location. "
    "Please clone() the tensor before performing the operation.");
}
```