# `.\pytorch\c10\core\SymbolicShapeMeta.cpp`

```py
// 包含必要的头文件
#include <c10/core/Contiguity.h>
#include <c10/core/MemoryFormat.h>
#include <c10/core/SymInt.h>
#include <c10/core/SymIntArrayRef.h>
#include <c10/core/SymbolicShapeMeta.h>

// 定义命名空间 c10
namespace c10 {

// 构造函数的实现，用于从另一个 SymbolicShapeMeta 实例复制非可变成员
// NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
SymbolicShapeMeta::SymbolicShapeMeta(const SymbolicShapeMeta& other)
    // 非可变成员可以在互斥锁外部访问
    : sizes_(other.sizes_),
      strides_(other.strides_),
      storage_offset_(other.storage_offset_),
      strides_valid_(other.strides_valid_) {
  // 使用互斥锁保护下面的代码块，确保线程安全地复制可变成员
  std::scoped_lock lock(other.mutables_);
  // 下面的成员变量必须在互斥锁的保护下复制，因此忽略 clang-tidy 的警告
  // NOLINTBEGIN(cppcoreguidelines-prefer-member-initializer)
  numel_ = other.numel_;
  is_contiguous_ = other.is_contiguous_;
  is_channels_last_contiguous_ = other.is_channels_last_contiguous_;
  is_channels_last_3d_contiguous_ = other.is_channels_last_3d_contiguous_;
  is_channels_last_ = other.is_channels_last_;
  is_channels_last_3d_ = other.is_channels_last_3d_;
  is_non_overlapping_and_dense_ = other.is_non_overlapping_and_dense_;
  available_.store(other.available_.load());
  // NOLINTEND(cppcoreguidelines-prefer-member-initializer)
}

// 根据 sizes 和 strides 规范化符号化数组的尺寸和步幅
static std::optional<
    std::tuple<SymNode, std::vector<SymNode>, std::vector<SymNode>>>
normalize_sym_sizes_strides(SymIntArrayRef sizes, SymIntArrayRef strides) {
  // 查找用于调度的 SymNode
  SymNode base;
  bool all_hinted = true;
  // 注意：sizes 和 strides 被保证为正数，因此只需要检查是否是堆分配
  for (const auto& s : sizes) {
    if (all_hinted && !s.has_hint()) {
      all_hinted = false;
    }
    if (!base && s.is_heap_allocated()) {
      base = s.toSymNode();
    }
  }
  for (const auto& s : strides) {
    if (all_hinted && !s.has_hint()) {
      all_hinted = false;
    }
    if (!base && s.is_heap_allocated()) {
      base = s.toSymNode();
    }
  }
  if (!base || all_hinted) {
    // 找不到合适的 SymNode，告诉调用者执行常规计算
    // 或者，如果所有都有提示，则也执行常规计算
    return c10::nullopt;
  }
  // 填充 SymNode 数组
  std::vector<SymNode> size_nodes;
  std::vector<SymNode> stride_nodes;
  size_nodes.reserve(sizes.size());
  stride_nodes.reserve(strides.size());
  for (const auto& s : sizes) {
    size_nodes.emplace_back(s.wrap_node(base));
  }
  for (const auto& s : strides) {
    stride_nodes.emplace_back(s.wrap_node(base));
  }
  return c10::make_optional(
      std::tuple<SymNode, std::vector<SymNode>, std::vector<SymNode>>(
          std::move(base), std::move(size_nodes), std::move(stride_nodes)));
}

// 根据当前对象的尺寸和步幅计算是否是连续的
SymBool SymbolicShapeMeta::compute_contiguous() const {
  if (!strides_valid_) {
    return false;
  }
  c10::SymIntArrayRef sizes(sizes_);
  c10::SymIntArrayRef strides(strides_);
  return _compute_contiguous(sizes, strides, numel());
}

// 剩下的成员函数实现略
// 请注意这里的注释不应该总结整个函数的功能，只需解释每个语句的作用即可

} // namespace c10
// 定义宏 `DEFINE_EAGER_SYMBOOL_COMPUTE`，用于创建针对具体节点实现的符号布尔计算函数
#define DEFINE_EAGER_SYMBOOL_COMPUTE(name, nodeimpl, fallback) \
  SymBool SymbolicShapeMeta::name() const {                    \
    // 如果步长无效，直接返回 false
    if (!strides_valid_) {                                     \
      return false;                                            \
    }                                                          \
    // 获取符号整数数组引用 `sizes` 和 `strides`
    c10::SymIntArrayRef sizes(sizes_);                         \
    c10::SymIntArrayRef strides(strides_);                     \
    // 调用指定的回调函数 `fallback`，返回计算结果
    return fallback(sizes, strides);                           \
  }

// 定义宏 `DEFINE_SYMBOOL_COMPUTE`，用于创建一般性符号布尔计算函数
#define DEFINE_SYMBOOL_COMPUTE(name, nodeimpl, fallback)        \
  SymBool SymbolicShapeMeta::name() const {                     \
    // 如果步长无效，直接返回 false
    if (!strides_valid_) {                                      \
      return false;                                             \
    }                                                           \
    // 标准化尺寸和步长，并获取标准化后的结果 `n`
    auto n = normalize_sym_sizes_strides(sizes_, strides_);     \
    // 如果成功标准化
    if (n.has_value()) {                                        \
      auto [base, size_nodes, stride_nodes] = *n;               \
      // 调用具体节点实现的函数 `nodeimpl` 进行计算，并返回计算结果
      return SymBool(base->nodeimpl(size_nodes, stride_nodes)); \
    } else {                                                    \
      // 获取符号整数数组引用 `sizes` 和 `strides`
      c10::SymIntArrayRef sizes(sizes_);                        \
      c10::SymIntArrayRef strides(strides_);                    \
      // 调用指定的回调函数 `fallback`，返回计算结果
      return fallback(sizes, strides);                          \
    }                                                           \
  }

// 定义具体的符号布尔计算函数并展开宏 `DEFINE_EAGER_SYMBOOL_COMPUTE` 和 `DEFINE_SYMBOOL_COMPUTE`

// 定义符号布尔计算函数 `compute_channels_last_contiguous_2d`，调用 `is_channels_last_contiguous_2d` 进行计算
DEFINE_EAGER_SYMBOOL_COMPUTE(compute_channels_last_contiguous_2d, is_channels_last_contiguous_2d, _compute_channels_last_contiguous_2d)
// 定义符号布尔计算函数 `compute_channels_last_contiguous_3d`，调用 `is_channels_last_contiguous_3d` 进行计算
DEFINE_EAGER_SYMBOOL_COMPUTE(compute_channels_last_contiguous_3d, is_channels_last_contiguous_3d, _compute_channels_last_contiguous_3d)
// 定义符号布尔计算函数 `compute_strides_like_channels_last_2d`，使用 `is_channels_last_strides_2d` 进行计算
DEFINE_EAGER_SYMBOOL_COMPUTE(compute_strides_like_channels_last_2d, is_channels_last_strides_2d, is_channels_last_strides_2d)
// 定义符号布尔计算函数 `compute_strides_like_channels_last_3d`，使用 `is_channels_last_strides_3d` 进行计算
DEFINE_EAGER_SYMBOOL_COMPUTE(compute_strides_like_channels_last_3d, is_channels_last_strides_3d, is_channels_last_strides_3d)
// 定义符号布尔计算函数 `compute_non_overlapping_and_dense`，调用 `is_non_overlapping_and_dense` 进行计算
DEFINE_SYMBOOL_COMPUTE(compute_non_overlapping_and_dense, is_non_overlapping_and_dense, _compute_non_overlapping_and_dense)

// 取消定义 `DEFINE_SYMBOOL_COMPUTE` 宏

// Glue compute
// NB: this logic very intentionally short circuits if possible.  Without
// short circuiting, it causes
// python test/functorch/test_aotdispatch.py -k
// test_aot_autograd_symbolic_exhaustive_nn_functional_unfold_cpu_float32 to run
// very slowly.

// 定义符号布尔计算函数 `compute_is_non_overlapping_and_dense_dim4`
SymBool SymbolicShapeMeta::compute_is_non_overlapping_and_dense_dim4() const {
  // 初始化是否连续
  init_is_contiguous();
  // 如果绝对为真，返回 true
  if (definitely_true(is_contiguous(), __FILE__, __LINE__)) {
    return true;
  }
  // 初始化是否通道为最后连续
  init_is_channels_last_contiguous();
  // 如果绝对为真，返回 true
  if (definitely_true(is_channels_last_contiguous(), __FILE__, __LINE__)) {
    return true;
  }
  // 返回是否连续或通道为最后连续或计算非重叠且密集的结果
  return is_contiguous() | is_channels_last_contiguous() |
      compute_non_overlapping_and_dense();
}
// 初始化判断当前对象是否以 channels_last_contiguous 的方式连续存储
SymBool SymbolicShapeMeta::compute_channels_last_contiguous_3d_dim5() const {
  init_is_channels_last_contiguous();
  // 如果当前对象已经是 channels_last_contiguous 的方式，则返回 false
  if (definitely_true(is_channels_last_contiguous(), __FILE__, __LINE__)) {
    return false;
  }
  // 否则返回当前对象不是 channels_last_contiguous 的方式与计算 channels_last_contiguous_3d 的结果的按位取反
  return ~is_channels_last_contiguous() & compute_channels_last_contiguous_3d();
}

// 初始化判断当前对象是否以 channels_last_3d_contiguous 的方式连续存储
SymBool SymbolicShapeMeta::compute_channels_last_2d_dim5() const {
  init_is_channels_last_3d_contiguous();
  // 如果当前对象已经是 channels_last_3d_contiguous 的方式，则返回 false
  if (definitely_true(is_channels_last_3d_contiguous(), __FILE__, __LINE__)) {
    return false;
  }
  // 否则返回当前对象不是 channels_last_3d_contiguous 的方式与计算 strides_like_channels_last_2d 的结果的按位取反
  return ~is_channels_last_3d_contiguous() &
      compute_strides_like_channels_last_2d();
}

// 判断当前对象是否以 channels_last 的方式存储
SymBool SymbolicShapeMeta::compute_channels_last_3d_dim5() const {
  // 如果当前对象已经是 channels_last 的方式，则返回 false
  if (definitely_true(is_channels_last(), __FILE__, __LINE__)) {
    return false;
  }
  // 否则返回当前对象不是 channels_last 的方式与计算 strides_like_channels_last_3d 的结果的按位取反
  return ~is_channels_last() & compute_strides_like_channels_last_3d();
}

// 判断当前对象是否是连续存储
SymBool SymbolicShapeMeta::compute_is_non_overlapping_and_dense_dim5() const {
  // 如果当前对象是连续存储，则返回 true
  if (definitely_true(is_contiguous(), __FILE__, __LINE__)) {
    return true;
  }
  // 如果当前对象是 channels_last_contiguous 的方式连续存储，则返回 true
  if (definitely_true(is_channels_last_contiguous(), __FILE__, __LINE__)) {
    return true;
  }
  // 如果当前对象是 channels_last_3d_contiguous 的方式连续存储，则返回 true
  if (definitely_true(is_channels_last_3d_contiguous(), __FILE__, __LINE__)) {
    return true;
  }
  // 否则返回当前对象是连续存储或是 channels_last_contiguous 或是 channels_last_3d_contiguous 的方式连续存储，或计算非重叠和密集的结果
  return is_contiguous() | is_channels_last_contiguous() |
      is_channels_last_3d_contiguous() | compute_non_overlapping_and_dense();
}

// 判断当前对象是否是连续存储或计算非重叠和密集的结果
SymBool SymbolicShapeMeta::compute_is_non_overlapping_and_dense_anydim() const {
  // 如果当前对象是连续存储，则返回 true
  if (definitely_true(is_contiguous(), __FILE__, __LINE__)) {
    return true;
  }
  // 否则返回当前对象是连续存储或计算非重叠和密集的结果
  return is_contiguous() | compute_non_overlapping_and_dense();
}

// 设置对象的 numel 属性
// NOLINTNEXTLINE(performance-unnecessary-value-param)
void SymbolicShapeMeta::set_numel(SymInt val) const {
  std::scoped_lock lock(mutables_);
  // 如果已经设置过 numel，则直接返回
  if (has_numel()) {
    return;
  }
  // 否则设置对象的 numel 属性为给定值，并将相应标志位设置为可用
  numel_ = std::move(val);
  available_.fetch_or(numel_avail);
}

// 设置对象的 is_contiguous 属性
void SymbolicShapeMeta::set_is_contiguous(SymBool val) const {
  std::scoped_lock lock(mutables_);
  // 如果已经设置过 is_contiguous，则直接返回
  if (has_is_contiguous()) {
    return;
  }
  // 否则设置对象的 is_contiguous 属性为给定值，并将相应标志位设置为可用
  is_contiguous_ = std::move(val);
  available_.fetch_or(is_contiguous_avail);
}

// 设置对象的 is_channels_last_contiguous 属性
void SymbolicShapeMeta::set_is_channels_last_contiguous(SymBool val) const {
  std::scoped_lock lock(mutables_);
  // 如果已经设置过 is_channels_last_contiguous，则直接返回
  if (has_is_channels_last_contiguous()) {
    return;
  }
  // 否则设置对象的 is_channels_last_contiguous 属性为给定值，并将相应标志位设置为可用
  is_channels_last_contiguous_ = std::move(val);
  available_.fetch_or(is_channels_last_contiguous_avail);
}

// 设置对象的 is_channels_last_3d_contiguous 属性
void SymbolicShapeMeta::set_is_channels_last_3d_contiguous(SymBool val) const {
  std::scoped_lock lock(mutables_);
  // 如果已经设置过 is_channels_last_3d_contiguous，则直接返回
  if (has_is_channels_last_3d_contiguous()) {
    return;
  }
  // 否则设置对象的 is_channels_last_3d_contiguous 属性为给定值，并将相应标志位设置为可用
  is_channels_last_3d_contiguous_ = std::move(val);
  available_.fetch_or(is_channels_last_3d_contiguous_avail);
}

// 设置对象的 is_channels_last 属性
void SymbolicShapeMeta::set_is_channels_last(SymBool val) const {
  std::scoped_lock lock(mutables_);
  // 如果已经设置过 is_channels_last，则直接返回
  if (has_is_channels_last()) {
    return;
  }
  // 否则设置对象的 is_channels_last 属性为给定值，并将相应标志位设置为可用
  is_channels_last_ = std::move(val);
  available_.fetch_or(is_channels_last_avail);
}

// 设置对象的 is_channels_last_3d 属性
void SymbolicShapeMeta::set_is_channels_last_3d(SymBool val) const {
  std::scoped_lock lock(mutables_);
  // 如果已经设置过 is_channels_last_3d，则直接返回
  if (has_is_channels_last_3d()) {
    return;
  }
  // 否则设置对象的 is_channels_last_3d 属性为给定值，并将相应标志位设置为可用
  is_channels_last_3d_ = std::move(val);
  available_.fetch_or(is_channels_last_3d_avail);
}
    return;
  }


    // 如果函数不应该继续执行，直接返回
    return;
  }



  is_channels_last_3d_ = std::move(val);


  // 将变量 val 移动赋值给成员变量 is_channels_last_3d_
  is_channels_last_3d_ = std::move(val);



  available_.fetch_or(is_channels_last_3d_avail);


  // 使用原子操作将 is_channels_last_3d_avail 的值设置到 available_ 中
  available_.fetch_or(is_channels_last_3d_avail);
// 设置符号形状元数据的is_non_overlapping_and_dense属性为给定的值
void SymbolicShapeMeta::set_is_non_overlapping_and_dense(SymBool val) const {
  // 使用互斥锁锁定可变数据
  std::scoped_lock lock(mutables_);
  // 如果已经设置过is_non_overlapping_and_dense属性，则直接返回
  if (has_is_non_overlapping_and_dense()) {
    return;
  }
  // 设置is_non_overlapping_and_dense属性为给定的值
  is_non_overlapping_and_dense_ = std::move(val);
  // 使用原子操作更新available_标志位，设置is_non_overlapping_and_dense_avail位
  available_.fetch_or(is_non_overlapping_and_dense_avail);
}

// 初始化符号形状元数据的numel属性
void SymbolicShapeMeta::init_numel() const {
  // 使用已有的sizes_计算并设置numel属性
  set_numel(multiply_integers(sizes_));
}

// 初始化符号形状元数据的is_contiguous属性
void SymbolicShapeMeta::init_is_contiguous() const {
  // 使用计算函数compute_contiguous()设置is_contiguous属性
  set_is_contiguous(compute_contiguous());
}

// 初始化符号形状元数据的is_channels_last_contiguous属性
void SymbolicShapeMeta::init_is_channels_last_contiguous() const {
  // 使用lambda函数根据维度情况设置is_channels_last_contiguous属性
  set_is_channels_last_contiguous([&] {
    switch (dim()) {
      case 5:
      case 4: {
        return compute_channels_last_contiguous_2d();
      }
      default:
        return SymBool{false};
    }
  }());
}

// 初始化符号形状元数据的is_channels_last_3d_contiguous属性
void SymbolicShapeMeta::init_is_channels_last_3d_contiguous() const {
  // 使用lambda函数根据维度情况设置is_channels_last_3d_contiguous属性
  set_is_channels_last_3d_contiguous([&] {
    switch (dim()) {
      case 5:
        return compute_channels_last_contiguous_3d_dim5();
      default:
        return SymBool{false};
    }
  }());
}

// 初始化符号形状元数据的is_channels_last属性
void SymbolicShapeMeta::init_is_channels_last() const {
  // 使用lambda函数根据维度情况设置is_channels_last属性
  set_is_channels_last([&] {
    switch (dim()) {
      case 5:
        return compute_channels_last_2d_dim5();
      case 4:
        return compute_strides_like_channels_last_2d();
      default:
        return SymBool{false};
    }
  }());
}

// 初始化符号形状元数据的is_channels_last_3d属性
void SymbolicShapeMeta::init_is_channels_last_3d() const {
  // 使用lambda函数根据维度情况设置is_channels_last_3d属性
  set_is_channels_last_3d([&] {
    switch (dim()) {
      case 5:
        return compute_channels_last_3d_dim5();
      default:
        return SymBool{false};
    }
  }());
}

// 初始化符号形状元数据的is_non_overlapping_and_dense属性
void SymbolicShapeMeta::init_is_non_overlapping_and_dense() const {
  // 使用lambda函数根据维度情况设置is_non_overlapping_and_dense属性
  set_is_non_overlapping_and_dense([&] {
    switch (dim()) {
      case 5:
        return compute_is_non_overlapping_and_dense_dim5();
      case 4:
        return compute_is_non_overlapping_and_dense_dim4();
      default:
        return compute_is_non_overlapping_and_dense_anydim();
    }
  }());
}

// namespace c10 结束
} // namespace c10
```