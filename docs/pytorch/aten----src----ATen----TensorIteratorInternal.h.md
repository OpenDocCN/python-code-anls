# `.\pytorch\aten\src\ATen\TensorIteratorInternal.h`

```
#pragma once
#include <ATen/native/TensorIterator.h>
#include <c10/util/SmallBuffer.h>
#include <c10/util/irange.h>

namespace at {

// 维度计数器结构体，用于迭代多维张量的索引范围
struct DimCounter {
  // 构造函数，初始化形状和范围
  DimCounter(IntArrayRef shape, Range range);

  // 增加计数器的步长
  void increment(const std::array<int64_t, 2>& step);

  // 检查计数器是否完成迭代
  bool is_done() const;

  // 返回最大的二维步长
  std::array<int64_t, 2> max_2d_step() const;

  // 张量的形状
  IntArrayRef shape;

  // 迭代范围
  Range range;

  // 存储计数器当前值的缓冲区
  c10::SmallBuffer<int64_t, 4> values;

  // 偏移量
  int64_t offset;
};

namespace internal {

// 获取数据指针数组的内联函数
inline void get_data_ptrs(
    char** ptrs,
    ArrayRef<char*> base,
    IntArrayRef strides,
    IntArrayRef counter) {
  // 获取基础指针数组的大小
  const auto ntensors = base.size();
  
  // 获取计数器的维度大小
  const auto ndim = counter.size();
  
  // 将基础指针数组的内容复制到目标指针数组
  std::copy(base.begin(), base.end(), ptrs);

  // 遍历每个维度
  for (const auto dim : c10::irange(ndim)) {
    // 获取当前维度的计数值
    int64_t value = counter[dim];
    
    // 遍历每个张量，并根据步长调整指针位置
    for (const auto arg : c10::irange(ntensors)) {
      ptrs[arg] += value * strides[dim * ntensors + arg];
    }
  }
}

// 并行迭代执行函数
inline void serial_for_each(
    IntArrayRef shape,
    IntArrayRef strides,
    char** base_ptrs,
    size_t ntensors,
    typename TensorIteratorBase::loop2d_t loop,
    Range range) {
  // 获取张量的维度数
  const auto ndim = shape.size();
  
  // 调试断言，检查步长数组的大小
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      strides.size() == ntensors * std::max(size_t{2}, ndim));

  // 如果张量的维度小于等于1
  if (ndim <= 1) {
    // 如果迭代范围从0开始，直接调用循环函数处理
    if (range.begin == 0) {
      loop(base_ptrs, strides.data(), range.size(), 1);
    } else {
      // 否则，创建指针数组并根据计数器获取数据指针，然后调用循环函数处理
      c10::SmallBuffer<char*, 4> ptrs(ntensors);
      get_data_ptrs(ptrs.data(), {base_ptrs, ntensors}, strides, {range.begin});
      loop(ptrs.data(), strides.data(), range.size(), 1);
    }
  } else {
    // 对于高维张量，创建指针数组和维度计数器对象
    c10::SmallBuffer<char*, 4> ptrs(ntensors);
    auto counter = DimCounter(shape, range);
    
    // 循环执行，直到计数器完成所有迭代
    while (!counter.is_done()) {
      // 获取数据指针数组
      get_data_ptrs(
          ptrs.data(), {base_ptrs, ntensors}, strides, counter.values);
      
      // 获取当前迭代步长，并调用循环函数处理
      auto step = counter.max_2d_step();
      loop(ptrs.data(), strides.data(), step[0], step[1]);
      
      // 更新计数器的当前值
      counter.increment(step);
    }
  }
}

} // namespace internal
} // namespace at
```