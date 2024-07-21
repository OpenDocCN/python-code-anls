# `.\pytorch\aten\src\ATen\native\cpu\IndexKernelUtils.h`

```py
#pragma once
#include <ATen/native/TensorIterator.h>
#include <c10/util/irange.h>

namespace at {
namespace native {

// 匿名命名空间中定义的静态函数，用于检查是否所有张量都有常量的索引
static bool is_constant_index(int ntensor, const int64_t* strides) {
  // 断言张量数量大于等于3
  AT_ASSERT(ntensor >= 3);
  // 遍历除去前两个张量以外的所有张量
  for (const auto arg : c10::irange(2, ntensor)) {
    // 如果当前张量的步长不为0，则返回false
    if (strides[arg] != 0) {
      return false;
    }
  }
  // 如果所有张量的步长都为0，则返回true
  return true;
}

// 索引器结构体，用于计算索引值
struct Indexer {
  Indexer(int64_t num_indexers, char** indexers, const int64_t* indexer_strides,
          IntArrayRef original_sizes, IntArrayRef original_strides)
    : num_indexers(num_indexers)
    , indexers(indexers)
    , indexer_strides(indexer_strides)
    , original_strides(original_strides.data())
    , original_sizes(original_sizes.data()) {
    // 断言原始步长和大小数组与索引数目相符
    AT_ASSERT(static_cast<int64_t>(original_strides.size()) == num_indexers);
    AT_ASSERT(static_cast<int64_t>(original_sizes.size()) == num_indexers);
  }

  int64_t num_indexers;
  char** indexers;
  const int64_t* indexer_strides;
  const int64_t* original_strides;
  const int64_t* original_sizes;

  // 根据索引值计算偏移量
  int64_t get(int64_t idx) {
    int64_t offset = 0;
    // 遍历所有索引器
    for (const auto j : c10::irange(num_indexers)) {
      // 计算当前索引器的值
      int64_t value = *(int64_t*)&indexers[j][idx * indexer_strides[j]];
      int64_t size = original_sizes[j];
      // 检查索引是否超出范围
      TORCH_CHECK_INDEX(value >= -size && value < size,
                        "index ", value, " is out of bounds for dimension ", j, " with size ", size);
      // 处理负索引
      if (value < 0) {
        value += size;
      }
      // 计算偏移量
      offset += value * original_strides[j];
    }
    return offset;
  }
};

// CPU索引核心函数模板，用于执行张量迭代器的索引操作
template <typename scalar_t, typename func_t>
void cpu_index_kernel(TensorIteratorBase& iter, IntArrayRef index_size, IntArrayRef index_stride,
                      const func_t& f, bool serial_execution=false)
{
  int ntensor = iter.ntensors();
  // 当启动索引并行版本时，设置一个相对较小的谷粒大小，小于INTERNAL::GRAIN_SIZE，
  // 以使整体可用线程数获得更平衡的工作负载和更好的缓存位置。
  const int index_parallel_grain_size = 3000;
  auto loop = [&](char** data, const int64_t* strides, int64_t n) {
    // 创建索引器对象
    auto indexer = Indexer(ntensor - 2, &data[2], &strides[2], index_size, index_stride);
    char* dst = data[0];
    char* src = data[1];
    // 如果所有张量具有常量索引，优化处理
    if (is_constant_index(ntensor, strides)) {
      int64_t offset = indexer.get(0);
      // 遍历所有元素，应用函数f
      for (const auto i : c10::irange(n)) {
        f(dst + strides[0] * i, src + strides[1] * i, offset);
      }
    } else {
      // 普通情况下的处理
      for (const auto i : c10::irange(n)) {
        int64_t offset = indexer.get(i);
        f(dst + strides[0] * i, src + strides[1] * i, offset);
      }
    }
  };
  // 根据是否串行执行选择不同的迭代方式
  if (serial_execution) {
    iter.serial_for_each(loop, {0, iter.numel()});
  } else {
    iter.for_each(loop, index_parallel_grain_size);
  }
}

} // at
} // native
```