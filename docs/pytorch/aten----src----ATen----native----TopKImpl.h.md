# `.\pytorch\aten\src\ATen\native\TopKImpl.h`

```
  // 如果 k 等于 0，则输出的值和索引是空张量，因此在其他维度上的迭代是没有意义的
  if (k == 0) {
    return;
  }
  // 使用 std::pair 将值和索引成对存储在队列中
  using elem_t = std::pair<accscalar_t, int64_t>;
  // 创建一个大小为 dim_size 的队列
  std::vector<elem_t> queue(dim_size);
  // 对于范围内的每个索引 i，执行以下操作
  for (const auto i : c10::irange(n)) {
    // 从 data 数组中读取模式值，并使用 TensorAccessor 进行访问
    TensorAccessor<scalar_t, 1> mode_values(
        reinterpret_cast<scalar_t*>(data[0] + i * strides[0]),
        &k, &mode_values_stride);
    // 从 data 数组中读取模式索引，并使用 TensorAccessor 进行访问
    TensorAccessor<int64_t, 1> mode_indices(
        reinterpret_cast<int64_t*>(data[1] + i * strides[1]),
        &k, &mode_indices_stride);
    // 从 data 数组中读取临时值，并使用 TensorAccessor 进行访问
    TensorAccessor<const scalar_t, 1> tmp_values(
        reinterpret_cast<scalar_t*>(data[2] + i * strides[2]),
        &dim_size, &tmp_values_stride);

    // 将 tmp_values 中的值复制到队列中，同时保存索引
    auto n_2 = dim_size;
    for (const auto j : c10::irange(n_2)) {
      queue[j].first = tmp_values[j];
      queue[j].second = j;
    }

    // 如果 k * 64 <= n_2，则使用部分排序算法
    auto use_partial_sort = k * 64 <= n_2;

    // 根据 largest 和 _isnan<accscalar_t> 的结果，使用部分排序对队列进行排序
    if (use_partial_sort) {
      if (largest) {
        std::partial_sort(queue.begin(), queue.begin() + k, queue.end(),
          [](const elem_t& x, const elem_t& y) -> bool {
            return ((_isnan<accscalar_t>(x.first) && !_isnan<accscalar_t>(y.first)) || (x.first > y.first));
          });
      } else {
        std::partial_sort(queue.begin(), queue.begin() + k, queue.end(),
          [](const elem_t& x, const elem_t& y) -> bool {
            return ((!_isnan<accscalar_t>(x.first) && _isnan<accscalar_t>(y.first)) || (x.first < y.first));
          });
      }
    }
    // 如果不是最大值模式，进入此分支
    } else {
      // 如果需要排序最大的 k 个元素
      if (largest) {
        // 使用 nth_element 函数对队列中前 k-1 个元素进行局部排序，使用 lambda 函数作为比较函数
        std::nth_element(queue.begin(), queue.begin() + k - 1, queue.end(),
          [](const elem_t& x, const elem_t& y) -> bool {
            // 返回 true 表示 x 应该排在 y 前面，考虑了 NaN 值情况和数值大小
            return ((_isnan<accscalar_t>(x.first) && !_isnan<accscalar_t>(y.first)) || (x.first > y.first));
          });
        // 如果需要完全排序
        if (sorted) {
          // 对局部排序后的前 k-1 个元素进行全局排序，使用 lambda 函数作为比较函数
          std::sort(queue.begin(), queue.begin() + k - 1,
            [](const elem_t& x, const elem_t& y) -> bool {
              // 返回 true 表示 x 应该排在 y 前面，考虑了 NaN 值情况和数值大小
              return ((_isnan<accscalar_t>(x.first) && !_isnan<accscalar_t>(y.first)) || (x.first > y.first));
            });
        }
      } else {
        // 使用 nth_element 函数对队列中前 k-1 个元素进行局部排序，使用 lambda 函数作为比较函数
        std::nth_element(queue.begin(), queue.begin() + k - 1, queue.end(),
          [](const elem_t& x, const elem_t& y) -> bool {
            // 返回 true 表示 x 应该排在 y 前面，考虑了 NaN 值情况和数值大小
            return ((!_isnan<accscalar_t>(x.first) && _isnan<accscalar_t>(y.first)) || (x.first < y.first));
          });
        // 如果需要完全排序
        if (sorted) {
          // 对局部排序后的前 k-1 个元素进行全局排序，使用 lambda 函数作为比较函数
          std::sort(queue.begin(), queue.begin() + k - 1,
            [](const elem_t& x, const elem_t& y) -> bool {
              // 返回 true 表示 x 应该排在 y 前面，考虑了 NaN 值情况和数值大小
              return ((!_isnan<accscalar_t>(x.first) && _isnan<accscalar_t>(y.first)) || (x.first < y.first));
            });
        }
      }
    }

    // 将排序后的前 k 个元素的第一个值存入 mode_values 数组
    for (const auto j : c10::irange(k)) {
      mode_values[j] = queue[j].first;
      // 将排序后的前 k 个元素的第二个值存入 mode_indices 数组
      mode_indices[j] = queue[j].second;
    }
  }
}

} // namespace CPU_CAPABILITY
} // namespace at::native
```