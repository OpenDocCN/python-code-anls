# `.\pytorch\c10\core\Contiguity.h`

```py
#pragma once
#include <c10/core/SymBool.h>  // 引入 SymBool 类型的头文件
#include <c10/core/SymInt.h>   // 引入 SymInt 类型的头文件
#include <c10/util/ArrayRef.h> // 引入 ArrayRef 类型的头文件
#include <c10/util/SmallVector.h>  // 引入 SmallVector 类型的头文件
#include <c10/util/irange.h>   // 引入 irange 函数

#include <algorithm>  // 引入算法库中的算法函数
#include <cstdint>    // 引入 C++ 标准整数类型

namespace c10 {

template <typename T>
bool _compute_contiguous(ArrayRef<T> sizes, ArrayRef<T> strides, T numel) {
  bool is_contiguous = true;  // 初始化是否连续的标志位为 true
  if (TORCH_GUARD_SIZE_OBLIVIOUS(sym_eq(numel, 0))) {  // 如果 numel 为零
    return is_contiguous;  // 直接返回 true
  }
  T z = 1;  // 初始化 z 为 1
  // NB: make sure we do signed arithmetic
  for (int64_t d = int64_t(sizes.size()) - 1; d >= 0; d--) {  // 逆序遍历 sizes
    const auto& size_d = sizes[d];  // 取当前维度的 size
    if (TORCH_GUARD_SIZE_OBLIVIOUS(sym_ne(size_d, 1))) {  // 如果 size_d 不为 1
      if (TORCH_GUARD_SIZE_OBLIVIOUS(sym_eq(strides[d], z))) {  // 如果 strides[d] 等于 z
        z *= size_d;  // 更新 z 为 size_d 的乘积
      } else {
        is_contiguous = false;  // 如果不满足条件，则置为 false
        break;  // 跳出循环
      }
    }
  }
  return is_contiguous;  // 返回是否连续的结果
}

template <typename T>
bool _compute_channels_last_contiguous_2d(
    ArrayRef<T> sizes,
    ArrayRef<T> strides) {
  // Please don't combine these code, constant array is used here to let
  // compiler fully unroll the loop to get better performance
  switch (sizes.size()) {  // 根据 sizes 的维度数量进行不同的处理
    case 4: {
      T expected = 1;  // 初始化期望的值为 1
      for (auto& d : {1, 3, 2, 0}) {  // 按指定顺序遍历维度
        const auto& size_d = sizes[d];  // 取当前维度的 size
        if (TORCH_GUARD_SIZE_OBLIVIOUS(sym_ne(size_d, 1))) {  // 如果 size_d 不为 1
          if (TORCH_GUARD_SIZE_OBLIVIOUS(sym_ne(strides[d], expected))) {  // 如果 strides[d] 不等于期望值
            return false;  // 返回 false
          }
          expected *= size_d;  // 更新期望值为 size_d 的乘积
        }
      }
      return true;  // 如果满足条件，返回 true
    }
      // NOLINTNEXTLINE(bugprone-branch-clone)
    case 3:
      // TODO dim == 3 case will be enabled once it is fully tested
      return false;  // 如果维度为 3，暂时返回 false
    default:
      return false;  // 其他情况均返回 false
  }
}

template <typename T>
bool _compute_channels_last_contiguous_3d(
    ArrayRef<T> sizes,
    ArrayRef<T> strides) {
  // Please don't combine these code, constant array is used here to let
  // compiler fully unroll the loop to get better performance
  switch (sizes.size()) {  // 根据 sizes 的维度数量进行不同的处理
    case 5: {
      T expected = 1;  // 初始化期望的值为 1
      for (auto& d : {1, 4, 3, 2, 0}) {  // 按指定顺序遍历维度
        const auto& size_d = sizes[d];  // 取当前维度的 size
        if (TORCH_GUARD_SIZE_OBLIVIOUS(sym_ne(size_d, 1))) {  // 如果 size_d 不为 1
          if (TORCH_GUARD_SIZE_OBLIVIOUS(sym_ne(strides[d], expected))) {  // 如果 strides[d] 不等于期望值
            return false;  // 返回 false
          }
          expected *= size_d;  // 更新期望值为 size_d 的乘积
        }
      }
      return true;  // 如果满足条件，返回 true
    }
      // NOLINTNEXTLINE(bugprone-branch-clone)
    case 4:
      // TODO dim == 4 case will be enabled once it is fully tested
      return false;  // 如果维度为 4，暂时返回 false
    default:
      return false;  // 其他情况均返回 false
  }
}

template <typename T>
bool _compute_non_overlapping_and_dense(
    ArrayRef<T> sizes,
    ArrayRef<T> strides) {
  auto dim = sizes.size();  // 获取 sizes 的维度数量
  if (dim == 1) {  // 如果维度数量为 1
    return sizes[0] < 2 || strides[0] == 1;  // 返回判断条件的结果
  }
  SmallVector<int64_t, 5> perm;  // 声明一个小向量 perm，容量为 5
  perm.resize(dim);  // 调整 perm 的大小为维度数量
  for (const auto i : c10::irange(dim)) {  // 遍历维度范围
    perm[i] = i;  // 初始化 perm 中的元素
  }
  // Sort by strides, leaving 0 and 1 sized dims at the end of the array
  std::sort(perm.begin(), perm.end(), [&](int64_t a, int64_t b) {  // 按照 strides 排序，将大小为 0 和 1 的维度放在数组末尾
    if (sizes[a] < 2) {  // 如果大小小于 2
      return false;  // 返回 false
    }
    } else if (sizes[b] < 2) {
      // 如果 sizes[b] 小于 2，返回 true，表示要求的条件满足
      return true;
    }
    // 比较 strides[a] 和 strides[b]，返回比较结果（true 或 false）
    return strides[a] < strides[b];
  });
  // 初始化需要的步幅为 1
  T require_stride = 1;
  // 遍历维度 dim
  for (const auto i : c10::irange(dim)) {
    // 获取当前维度 perm[i] 的大小 size_perm_i
    const auto& size_perm_i = sizes[perm[i]];
    // 如果 size_perm_i 小于 2，返回 true，表示要求的条件满足
    if (size_perm_i < 2) {
      return true;
    }
    // 检查当前维度 perm[i] 的步幅 strides[perm[i]] 是否等于 require_stride
    if (strides[perm[i]] != require_stride) {
      // 如果步幅不符合要求，返回 false
      return false;
    }
    // 更新 require_stride，乘以当前维度大小 size_perm_i
    require_stride *= size_perm_i;
  }
  // 所有维度的步幅要求都符合，返回 true
  return true;
}

} // namespace c10
```