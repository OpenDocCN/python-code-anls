# `.\pytorch\c10\util\strides.h`

```py
#pragma once
// 包含必要的头文件，用于处理数组引用和维度向量
#include <c10/util/ArrayRef.h>
#include <c10/util/DimVector.h>
// 包含标准库中的算法头文件，用于使用 std::max 函数
#include <algorithm>

// 定义命名空间 c10
namespace c10 {

// 计算张量的连续步长，根据其尺寸（sizes）来计算。
// 参数 sizes 是一个 IntArrayRef 类型，表示张量的尺寸。
inline DimVector contiguous_strides(const IntArrayRef sizes) {
  // 定义 Int 类型，用于表示 IntArrayRef 的值类型
  using Int = IntArrayRef::value_type;
  // 获取张量的维度数
  const Int dims = static_cast<Int>(sizes.size());

  // 初始化步长向量 strides，所有元素设置为 1
  DimVector strides(dims, 1);

  // 从倒数第二维开始向前计算步长
  for (auto i = dims - 2; i >= 0; --i) {
    // 计算当前维度的步长，确保步长不为 0，即使尺寸为 0 的情况下也要如此。
    strides[i] = strides[i + 1] * std::max(sizes[i + 1], Int{1});
  }

  // 返回计算得到的步长向量
  return strides;
}

} // namespace c10
```