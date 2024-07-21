# `.\pytorch\aten\src\ATen\TensorGeometry.cpp`

```
// 包含 ATen 库中的 TensorGeometry.h 文件

#include <ATen/TensorGeometry.h>

// 包含标准库头文件
#include <limits>
#include <cstddef>

// ATen 命名空间
namespace at {

// 在 TensorGeometry.h 中解释为何缓存 is_contiguous 很有用

// 模板函数，用于检查给定的 sizes 和 strides 是否表示连续存储
template <typename T>
bool _geometry_is_contiguous(ArrayRef<T> sizes, ArrayRef<T> strides) {
  // 断言确保 sizes.size() 不会导致整数溢出
  assert(!overflows<std::int64_t>(sizes.size()));
  // 将 sizes.size() 转换为 std::int64_t 类型
  auto dim = static_cast<std::int64_t>(sizes.size());
  // 初始期望的步长为 1
  T expected_stride = 1;
  // 如果非空则默认是连续的
  bool contig_if_nonempty = true;
  
  // 从最后一个维度开始向前遍历
  for (int64_t i = dim - 1; i >= 0; i--) {
    // 如果遇到某个维度大小为 0，视为连续
    if (sizes[i] == 0) {
      return true;
    }
    // 如果仍然假定是连续的
    if (contig_if_nonempty) {
      // 如果当前维度大小不为 1 且步长与期望步长不符，则标记为非连续
      if (sizes[i] != 1 && strides[i] != expected_stride) {
        contig_if_nonempty = false;
      }
      // 更新期望步长
      expected_stride *= sizes[i];
    }
  }
  // 返回是否连续的标志
  return contig_if_nonempty;
}

// 非模板函数，调用模板函数 _geometry_is_contiguous
bool geometry_is_contiguous(IntArrayRef sizes, IntArrayRef strides) {
  return _geometry_is_contiguous(sizes, strides);
}

// TensorGeometry 类的成员函数，检查张量是否是连续的
bool TensorGeometry::is_contiguous() const {
  // 如果张量元素个数为 0，则视为连续
  if (numel_ == 0) {
    return true;
  }
  // 调用 _geometry_is_contiguous 检查是否连续，使用 SymInt 类型
  return at::_geometry_is_contiguous<c10::SymInt>(sizes_, strides_);
}

} // namespace at
```