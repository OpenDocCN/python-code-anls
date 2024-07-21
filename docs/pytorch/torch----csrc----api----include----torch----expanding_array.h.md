# `.\pytorch\torch\csrc\api\include\torch\expanding_array.h`

```
#pragma once

#include <c10/util/ArrayRef.h>
#include <c10/util/Exception.h>
#include <c10/util/Optional.h>
#include <c10/util/irange.h>

#include <algorithm>   // 包含标准库算法头文件，用于算法操作
#include <array>       // 包含标准库数组头文件，用于数组操作
#include <cstdint>     // 包含标准整数类型头文件，定义了特定宽度的整数类型
#include <initializer_list>  // 包含初始化列表头文件，用于处理初始化列表
#include <string>      // 包含标准字符串头文件，定义了字符串类型和操作
#include <vector>      // 包含标准向量头文件，定义了向量类型和操作

namespace torch {

/// A utility class that accepts either a container of `D`-many values, or a
/// single value, which is internally repeated `D` times. This is useful to
/// represent parameters that are multidimensional, but often equally sized in
/// all dimensions. For example, the kernel size of a 2D convolution has an `x`
/// and `y` length, but `x` and `y` are often equal. In such a case you could
/// just pass `3` to an `ExpandingArray<2>` and it would "expand" to `{3, 3}`.
template <size_t D, typename T = int64_t>
class ExpandingArray {
 public:
  /// Constructs an `ExpandingArray` from an `initializer_list`. The extent of
  /// the length is checked against the `ExpandingArray`'s extent parameter `D`
  /// at runtime.
  /*implicit*/ ExpandingArray(std::initializer_list<T> list)
      : ExpandingArray(at::ArrayRef<T>(list)) {}

  /// Constructs an `ExpandingArray` from an `std::vector`. The extent of
  /// the length is checked against the `ExpandingArray`'s extent parameter `D`
  /// at runtime.
  /*implicit*/ ExpandingArray(std::vector<T> vec)
      : ExpandingArray(at::ArrayRef<T>(vec)) {}

  /// Constructs an `ExpandingArray` from an `at::ArrayRef`. The extent of
  /// the length is checked against the `ExpandingArray`'s extent parameter `D`
  /// at runtime.
  /*implicit*/ ExpandingArray(at::ArrayRef<T> values) {
    // clang-format off
    // 使用 clang-format 工具格式化源码显示
    TORCH_CHECK(
        values.size() == D,
        "Expected ", D, " values, but instead got ", values.size());
    // clang-format on
    // 将 values 的内容复制到 values_ 数组中
    std::copy(values.begin(), values.end(), values_.begin());
  }

  /// Constructs an `ExpandingArray` from a single value, which is repeated `D`
  /// times (where `D` is the extent parameter of the `ExpandingArray`).
  /*implicit*/ ExpandingArray(T single_size) {
    // 将 single_size 值复制 D 次填充到 values_ 数组中
    values_.fill(single_size);
  }

  /// Constructs an `ExpandingArray` from a correctly sized `std::array`.
  /*implicit*/ ExpandingArray(const std::array<T, D>& values)
      : values_(values) {}

  /// Accesses the underlying `std::array`.
  std::array<T, D>& operator*() {
    // 返回 values_ 数组的引用
    return values_;
  }

  /// Accesses the underlying `std::array`.
  const std::array<T, D>& operator*() const {
    // 返回 values_ 数组的常量引用
    return values_;
  }

  /// Accesses the underlying `std::array`.
  std::array<T, D>* operator->() {
    // 返回 values_ 数组的指针
    return &values_;
  }

  /// Accesses the underlying `std::array`.
  const std::array<T, D>* operator->() const {
    // 返回 values_ 数组的常量指针
    return &values_;
  }

  /// Returns an `ArrayRef` to the underlying `std::array`.
  operator at::ArrayRef<T>() const {
    // 将 values_ 数组转换为 at::ArrayRef<T> 类型并返回
    return values_;
  }

  /// Returns the extent of the `ExpandingArray`.
  size_t size() const noexcept {
    // 返回 ExpandingArray 的长度 D
    return D;
  }

 private:
  // 内部存储的数据结构为 std::array<T, D>
  std::array<T, D> values_;
};

}  // namespace torch
    return D;
  }



# 返回数组的维度 D
  }

 protected:
  /// The backing array.
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  std::array<T, D> values_;
/// 重载运算符 `<<`，用于将 `ExpandingArray<D, T>` 对象输出到流中
template <size_t D, typename T>
std::ostream& operator<<(
    std::ostream& stream,
    const ExpandingArray<D, T>& expanding_array) {
  // 如果 `ExpandingArray` 的大小为1，则直接输出其第一个元素
  if (expanding_array.size() == 1) {
    return stream << expanding_array->at(0);
  }
  // 否则，将 `ExpandingArray` 转换为 `at::ArrayRef<T>` 后输出到流中
  return stream << static_cast<at::ArrayRef<T>>(expanding_array);
}

/// `ExpandingArrayWithOptionalElem` 是一个实用类，可以接受包含 `D` 个 `std::optional<T>` 值的容器，
/// 或者一个单独的 `std::optional<T>` 值，该值在内部重复 `D` 次。还可以接受 `T` 的容器，并将其转换为 `std::optional<T>` 的容器。
template <size_t D, typename T = int64_t>
class ExpandingArrayWithOptionalElem
    : public ExpandingArray<D, std::optional<T>> {
 public:
  using ExpandingArray<D, std::optional<T>>::ExpandingArray;

  /// 从 `initializer_list` 构造 `ExpandingArrayWithOptionalElem`，其中元素类型为 `T`。
  /// 运行时检查列表的长度是否与 `D` 相符。
  /*implicit*/ ExpandingArrayWithOptionalElem(std::initializer_list<T> list)
      : ExpandingArrayWithOptionalElem(at::ArrayRef<T>(list)) {}

  /// 从 `std::vector` 构造 `ExpandingArrayWithOptionalElem`，其中元素类型为 `T`。
  /// 运行时检查向量的长度是否与 `D` 相符。
  /*implicit*/ ExpandingArrayWithOptionalElem(std::vector<T> vec)
      : ExpandingArrayWithOptionalElem(at::ArrayRef<T>(vec)) {}

  /// 从 `at::ArrayRef` 构造 `ExpandingArrayWithOptionalElem`，其中元素类型为 `T`。
  /// 运行时检查数组的长度是否与 `D` 相符。
  /*implicit*/ ExpandingArrayWithOptionalElem(at::ArrayRef<T> values)
      : ExpandingArray<D, std::optional<T>>(0) {
    // clang-format off
    // 检查输入数组的长度是否与 `D` 相符
    TORCH_CHECK(
        values.size() == D,
        "Expected ", D, " values, but instead got ", values.size());
    // clang-format on
    // 将输入数组的值复制到 `values_` 数组中
    for (const auto i : c10::irange(this->values_.size())) {
      this->values_[i] = values[i];
    }
  }

  /// 从单个值构造 `ExpandingArrayWithOptionalElem`，该值的类型为 `T`，
  /// 并重复 `D` 次（`D` 为 `ExpandingArrayWithOptionalElem` 的长度参数）。
  /*implicit*/ ExpandingArrayWithOptionalElem(T single_size)
      : ExpandingArray<D, std::optional<T>>(0) {
    // 将单个值重复填充到 `values_` 数组中
    for (const auto i : c10::irange(this->values_.size())) {
      this->values_[i] = single_size;
    }
  }

  /// 从正确尺寸的 `std::array` 构造 `ExpandingArrayWithOptionalElem`，其中元素类型为 `T`。
  /*implicit*/ ExpandingArrayWithOptionalElem(const std::array<T, D>& values)
      : ExpandingArray<D, std::optional<T>>(0) {
    // 将 `std::array` 的值复制到 `values_` 数组中
    for (const auto i : c10::irange(this->values_.size())) {
      this->values_[i] = values[i];
    }
  }
};
// 定义模板函数，重载流插入运算符<<，用于输出带有可选元素的动态数组
template <size_t D, typename T>
std::ostream& operator<<(
    std::ostream& stream,  // 输出流对象
    const ExpandingArrayWithOptionalElem<D, T>& expanding_array_with_opt_elem) {  // 带有可选元素的动态数组引用参数
  // 如果数组大小为1
  if (expanding_array_with_opt_elem.size() == 1) {
    // 获取数组的第一个元素
    const auto& elem = expanding_array_with_opt_elem->at(0);
    // 输出元素的值（若有值则转换为字符串，否则输出"None"）
    stream << (elem.has_value() ? c10::str(elem.value()) : "None");
  } else {
    // 创建一个存放字符串的向量
    std::vector<std::string> str_array;
    // 遍历数组中的每个元素
    for (const auto& elem : *expanding_array_with_opt_elem) {
      // 将元素的值转换为字符串并加入向量（若有值则转换为字符串，否则加入"None"）
      str_array.emplace_back(
          elem.has_value() ? c10::str(elem.value()) : "None");
    }
    // 输出字符串向量作为数组引用
    stream << at::ArrayRef<std::string>(str_array);
  }
  // 返回输出流对象
  return stream;
}

} // namespace torch
```