# `.\pytorch\torch\csrc\lazy\core\helpers.h`

```py
#pragma once

#include <c10/core/Scalar.h>  // 包含标量相关的头文件
#include <c10/util/BFloat16.h>  // 包含BFloat16相关的头文件
#include <c10/util/Half.h>  // 包含Half相关的头文件
#include <c10/util/Optional.h>  // 包含Optional相关的头文件
#include <torch/csrc/lazy/core/permutation_util.h>  // 包含置换工具函数的头文件
#include <torch/csrc/lazy/core/shape.h>  // 包含形状相关的头文件
#include <torch/csrc/lazy/core/util.h>  // 包含通用工具函数的头文件

#include <complex>  // 复数库
#include <functional>  // 函数库
#include <tuple>  // 元组库
#include <vector>  // 向量库

// TODO: Consolidate this file with util.h

namespace torch {
namespace lazy {

// Converts an iterable container to a vector of int64's.
// 将可迭代容器转换为 int64_t 类型的向量
template <typename S>
static std::vector<int64_t> ToI64Vector(const S& input) {
  return ToVector<int64_t>(input);  // 调用 ToVector 将 input 转换为 int64_t 类型向量
}

// Creates a set of dimension by dropping the drop_dims ones.
// 通过丢弃指定维度来创建一个新的维度集合
TORCH_API std::vector<int64_t> DropDimensions(
    c10::ArrayRef<int64_t> sizes,  // 原始尺寸数组的引用
    c10::ArrayRef<int64_t> drop_dims);  // 要丢弃的维度数组的引用

// Get the canonical dimension index in the [0, rank) interval. Negative
// indices are interpreted as follows: -1 is rank-1, -2 is rank-2 etc.
// 获取规范化的维度索引，范围在 [0, rank) 内。负数索引的解释如下：-1 表示 rank-1，-2 表示 rank-2，依此类推。
TORCH_API int64_t GetCanonicalDimensionIndex(int64_t dim, int64_t rank);

// Same as above, for multiple dimensions.
// 与上面类似，针对多个维度
TORCH_API std::vector<int64_t> GetCanonicalDimensionIndices(
    c10::ArrayRef<int64_t> dimensions,  // 维度数组的引用
    int64_t rank);  // 维度的总数

// Returns the canonical position in the dim dimension, handling negative
// values for the position.
// 返回在指定维度 dim 中的规范化位置，处理负值的情况
TORCH_API int64_t GetCanonicalPosition(
    c10::ArrayRef<int64_t> dimensions,  // 维度数组的引用
    int64_t dim,  // 指定的维度
    int64_t pos);  // 位置索引

// Creates a transposition from the given input and dimensions.
// 根据给定的输入和维度创建一个置换
TORCH_API std::vector<int64_t> MakeTransposePermutation(
    int64_t dim0,  // 第一个维度
    int64_t dim1,  // 第二个维度
    int64_t rank);  // 总维度数

// Calculates the protomoted shape to which the input shapes should be
// broadcasted for an elementwise operation. The size of the common dimensions
// (2,3,4 for shape1, and 0,1,2 for shape2) must either match, or either one
// of the two be 1.
// 计算元素级操作中输入形状应该广播到的推广形状。共同维度的大小（shape1 的 2,3,4 和 shape2 的 0,1,2）必须匹配，或者其中一个为 1。
// 示例：
//   shape1       = [9, 7, 6, 1, 2]
//   shape2       =       [6, 5, 2]
//   result_shape = [9, 7, 6, 5, 2]
TORCH_API std::vector<int64_t> GetPromotedShape(
    c10::ArrayRef<int64_t> shape1_dims,  // shape1 的维度数组引用
    c10::ArrayRef<int64_t> shape2_dims);  // shape2 的维度数组引用

// Returns the promoted shape for binary operations between two shapes.
// 返回两个形状进行二元操作时的推广形状
TORCH_API Shape GetPromotedBinaryOpShape(const Shape& shape1, const Shape& shape2);

// Splits a string into a vector of strings using a delimiter.
// 使用分隔符将字符串拆分为字符串向量
TORCH_API std::vector<std::string> StrSplit(c10::string_view text, char delim);

} // namespace lazy
} // namespace torch
```