# `.\pytorch\torch\csrc\lazy\core\permutation_util.h`

```
#pragma once

#include <c10/util/ArrayRef.h>
#include <c10/util/Exception.h>
#include <c10/util/irange.h>

#include <vector>

namespace torch {
namespace lazy {

// 返回输入排列的逆排列
TORCH_API std::vector<int64_t> InversePermutation(
    c10::ArrayRef<int64_t> input_permutation);

// 检查给定的数组是否是排列
TORCH_API bool IsPermutation(c10::ArrayRef<int64_t> permutation);

// 使用排列指定的顺序收集输入。对于每个 i，output[i] = dimensions[permutation[i]]。
// 给定的排列必须与输入的尺寸相同。
template <typename Container>
std::vector<typename Container::value_type> PermuteDimensions(
    c10::ArrayRef<int64_t> permutation,
    const Container& dimensions) {
  using T = typename Container::value_type;
  // 检查尺寸是否匹配，如果不匹配，抛出错误信息
  TORCH_CHECK(
      dimensions.size() == permutation.size(),
      "Invalid permutation specified. dimensions.size() != permutation.size()  (",
      dimensions.size(),
      " vs. ",
      permutation.size(),
      ")");
  // 检查排列是否有效，如果不是有效的排列，抛出错误信息
  TORCH_CHECK(
      IsPermutation(permutation),
      "Invalid permutation specified. Permutation is not a permutation");
  
  // 创建一个与 dimensions 同样大小的输出向量
  std::vector<T> output(dimensions.size());
  
  // 对于每个索引 i，根据排列将 dimensions 中的元素复制到输出向量中
  for (const auto i : c10::irange(permutation.size())) {
    output[i] = dimensions[permutation[i]];
  }
  
  // 返回构建好的输出向量
  return output;
}

} // namespace lazy
} // namespace torch


这些注释解释了每行代码的作用和目的，确保了读者能够理解每个函数和检查的含义以及它们的用途。
```