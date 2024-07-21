# `.\pytorch\torch\csrc\lazy\core\permutation_util.cpp`

```py
// 定义命名空间 torch::lazy，并实现其中的函数
namespace torch {
namespace lazy {

// 根据输入的排列，返回其逆排列
std::vector<int64_t> InversePermutation(
    c10::ArrayRef<int64_t> input_permutation) {
  
  // 检查输入排列是否为合法的排列
  TORCH_CHECK(IsPermutation(input_permutation));

  // 创建与输入排列大小相同的输出排列，初始值为 -1
  std::vector<int64_t> output_permutation(input_permutation.size(), -1);
  
  // 遍历输入排列的每个元素及其索引
  for (const auto i : c10::irange(input_permutation.size())) {
    // 将输入排列中的每个元素及其索引存入输出排列中
    output_permutation.at(input_permutation.at(i)) = i;
  }
  
  // 返回逆排列
  return output_permutation;
}

// 检查给定的数组是否为一个合法的排列
bool IsPermutation(c10::ArrayRef<int64_t> permutation) {
  
  // 创建一个理想的排列，即 [0, 1, 2, ..., permutation.size()-1]
  std::vector<int64_t> trivial_permutation(permutation.size());
  std::iota(trivial_permutation.begin(), trivial_permutation.end(), 0);
  
  // 判断给定的排列是否与理想排列是一种排列关系
  return std::is_permutation(
      permutation.begin(), permutation.end(), trivial_permutation.begin());
}

} // namespace lazy
} // namespace torch
```