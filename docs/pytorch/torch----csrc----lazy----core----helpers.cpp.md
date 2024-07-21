# `.\pytorch\torch\csrc\lazy\core\helpers.cpp`

```
// 包含 Torch 核心库中的辅助函数头文件
#include <torch/csrc/lazy/core/helpers.h>

// 包含 C10 库中的 Half 类和 irange 函数头文件
#include <c10/util/Half.h>
#include <c10/util/irange.h>
// 包含 Torch 核心库中的张量工具头文件
#include <torch/csrc/lazy/core/tensor_util.h>

// 包含标准库头文件
#include <limits>

// Torch 命名空间开始
namespace torch {
namespace lazy {

// 函数：删除给定维度列表中的维度
std::vector<int64_t> DropDimensions(
    c10::ArrayRef<int64_t> sizes,     // 输入张量的维度列表
    c10::ArrayRef<int64_t> drop_dims) {  // 要删除的维度列表
  std::vector<int64_t> new_dims;  // 存储删除后的维度列表
  size_t drop_index = 0;  // 记录当前处理的要删除维度的索引
  for (const auto i : c10::irange(sizes.size())) {  // 遍历输入张量的维度
    if (drop_index < drop_dims.size() &&  // 如果还有要删除的维度
        static_cast<int64_t>(i) == drop_dims[drop_index]) {
      ++drop_index;  // 更新要删除的维度索引
    } else {
      new_dims.push_back(sizes[i]);  // 将保留的维度添加到新维度列表中
    }
  }
  TORCH_CHECK(drop_index == drop_dims.size());  // 检查所有要删除的维度是否处理完毕
  return new_dims;  // 返回删除指定维度后的新维度列表
}

// 函数：获取规范化后的维度索引
int64_t GetCanonicalDimensionIndex(int64_t dim, int64_t rank) {
  int64_t min_shape_dim = -rank;  // 计算最小可能的维度值
  int64_t max_shape_dim = rank - 1;  // 计算最大可能的维度值
  TORCH_CHECK(
      min_shape_dim <= dim && dim <= max_shape_dim,  // 检查维度值是否在有效范围内
      "Value out of range (expected to be in range of [",
      min_shape_dim,
      ", ",
      max_shape_dim,
      "], but got ",
      dim,
      ")");
  int64_t dim_index = dim < 0 ? rank + dim : dim;  // 计算规范化后的维度索引
  TORCH_CHECK(dim_index >= 0);  // 检查规范化后的维度索引是否非负
  TORCH_CHECK(dim_index < rank);  // 检查规范化后的维度索引是否小于总维度数
  return dim_index;  // 返回规范化后的维度索引
}

// 函数：获取规范化后的维度索引列表
std::vector<int64_t> GetCanonicalDimensionIndices(
    c10::ArrayRef<int64_t> dimensions,  // 待规范化的维度列表
    int64_t rank) {  // 张量的总维度数
  std::vector<int64_t> canonical_dim_indices;  // 存储规范化后的维度索引列表
  for (int64_t dim : dimensions) {  // 遍历待规范化的维度列表
    canonical_dim_indices.push_back(GetCanonicalDimensionIndex(dim, rank));  // 获取并存储每个维度的规范化索引
  }
  return canonical_dim_indices;  // 返回规范化后的维度索引列表
}

// 函数：获取规范化后的位置索引
int64_t GetCanonicalPosition(
    c10::ArrayRef<int64_t> dimensions,  // 张量的维度列表
    int64_t dim,  // 待规范化的维度
    int64_t pos) {  // 待规范化的位置索引
  dim = GetCanonicalDimensionIndex(dim, dimensions.size());  // 获取维度的规范化索引
  if (pos < 0) {  // 如果位置索引为负数
    pos = GetCanonicalDimensionIndex(pos, dimensions[dim]);  // 获取位置索引的规范化值
  } else {
    pos = std::min<int64_t>(pos, dimensions[dim]);  // 获取位置索引和维度值的较小者
  }
  return pos;  // 返回规范化后的位置索引
}

// 函数：生成用于转置的维度置换列表
std::vector<int64_t> MakeTransposePermutation(
    int64_t dim0,  // 第一个要交换的维度
    int64_t dim1,  // 第二个要交换的维度
    int64_t rank) {  // 张量的总维度数
  int64_t canonical_dim0 = GetCanonicalDimensionIndex(dim0, rank);  // 获取第一个维度的规范化索引
  int64_t canonical_dim1 = GetCanonicalDimensionIndex(dim1, rank);  // 获取第二个维度的规范化索引
  auto permute_dims = Iota<int64_t>(rank);  // 生成一个包含所有维度的索引列表
  std::swap(permute_dims[canonical_dim0], permute_dims[canonical_dim1]);  // 交换指定的两个维度索引
  return permute_dims;  // 返回用于转置的维度置换列表
}

// 函数：获取两个形状维度的推广形状
std::vector<int64_t> GetPromotedShape(
    c10::ArrayRef<int64_t> shape1_dims,  // 第一个形状的维度列表
    c10::ArrayRef<int64_t> shape2_dims) {  // 第二个形状的维度列表
  std::vector<int64_t> dimensions;  // 存储推广后的维度列表

  // 如果第一个形状的维度数大于第二个形状的维度数，将第一个形状的维度填充到 dimensions 中
  if (shape1_dims.size() > shape2_dims.size()) {
    dimensions.insert(
        dimensions.end(),
        shape1_dims.begin(),
        shape1_dims.begin() + (shape1_dims.size() - shape2_dims.size()));
  } else if (shape2_dims.size() > shape1_dims.size()) {
    // 如果第二个形状的维度数大于第一个形状的维度数，暂不处理（需补充完整代码）
  }
    // 将 shape2_dims 的前部分维度插入到 dimensions 的末尾
    dimensions.insert(
        dimensions.end(),
        shape2_dims.begin(),
        shape2_dims.begin() + (shape2_dims.size() - shape1_dims.size()));
    
    
    
    // 对于共同的维度，它们必须匹配，或者其中一个为 1
    size_t min_size = std::min(shape1_dims.size(), shape2_dims.size());
    for (const auto i : c10::irange(min_size)) {
      // 获取 shape1_dims 和 shape2_dims 中的对应维度
      int64_t dim1 = shape1_dims[shape1_dims.size() - min_size + i];
      int64_t dim2 = shape2_dims[shape2_dims.size() - min_size + i];
      // 检查维度是否匹配或者为 1，否则抛出异常
      TORCH_CHECK(
          dim1 == dim2 || dim1 == 1 || dim2 == 1,
          "(",
          c10::Join(", ", shape1_dims),
          ") and (",
          c10::Join(", ", shape2_dims),
          ")");
      // 如果任何一个维度为 0，则在 dimensions 中添加 0；否则添加较大的维度值
      if (dim1 == 0 || dim2 == 0) {
        dimensions.push_back(0);
      } else {
        dimensions.push_back(std::max<int64_t>(dim1, dim2));
      }
    }
    
    
    
    // 返回合并后的维度 dimensions
    return dimensions;
}

// 结束命名空间 'lazy' 和 'torch'

Shape GetPromotedBinaryOpShape(const Shape& shape1, const Shape& shape2) {
  // 返回推广后的操作形状，包括类型和尺寸
  return Shape(
      // 推广操作的类型
      promoteTypes(shape1.scalar_type(), shape2.scalar_type()),
      // 推广操作的形状尺寸
      GetPromotedShape(shape1.sizes(), shape2.sizes()));
}

std::vector<std::string> StrSplit(c10::string_view text, char delim) {
  // 开始位置和结束位置的初始化
  size_t start = 0;
  size_t end = 0;

  // 存储分割后的字符串片段
  std::vector<std::string> tokens;
  // 当还有未处理的文本时循环分割
  while ((start = text.find_first_not_of(delim, end)) != std::string::npos) {
    // 查找下一个分隔符的位置
    end = text.find(delim, start);
    // 提取当前片段并存储
    auto token = text.substr(start, end - start);
    tokens.emplace_back(token.begin(), token.end());
  }
  // 返回所有分割后的字符串片段
  return tokens;
}

} // namespace lazy
} // namespace torch
```