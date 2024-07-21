# `.\pytorch\torch\csrc\lazy\core\ops\utils.cpp`

```py
namespace torch {
namespace lazy {

// 检查给定的步长数组是否被支持，返回是否支持的布尔值
bool StrideIsSupported(c10::ArrayRef<int64_t> stride) {
  // 将步长数组复制到新的向量，并进行排序
  std::vector<int64_t> sorted_stride(stride.begin(), stride.end());
  std::sort(sorted_stride.begin(), sorted_stride.end());
  // 如果步长数组为空，或者排序后的最小步长为1，则返回真
  return stride.empty() || sorted_stride.front() == 1;
}

// 根据步长数组生成一个用于排序的排列数组
std::vector<int64_t> GetArrayStridePermutation(c10::ArrayRef<int64_t> stride) {
  // 创建一个从0到步长数组大小的初始排列
  std::vector<int64_t> permutation = Iota<int64_t>(stride.size());
  // 使用步长数组的值对排列进行排序
  std::sort(permutation.begin(), permutation.end(), [&](int64_t a, int64_t b) {
    return stride[a] > stride[b];
  });
  return permutation;
}

// 根据给定的形状、偏移量、维度1和维度2，生成对角线形状
Shape MakeDiagonalShape(
    const Shape& shape,
    int64_t offset,
    int64_t dim1,
    int64_t dim2) {
  std::vector<int64_t> dimensions;
  // 遍历形状的维度
  for (const auto dim : c10::irange(shape.dim())) {
    // 如果当前维度不是dim1和dim2，则将其大小添加到维度向量中
    if (dim != dim1 && dim != dim2) {
      dimensions.push_back(shape.size(dim));
    }
  }
  int64_t dsize = 0;
  // 根据偏移量的正负值计算对角线的大小
  if (offset >= 0) {
    dsize = std::max<int64_t>(
        std::min(shape.size(dim1), shape.size(dim2) - offset), 0);
  } else {
    dsize = std::max<int64_t>(
        std::min(shape.size(dim1) + offset, shape.size(dim2)), 0);
  }
  dimensions.push_back(dsize);
  return Shape(shape.scalar_type(), dimensions);
}

// 根据源形状和排列数组生成一个置换形状
Shape MakePermuteShape(
    const Shape& source_shape,
    c10::ArrayRef<int64_t> permutation) {
  return Shape(
      source_shape.scalar_type(),
      PermuteDimensions(permutation, source_shape.sizes()));
}

// 根据给定的形状、维度、起始、结束和步长生成选择形状
Shape MakeSelectShape(
    const Shape& shape,
    int64_t dim,
    int64_t start,
    int64_t end,
    int64_t stride) {
  // 计算有效步长
  int64_t effective_stride = GetStride(start, end, stride);
  // 创建一个选择形状的副本
  Shape select_shape(shape);
  // 设置选择形状指定维度的大小
  select_shape.set_size(
      dim, (end - start + effective_stride - 1) / effective_stride);
  return select_shape;
}

// 获取有效的步长
int64_t GetStride(int64_t start, int64_t end, int64_t stride) {
  if (stride == 0) {
    // 如果步长为0，确保起始和结束位置相等，并将步长设置为1
    TORCH_CHECK_EQ(start, end);
    stride = 1;
  }
  return stride;
}

// 构建被挤压维度的输出维度
std::vector<int64_t> BuildSqueezedDimensions(
    c10::ArrayRef<int64_t> dimensions,
    int64_t squeeze_dim) {
  std::vector<int64_t> output_dimensions;
  // 遍历维度数组
  for (const auto i : c10::irange(dimensions.size())) {
    int64_t dim = dimensions[i];
    // 如果维度不为1或者不是挤压的维度，则将其添加到输出维度向量中
    if (dim != 1 ||
        (static_cast<int64_t>(i) != squeeze_dim && squeeze_dim >= 0)) {
      output_dimensions.push_back(dim);
    }
  }
  return output_dimensions;
}

// 构建未被挤压维度的输出维度
std::vector<int64_t> BuildUnsqueezedDimensions(
    c10::ArrayRef<int64_t> dimensions,
    int64_t squeeze_dim) {
  std::vector<int64_t> output_dimensions(
      dimensions.cbegin(), dimensions.cend());
  // 在指定的挤压维度处插入大小为1的维度
  output_dimensions.insert(output_dimensions.begin() + squeeze_dim, 1);
  return output_dimensions;
}

} // namespace lazy
} // namespace torch
```