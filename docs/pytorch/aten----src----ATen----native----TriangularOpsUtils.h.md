# `.\pytorch\aten\src\ATen\native\TriangularOpsUtils.h`

```
/*
 * 给定批次的具有任意批次维度的矩阵，
 * 计算 Triu 和 Tril 的批次数。这忽略了步幅为 0 的维度。
 */
static inline int64_t batchCountTrilTriu(const Tensor& batched_matrices) {
  // 初始化结果为 1
  int64_t result = 1;
  // 遍历维度，但跳过最后两个维度
  for (int64_t i = 0; i < batched_matrices.ndimension() - 2; i++) {
    // 如果当前维度的步幅不为 0
    if (batched_matrices.stride(i) != 0) {
      // 结果乘以当前维度的大小
      result *= batched_matrices.size(i);
    }
  }
  // 返回计算结果
  return result;
}

/* 检查 Triu 和 Tril 实现的必要属性，因此命名为这样。
 * 在维度大于 4 的张量上检查批次连续性。
 * 连续的张量和小于 3 维度的张量通过此检查。
 */
static inline std::tuple<bool, Tensor> checkTrilTriuBatchContiguous(const Tensor& tensor, bool allow_zero_stride) {
  // 完全连续性是最理想的属性，因此如果张量是连续的，则返回 true
  if (tensor.is_contiguous()) {
    // 计算默认步幅以便与尺寸相匹配的批次矩阵
    auto default_strides_for_size = batched_matrix_contiguous_strides(tensor.sizes());
    // 如果张量的步幅与默认步幅相同
    if (tensor.strides() == default_strides_for_size) {
      return std::make_tuple(true, tensor);
    } else {
      // 否则返回一个通过自定义步幅的新张量
      return std::make_tuple(false, tensor.as_strided(tensor.sizes(), default_strides_for_size));
    }
  }

  int64_t dims = tensor.dim();

  // 小于 4 维度的张量默认处理
  if (allow_zero_stride && dims <= 3) {
    return std::make_tuple(true, tensor);
  }

  // 期望的步幅为张量最后两个维度的大小乘积
  int64_t expected_stride = tensor.size(-1) * tensor.size(-2);
  // 从倒数第三个维度开始向前遍历
  for (int64_t i = dims - 3; i >= 0; i--) {
    // 跳过不重要的维度
    if (allow_zero_stride && i == 0 && (tensor.stride(i) == 0 || tensor.size(i) == 1)) {
      continue;
    }
    // 如果当前维度的步幅与期望的步幅不同
    if (expected_stride != tensor.stride(i)) {
      // 返回一个连续的张量
      return std::make_tuple(false, tensor.contiguous());
    }
    // 更新期望的步幅
    expected_stride *= tensor.size(i);
  }
  // 返回连续性检查的结果
  return std::make_tuple(true, tensor);
}
```