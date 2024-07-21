# `.\pytorch\aten\src\ATen\native\RowwisePrune.cpp`

```py
// 定义宏以限制仅在方法操作器中使用
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS

// 包含张量的头文件
#include <ATen/core/Tensor.h>
// 包含分发函数头文件
#include <ATen/Dispatch.h>
// 包含范围迭代工具
#include <c10/util/irange.h>

// 如果未定义每个操作符的头文件，则包含函数和本地函数头文件，否则包含特定头文件
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_rowwise_prune_native.h>
#include <ATen/ops/empty.h>
#endif

// 命名空间：at::native
namespace at::native {

// 未命名命名空间下的模板函数，用于辅助行压缩操作
namespace {

// 函数模板：_rowwise_prune_helper
template <typename input_t>
std::tuple<Tensor, Tensor> _rowwise_prune_helper(
      const Tensor& weights, const Tensor& mask,
      ScalarType compressed_indices_dtype) {
  // 计算非掩码行数
  int num_non_masked_rows = 0;
  // 创建掩码的连续版本，并获取其数据指针
  auto mask_contig = mask.contiguous();
  auto mask_data = mask_contig.data_ptr<bool>();
  // 遍历掩码中的元素，统计非掩码行数
  for (const auto i : c10::irange(mask.numel())) {
    num_non_masked_rows += (((mask_data[i] == true)) ? 1 : 0);
  }
  // 获取权重张量的列数
  int num_cols = weights.size(1);
  // 创建一个空的二维张量，以存储压缩后的权重
  auto pruned_2d_tensor = at::empty({num_non_masked_rows, num_cols},
      weights.options());
  // 创建一个空的索引映射张量，以存储压缩索引
  auto compressed_indices_mapping = at::empty({mask.numel()},
      compressed_indices_dtype);
  
  // 使用分发策略处理所有数据类型，并执行行压缩操作
  AT_DISPATCH_ALL_TYPES_AND2(at::ScalarType::Half,
                             at::ScalarType::BFloat16,
                             weights.scalar_type(),
                            "rowwise_prune_helper", [&]() {
    // 获取压缩后的二维张量数据指针和压缩索引映射数据指针
    auto* pruned_2d_tensor_data = pruned_2d_tensor.data_ptr<scalar_t>();
    auto compressed_indices_mapping_data =
        compressed_indices_mapping.data_ptr<input_t>();
    auto weights_data = weights.data_ptr<scalar_t>();
    int last_row_kept = 0;
    // 遍历掩码，根据掩码状态复制权重数据到压缩后的张量中，并更新压缩索引映射
    for (const auto i : c10::irange(mask.numel())) {
      if (mask_data[i]) {
        memcpy(pruned_2d_tensor_data + last_row_kept * num_cols,
              weights_data + i * num_cols,
              num_cols * sizeof (scalar_t));
        compressed_indices_mapping_data[i] = last_row_kept;
        last_row_kept++;
      } else {
        compressed_indices_mapping_data[i] = -1;
      }
    }
  });

  // 返回压缩后的二维张量和压缩索引映射张量的元组
  return std::tuple<Tensor, Tensor>(pruned_2d_tensor,
      compressed_indices_mapping);
}

} // namespace

// 该操作符通过重要性指示器掩码向权重矩阵引入稀疏性。
//
// 如果特定行的掩码值为1（True），则认为该行重要且不被压缩；否则不重要。
//
// 该操作符不会直接将被压缩的行置零。相反，它返回一个元组，其中包含一个被压缩的权重张量，
// 以及一个可用于查找原始行在压缩权重张量中位置的映射。我们将这个映射称为“压缩索引映射”。

// “压缩索引映射”是一个一维张量，包含每个原始权重行的一个条目。数组索引是原始未压缩权重张量的索引，
// 值是重新映射后的压缩权重张量中的索引。如果索引处的值为-1，则表示对应行已从原始权重张量中被压缩。

// 参数：
// 'weights' - 需要进行稀疏化处理的二维权重矩阵。
// 'mask' - 表示每行重要性的一维布尔张量。值为1表示保留该行，值为0表示裁剪该行。
//
// 返回：
// 包含两个张量的元组，
// 1. 一个稀疏化处理后仅保留的权重张量。
// 2. 一个一维张量，包含原始权重行与稀疏化处理后权重张量对应行的映射关系。
std::tuple<Tensor, Tensor> _rowwise_prune(const Tensor& weights,
                                          const Tensor& mask,
                                          ScalarType compressed_indices_dtype) {
  TORCH_CHECK(weights.ndimension() == 2,
      "'weights' should have 2 dimensions.");  // 检查权重张量是否为二维

  TORCH_CHECK(
    mask.numel() == weights.size(0),
    "Number of elements in 'mask' should be equivalent to the "
    "number of rows in 'weights'."
  )  // 检查mask张量元素数量是否等于权重张量的行数

  TORCH_CHECK(
      compressed_indices_dtype == ScalarType::Int ||
      compressed_indices_dtype == ScalarType::Long,
      "compressed_indices_dtype should be either int(int32) or long(int64)."
  )  // 检查compressed_indices_dtype是否为int（int32）或long（int64）

  // 根据compressed_indices_dtype的类型选择相应的辅助函数进行稀疏化处理
  if (compressed_indices_dtype == at::ScalarType::Int) {
    return _rowwise_prune_helper<int32_t>(weights, mask,
                                          compressed_indices_dtype);
  }
  return _rowwise_prune_helper<int64_t>(weights, mask,
                                        compressed_indices_dtype);
}

} // namespace at::native
```