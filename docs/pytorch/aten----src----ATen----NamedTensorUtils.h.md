# `.\pytorch\aten\src\ATen\NamedTensorUtils.h`

```py
#pragma once
#include <ATen/NamedTensor.h>
#include <ATen/TensorNames.h>
#include <ATen/WrapDimUtilsMulti.h>

#include <ATen/core/DimVector.h>
#include <ATen/core/Tensor.h>
#include <functional>

namespace at {

using NameVector = SmallVector<Dimname, kDimVectorStaticSize>;

// 检查给定的张量列表中是否有任何张量具有命名维度
inline bool has_names(const ITensorListRef& tensors) {
  return std::any_of(tensors.begin(), tensors.end(), [](const Tensor& t) {
    return t.has_names();
  });
}

// 将命名维度转换为位置索引。如果 `dim` 不能用于引用张量的任何维度，则报错。
TORCH_API int64_t dimname_to_position(const Tensor& tensor, Dimname dim);

// 将一组命名维度转换为位置索引的向量
TORCH_API std::vector<int64_t> dimnames_to_positions(
    const Tensor& tensor,
    DimnameList dims);

// 统一两个命名维度列表以生成第三个列表。这对于实现二进制广播操作（如加法）的命名推断规则非常有用。
//
// 主要有三个约束条件：
// 1) 检查匹配：名称必须从右侧按位置匹配。
// 2) 检查不对齐：如果名称 `n` 在 `names` 中，则它必须在 `other` 中与右侧相同的索引处出现。
// 3) 输出名称是通过逐个从右侧统一名称得到的。
TORCH_API std::vector<Dimname> unify_from_right(
    DimnameList names,
    DimnameList other,
    const char* action = "broadcast");

// 报告不支持的命名维度重载，用于在不支持命名维度传递给操作时抛出错误。
[[noreturn]] inline void reportNYIDimnameOverload(const char* op_name) {
  TORCH_CHECK(
      false,
      op_name,
      ": You passed a dimname (string) to this op in place of a dimension "
      "index but it does not yet support this behavior. Please pass a dimension "
      "index to work around this.");
}

// [NOTE] 编写命名推断规则
//
// 支持命名张量的操作要么由支持命名张量的操作组成，要么实现某种命名推断规则。一个实现自己命名推断规则的操作通常如下所示：
//
// Tensor op(...) {
//   perform_shape_checks(...);
//   # (1)
//   auto maybe_outnames = compute_outnames(...);
//   auto result = [&]() {
//     NoNamesGuard guard;
//     return op_impl(...);
//   }();
//   # (2)
//   propagate_names_if_nonempty(result, maybe_outnames);
//
// 每个操作都有 (1) 计算输出名称步骤和 (2) 传播名称步骤。
//
// compute_outnames 负责检查输入名称匹配并确定输出名称应该是什么。它返回：
// - {}（如果所有输入张量都没有命名）
// - 非空的输出名称。
//
// propagate_names_if_nonempty 如果存在输出名称，则传播到结果张量。
//
// {} 情况是一种优化；如果用户不使用命名张量，他们不会因此支付性能成本。

namespace namedinference {

// 如果 `maybe_names` 不为空且非空，则将 `names` 传播到 `result` 张量。
const Tensor& propagate_names_if_present_and_nonempty(
    const Tensor& result,
    std::optional<DimnameList> maybe_names,
    bool validate_names = false);
// Propagates `names` to `result` if `names` is not empty.
// `names` 可能为空；参见 [NOTE] 写入名称推断规则
// 如果 `names` 不为空，则 `names.size()` 应等于 `result.dim()`。
// 当存在疑惑时，应该使用此重载而不是其他重载。
TORCH_API const Tensor& propagate_names_if_nonempty(
    const Tensor& result,
    DimnameList maybe_names,
    bool validate_names = false);

// 将 `names` 传播到 `result`。仅当我们确定有名称需要传播（即 `names` 不为空）时使用此函数。
TORCH_API const Tensor& propagate_names(
    const Tensor& result,
    DimnameList names,
    bool validate_names = false);

// 将所有名称从 `src` 传播到 `result`。
TORCH_API void propagate_names(const Tensor& result, const Tensor& src);

// 传播除了在 `excluded_idxs` 中指定的索引之外的所有名称。
TORCH_API void propagate_names_except(
    const Tensor& result,
    const Tensor& src,
    IntArrayRef excluded_idxs);

// 用于具有 `keepdim` 参数的缩减操作。
TORCH_API void propagate_names_for_reduction(
    const Tensor& result,
    const Tensor& src,
    IntArrayRef excluded_idxs,
    bool keepdim);

// 为扩展操作传播名称。
TORCH_API void propagate_names_for_expand(
    const Tensor& result,
    const Tensor& self);

// 计算 concatenate 操作的输出名称。
TORCH_API std::vector<Dimname> compute_cat_outnames(
    const MaterializedITensorListRef& tensors);

// 计算 broadcast 操作的输出名称。
TORCH_API std::vector<Dimname> compute_broadcast_outnames(
    const Tensor& self,
    const Tensor& other);

// 根据操作名称计算 broadcast 到目标张量的输出名称。
TORCH_API std::vector<Dimname> broadcast_to_outnames(
    const Tensor& tensor,
    const Tensor& reference_tensor,
    const char* op_name);

// 计算矩阵乘法操作的输出名称。
TORCH_API std::vector<Dimname> compute_matmul_outnames(
    const Tensor& self,
    const Tensor& other);

// 计算 cdist 操作的输出名称。
TORCH_API std::vector<Dimname> compute_cdist_outnames(
    const Tensor& self,
    const Tensor& other);

// 计算 bmm 操作的输出名称。
TORCH_API std::vector<Dimname> compute_bmm_outnames(
    const Tensor& result,
    const Tensor& self,
    const Tensor& other);

// 计算 squeeze 操作的输出名称。
TORCH_API std::vector<Dimname> compute_squeeze_outnames(const Tensor& tensor);
TORCH_API std::vector<Dimname> compute_squeeze_outnames(
    const Tensor& tensor,
    std::bitset<dim_bitset_size> dims);

// 计算对角线操作的输出名称。
std::vector<Dimname> compute_diagonal_outnames(
    const Tensor& tensor,
    int64_t dim1,
    int64_t dim2);

// 对于 Legacy TH/THC 代码的 TensorImpl* 重载。请谨慎使用这些函数。
TORCH_API TensorImpl* propagate_names_if_nonempty(
    TensorImpl* result,
    DimnameList maybe_names,
    bool validate_names = false);

TORCH_API TensorImpl* propagate_names(
    TensorImpl* result,
    DimnameList names,
    bool validate_names = false);

// 将名称从 `src` 传播到 `result` 的 TensorImpl* 版本。
TORCH_API void propagate_names(TensorImpl* result, /*const */ TensorImpl* src);

// TensorBase& 版本的内联函数，用于传播名称。
TORCH_API inline void propagate_names(
    const TensorBase& result,
    DimnameList names,
    bool validate_names = false) {
  propagate_names(result.unsafeGetTensorImpl(), names, validate_names);
}

// TensorBase& 版本的内联函数，用于传播非空名称。
TORCH_API inline void propagate_names_if_nonempty(
    const TensorBase& result,
    DimnameList names,
    # 定义一个布尔变量 validate_names，并初始化为 false
    bool validate_names = false) {
  # 调用函数 propagate_names_if_nonempty 来传播名称（如果名称不为空的话），
  # 传递 result.unsafeGetTensorImpl() 作为参数，同时传递 names 和 validate_names 作为其他参数
  propagate_names_if_nonempty(
      result.unsafeGetTensorImpl(), names, validate_names);
// 结束了一个名为 `propagate_names` 的内联函数的定义，用于在 Torch API 中传播张量的名称信息
TORCH_API inline void propagate_names(
    const TensorBase& result,
    const TensorBase& src) {
  // 调用底层实现的 unsafeGetTensorImpl 方法获取 result 和 src 的张量实现
  propagate_names(result.unsafeGetTensorImpl(), src.unsafeGetTensorImpl());
}

// 使用 m1 和 m2 进行矩阵乘法，并加上偏置 bias，传播名称信息
// 返回一个 Dimname 向量，描述输出张量的维度名称
TORCH_API std::vector<Dimname> propagate_names_for_addmm(
    const Tensor& m1,
    const Tensor& m2,
    const Tensor& bias);

// 使用 mat 和 vec 进行矩阵向量乘法，并加上偏置 bias，传播名称信息
// 返回一个 Dimname 向量，描述输出张量的维度名称
TORCH_API std::vector<Dimname> propagate_names_for_addmv(
    const Tensor& mat,
    const Tensor& vec,
    const Tensor& bias);

// 检查两个张量实现的维度名称是否相等
TORCH_API void check_names_for_dot(TensorImpl* vec1, TensorImpl* vec2);

// 计算 BADDMM 操作的输出张量的维度名称
// result 是结果张量，self 和 other 是输入张量，bias 是偏置张量
// 返回一个 Dimname 向量，描述输出张量的维度名称
TORCH_API std::vector<Dimname> compute_baddbmm_outnames(
    const Tensor& result,
    const Tensor& self,
    const Tensor& other,
    const Tensor& bias);

// 检查两个张量实现的维度名称是否相等
TORCH_API bool are_names_equal(TensorImpl* self, TensorImpl* other);

// 结束了 namedinference 命名空间的定义，用于 Torch 的命名推断
} // namespace namedinference

// 结束了 at 命名空间的定义，命名空间用于处理张量运算
} // namespace at
```