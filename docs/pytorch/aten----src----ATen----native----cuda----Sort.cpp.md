# `.\pytorch\aten\src\ATen\native\cuda\Sort.cpp`

```
// 定义宏，用于在头文件中声明仅使用方法操作符
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
// 包含 CUDA 排序所需的头文件
#include <ATen/native/cuda/Sort.h>
// 包含张量相关的核心头文件
#include <ATen/core/Tensor.h>
// 包含张量操作中的扩展工具函数
#include <ATen/ExpandUtils.h>
// 包含张量内存重叠检测相关的头文件
#include <ATen/MemoryOverlap.h>
// 包含张量工具函数的头文件
#include <ATen/TensorUtils.h>
// 包含张量维度包装工具函数的头文件
#include <ATen/WrapDimUtils.h>
// 包含张量排序相关的头文件
#include <ATen/native/Sorting.h>
// 包含张量调整大小相关的头文件
#include <ATen/native/Resize.h>

// 如果未定义每个操作符的头文件，则包含所有操作符的头文件
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
// 否则，仅包含每个操作符的特定头文件
#else
#include <ATen/ops/arange.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/empty_strided.h>
#include <ATen/ops/sort_native.h>
#include <ATen/ops/zeros.h>
#endif

// 包含 C++ 标准库中的数值极限定义
#include <limits>

// at 命名空间下的 native 命名空间
namespace at::native {

// 推断密集步长的最后维度
std::vector<int64_t> infer_dense_strides_dim_last(const Tensor & self, int64_t dim);

// 用索引填充切片的值
void fillSliceWithIndex(const Tensor& t, int dim) {
  // 如果张量非空
  if (t.numel()) {
    // 创建大小与张量维度相同的大小向量
    auto sizes = DimVector(t.dim(), 1);
    // 设置指定维度的大小为张量当前维度的大小
    sizes[dim] = t.sizes()[dim];
    // 使用张量选项创建一个范围内的张量
    auto range = at::arange(t.sizes()[dim], t.options());
    // 将范围视图调整为指定大小的视图
    auto rangeview = range.view(sizes);
    // 将切片填充为索引范围的值
    t.copy_(rangeview);
  }
}

// 在 CUDA 内核中执行分段排序
void sort_cuda_kernel(
    const TensorBase& self_base,
    const TensorBase& values_base,
    const TensorBase& indices_base,
    int64_t dim,
    bool descending,
    bool stable) {
  // 此算法始终稳定

  // 宏用于将 `TensorBase` 转换为 `Tensor`，无需增加引用计数
#define TOTENSOR(BASE, VAR)           \
  OptionalTensorRef opt_##BASE(BASE); \
  const Tensor& VAR = *opt_##BASE;

  // 将 TensorBase 转换为 Tensor
  // 从这一点开始，我们将需要 Tensor 的方法
  TOTENSOR(self_base, self);
  TOTENSOR(values_base, values);
  TOTENSOR(indices_base, indices);

  // 检查被排序维度的元素数不超过 INT_MAX
  TORCH_CHECK(self.sizes()[dim] <= std::numeric_limits<int>::max(),
    "The dimension being sorted can not have more than INT_MAX elements.");

  // 检查是否支持布尔类型的排序
  const auto self_dtype = self.dtype();
  TORCH_CHECK(self_dtype != ScalarType::Bool,
    "Sort currently does not support bool dtype on CUDA.");
  // 检查是否支持复数类型的排序
  TORCH_CHECK(self_dtype != ScalarType::ComplexFloat && self_dtype != ScalarType::ComplexDouble,
    "Sort currently does not support complex dtypes on CUDA.");

  // 对于较小的输入大小且无需稳定性时使用原地排序算法
  if (should_use_small_sort(self, dim)) {
    // 用索引填充切片
    fillSliceWithIndex(indices, dim);

    // 将未排序的输入复制到输出
    values.copy_(self);

    // 使用支持任意布局的原地 k/v 内核进行排序
    sortKeyValueInplace(values, indices, dim, descending, stable);
    return;
  }

  // 如果输入是非重叠且密集的，且指定维度的步长为 1，则直接使用 self
  Tensor self_;
  bool newself = false;
  if (self.is_non_overlapping_and_dense() && self.stride(dim) == 1) {
    self_ = self;
  } else {
    // 推断在最后一个维度上的密集步长
    auto new_strides_unsort = infer_dense_strides_dim_last(self, dim);
  // 创建一个新的张量 `self_`，使用指定的步幅 `new_strides_unsort`，并与当前张量 `self` 共享存储空间
  self_ = at::empty_strided(self.sizes(), new_strides_unsort, self.options());
  // 将当前张量 `self` 的数据复制到新创建的张量 `self_` 中
  self_.copy_(self);
  // 标记已经创建了新的张量
  newself = true;
}

c10::MaybeOwned<Tensor> values_tmp, indices_tmp;
// 检查 `values` 张量与 `self_` 张量的步幅是否相同，如果相同并且是新创建的张量或者二者不重叠，则共享存储空间
if (values.strides() == self_.strides() && (newself || get_overlap_status(self, values) == MemOverlapStatus::No)) {
  // 如果条件成立，将 `values_tmp` 设置为共享 `values` 张量的引用
  values_tmp = c10::MaybeOwned<Tensor>::borrowed(values);
} else {
  // 否则，创建一个与 `self_` 张量相同形状和步幅的新张量，作为 `values_tmp`
  values_tmp = c10::MaybeOwned<Tensor>::owned(
      at::empty_strided(self_.sizes(), self_.strides(), self_.options()));
}

// 检查 `indices` 张量与 `self_` 张量的步幅是否相同
if (indices.strides() != self_.strides()) {
  // 如果步幅不同，创建一个与 `self_` 张量相同形状和步幅的新张量，数据类型为 `kLong`，作为 `indices_tmp`
  indices_tmp = c10::MaybeOwned<Tensor>::owned(
      at::empty_strided(self_.sizes(), self_.strides(), self_.options().dtype(kLong)));
} else {
  // 否则，将 `indices_tmp` 设置为共享 `indices` 张量的引用
  indices_tmp = c10::MaybeOwned<Tensor>::borrowed(indices);
}

// 调用一个稳定排序的核函数，对 `self_` 张量沿指定维度 `dim` 进行排序，根据 `descending` 确定降序或升序排序，使用 `values_tmp` 和 `indices_tmp`
launch_stable_sort_kernel(self_, dim, descending, *values_tmp, *indices_tmp);

// 如果 `values_tmp` 没有与 `values` 张量共享存储空间，则将 `values_tmp` 的数据复制回 `values`
if (!values_tmp->is_same(values)) {
  values.copy_(*values_tmp);
}
// 如果 `indices_tmp` 没有与 `indices` 张量共享存储空间，则将 `indices_tmp` 的数据复制回 `indices`
if (!indices_tmp->is_same(indices)) {
  indices.copy_(*indices_tmp);
}
}

// TODO: 当我们开始使用 REGISTER_HIP_DISPATCH 时，应该相应地处理此处的情况，
// 因为在这个 cpp 文件中 REGISTER_DISPATCH 将无法工作。
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CUDA_DISPATCH(sort_stub, &sort_cuda_kernel);

}  // namespace at::native
```