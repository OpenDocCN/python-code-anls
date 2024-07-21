# `.\pytorch\aten\src\ATen\native\cuda\Sorting.cpp`

```py
// 定义宏 TORCH_ASSERT_ONLY_METHOD_OPERATORS，用于限制只包含方法操作符
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS

// 引入相关的头文件，涵盖了 CUDA 实现的排序功能和张量操作
#include <ATen/native/cuda/Sorting.h>
#include <ATen/core/Tensor.h>
#include <ATen/core/NamedTensor.h>
#include <ATen/Context.h>
#include <ATen/TensorUtils.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/WrapDimUtils.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/detail/TensorInfo.cuh>

// 引入排序和归约操作的实用函数和定义
#include <ATen/native/SortingUtils.h>
#include <ATen/native/ReduceOpsUtils.h>

// 根据条件引入不同的头文件集合，影响 AT_PER_OPERATOR_HEADERS 的定义
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/full.h>
#include <ATen/ops/kthvalue_native.h>
#include <ATen/ops/median_native.h>
#include <ATen/ops/nanmedian_native.h>
#include <ATen/ops/where.h>
#endif

// 命名空间 at::native 下的匿名命名空间开始
namespace at::native {
namespace {

// CUDA 实现的 kthvalue_out_impl_cuda 函数，计算第 k 小值并返回值和索引的元组
std::tuple<Tensor&, Tensor&> kthvalue_out_impl_cuda(
    Tensor& values,                   // 输出张量，保存第 k 小值
    Tensor& indices,                  // 输出张量，保存第 k 小值的索引
    const Tensor& self,               // 输入张量，进行计算的原始数据
    int64_t k,                        // 第 k 小值
    int64_t dim_,                     // 操作的维度
    bool keepdim) {                   // 是否保持维度不变

  // 确定有效的维度
  int64_t dim = maybe_wrap_dim(dim_, self.dim());
  // 确保 self 的指定维度 dim 上有数据
  zero_numel_check_dims(self, dim, "kthvalue()");

  // 检查 k 是否在有效范围内
  TORCH_CHECK(k >= 1 && k <= self.size(dim),
              "kthvalue(): selected number k out of range for dimension ", dim);

  // 检查 self 和 values 是否有重叠的数据区域
  at::assert_no_overlap(self, values);

  // 分配或调整输出张量 values 和 indices 的大小
  _reduction_with_indices_allocate_or_resize_output(
      values, indices, self, dim, keepdim);

  // 处理 self 是标量的情况
  if (self.dim() == 0 && self.numel() == 1) {
    values.copy_(self);  // 将 self 的值复制给 values
    indices.zero_();     // 将 indices 置零
    return std::forward_as_tuple(values, indices);  // 返回结果元组
  }

  // 检查 self 的维度是否超过最大限制
  TORCH_CHECK(
      self.dim() <= MAX_TENSORINFO_DIMS,
      "cannot operate on more than ",
      MAX_TENSORINFO_DIMS,
      " dimensions");

  // 根据索引类型的需求，选择合适的算法并执行
  if (self.numel() != 0) {
    launch_kthvalue_kernel(values, indices, self, dim, k);  // 调用 CUDA 核心函数计算第 k 小值
  }

  // 如果 keepdim 为 false，则压缩维度 dim 上的 values 和 indices
  if (!keepdim) {
    values.squeeze_(dim);
    indices.squeeze_(dim);
  }

  // 返回计算结果的元组
  return std::forward_as_tuple(values, indices);
}

// median_with_indices_impl 函数的声明，用于计算中位数及其索引
std::tuple<Tensor&, Tensor&> median_with_indices_impl(
    Tensor& values,               // 输出张量，保存中位数
    Tensor& indices,              // 输出张量，保存中位数的索引
    const Tensor& self,           // 输入张量，进行计算的原始数据
    int64_t dim,                  // 操作的维度
    bool keepdim,


这里只注释了部分代码，其余部分类似。
    bool ignore_nan) {
  // See note [Writing Nondeterministic Operations]
  // 如果中位数有重复元素，则选择用于输出索引的重复元素的过程是非确定性的。

  // 发出警告，表明使用 CUDA 计算中位数并输出索引的操作是非确定性的
  at::globalContext().alertNotDeterministic("median CUDA with indices output");
  // 禁用命名保护
  NoNamesGuard guard;

  // 确定操作维度的有效性，处理负数维度情况
  dim = at::maybe_wrap_dim(dim, self.dim());
  // 如果 self 的维度大于 0，则将其转换为连续存储的张量，否则在第 0 维度上添加维度
  Tensor in = self.dim() > 0 ? self.contiguous() : self.unsqueeze(0);

  // 检查所有相关张量的设备类型是否一致，应为当前张量的设备类型
  checkDeviceType("median", {values, indices}, self.device().type());
  // 检查 indices 张量的标量类型为 kLong
  checkScalarType("median", {indices, "indices", 1}, kLong);
  // 检查 values 张量的数据类型与 self 张量的数据类型一致
  checkSameType("median", {values, "values", 0}, {self, "self", 2});

  // 检查 self 张量的维度是否超过限制
  TORCH_CHECK(
      self.dim() <= MAX_TENSORINFO_DIMS,
      "median() cannot operate on more than ",
      MAX_TENSORINFO_DIMS,
      " dimensions");

  // 获取输出张量的形状，并确保对应维度不是零
  std::vector<int64_t> out_shape = self.sizes().vec();
  zero_numel_check_dims(self, dim, "median()");
  if (self.dim() > 0) {
    // 断言维度的有效性
    assert(dim >= 0);
    assert(dim < static_cast<int64_t>(out_shape.size()));

    // 如果 keepdim 为 true，则在对应维度上保持维度为 1，否则在 out_shape 中移除对应维度
    if (keepdim) {
      out_shape[dim] = 1;
    } else {
      out_shape.erase(out_shape.begin() + dim);
    }
  }

  // 重置 values 和 indices 张量的形状
  values.resize_(out_shape);
  indices.resize_(out_shape);

  // 只有当 self 张量的元素个数大于 0 时才启动核函数计算
  if (self.numel() > 0) {
    // 如果 keepdim 为 true 且 self 的维度大于 0，则使用 values 和 indices 张量，
    // 否则在 dim 维度上添加维度
    Tensor vals = keepdim && self.dim() > 0 ? values : values.unsqueeze(dim);
    Tensor inds = keepdim && self.dim() > 0 ? indices : indices.unsqueeze(dim);

    // 调用核函数计算中位数，并填充 values 和 indices 张量
    launch_median_kernel(vals, inds, in, dim, ignore_nan);
  }

  // 重置命名推断，用于 reduction 操作的 names
  guard.reset();
  namedinference::propagate_names_for_reduction(values, self, dim, keepdim);
  namedinference::propagate_names_for_reduction(indices, self, dim, keepdim);

  // 返回包含 values 和 indices 的元组
  return std::forward_as_tuple(values, indices);
} // namespace (anonymous)
```