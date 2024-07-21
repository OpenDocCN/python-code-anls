# `.\pytorch\aten\src\ATen\native\ReduceOpsUtils.h`

```
#pragma once

#include <limits> // 包含标准库中数值极限的头文件
#include <ATen/core/Tensor.h> // 包含 PyTorch 中 Tensor 的核心头文件
#include <ATen/native/Resize.h> // 包含 PyTorch 中 Tensor Resize 的头文件
#include <ATen/native/TensorIterator.h> // 包含 PyTorch 中 Tensor 迭代器的头文件
#include <ATen/native/NonEmptyUtils.h> // 包含 PyTorch 中非空实用工具的头文件
#include <ATen/WrapDimUtilsMulti.h> // 包含 PyTorch 中 WrapDimUtilsMulti 的头文件
#include <c10/core/ScalarType.h> // 包含 PyTorch 中标量类型的头文件
#include <c10/util/irange.h> // 包含 c10 中 irange 的头文件

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h> // 如果未定义 AT_PER_OPERATOR_HEADERS，则包含 PyTorch 中的函数头文件
#else
#include <ATen/ops/empty.h> // 如果定义了 AT_PER_OPERATOR_HEADERS，则包含 PyTorch 中 empty 操作的头文件
#include <ATen/ops/scalar_tensor.h> // 如果定义了 AT_PER_OPERATOR_HEADERS，则包含 PyTorch 中 scalar_tensor 操作的头文件
#endif

namespace at::native {

// 定义获取标量类型的最大可能值，包括无穷大
template <typename scalar_t>
constexpr scalar_t upper_bound() {
  using lim = std::numeric_limits<scalar_t>;
  return lim::has_infinity ? lim::infinity() : lim::max();
}

// 定义获取标量类型的最小可能值，包括负无穷大
template <typename scalar_t>
constexpr scalar_t lower_bound() {
  using lim = std::numeric_limits<scalar_t>;
  return lim::has_infinity ? -lim::infinity() : lim::lowest();
}

// 重新设定张量的某一维度，使其步长为 0
inline Tensor restride_dim(
  const Tensor& src, int64_t dim,
  IntArrayRef replacement_shape
) {
  auto strides = ensure_nonempty_vec(src.strides().vec()); // 确保步长向量非空
  strides[dim] = 0; // 将指定维度的步长设为 0
  return src.as_strided(replacement_shape, strides); // 返回重新设定步长后的张量
}

// 设置维度缩减操作的初始步骤，调整结果张量的大小以适应维度缩减
inline void _dimreduce_setup(const Tensor &result, const Tensor &self,
                                int64_t dim) {
  IntArrayRef self_sizes = self.sizes(); // 获取输入张量的大小
  std::vector<int64_t> result_sizes; // 定义结果张量的大小向量
  result_sizes.insert(result_sizes.end(), self_sizes.begin(), self_sizes.end()); // 复制输入张量的大小到结果张量
  result_sizes[dim] = 1; // 将指定维度的大小设为 1
  result.resize_(result_sizes); // 调整结果张量的大小
}

// 检查是否可以通过简单规约返回结果（如标量），并执行必要的操作
inline bool _dimreduce_return_trivial(const Tensor &result, const Tensor &self,
                                      const Scalar& ident, int64_t dim, bool keepdim) {
  if (self.numel() == 1 && self.ndimension() == 0) {
    result.resize_({}); // 调整结果张量为标量
    result.fill_(self); // 用输入张量的值填充结果张量
    return true; // 返回成功
  }
  // 返回身份元素
  if (self.numel() == 0) {
    _dimreduce_setup(result, self, dim); // 设置维度缩减的初始步骤
    result.fill_(ident); // 用身份元素填充结果张量
    if (!keepdim) result.squeeze_(dim); // 如果不保留维度，压缩结果张量的指定维度
    return true; // 返回成功
  }
  return false; // 返回失败
}

// 检查是否可以通过简单规约返回结果（不使用身份元素），并执行必要的操作
inline bool _dimreduce_return_trivial_no_ident(Tensor &result, const Tensor &self,
                                               int64_t /*dim*/, bool /*keepdim*/, const char* /*fn_name*/) {
  if (self.numel() == 1 && self.ndimension() == 0) {
    result.resize_({}); // 调整结果张量为标量
    result.fill_(self); // 用输入张量的值填充结果张量
    return true; // 返回成功
  }

  return false; // 返回失败
}

// 检查是否可以通过简单规约返回所有结果，并执行必要的操作
inline std::optional<Tensor> _allreduce_return_trivial(
    const Tensor& self,
    const Scalar& ident) {
  // 返回身份元素
  if (self.numel() == 0) {
    return at::scalar_tensor(ident, self.options()); // 返回一个标量张量
  }
  return c10::nullopt; // 返回空值
}

// 宏定义，检查两个张量的标量类型、设备和布局是否相等
#define OPTION_TYPE_EQUALITY_CHECK(option, out, self) \
{ \
  TORCH_CHECK(\
    out.option() == self.option(),\
    "expected ", #option, " ",\
    self.option(),\
    " but found ", out.option())\
}

// 检查两个张量的标量类型、设备和布局是否相等
inline void check_scalar_type_device_layout_equal(const Tensor& out, const Tensor& self) {
  OPTION_TYPE_EQUALITY_CHECK(scalar_type, out, self); // 检查标量类型是否相等
  OPTION_TYPE_EQUALITY_CHECK(device, out.options(), self.options()); // 检查设备是否相等
  OPTION_TYPE_EQUALITY_CHECK(layout, out.options(), self.options()); // 检查布局是否相等
}

} // namespace at::native
// 将输入张量 self 的数据类型向上转型，返回转型后的张量
inline Tensor integer_upcast(const Tensor& self, std::optional<ScalarType> dtype) {
  // 获取输入张量 self 的数据类型
  ScalarType scalarType = self.scalar_type();
  // 检查是否是未实现的无符号类型，如果是则报错
  TORCH_CHECK(!isBarebonesUnsignedType(scalarType), "integer upcasting for uint16, uint32 and uint64 is not currently implemented");
  // 根据可选的 dtype 或者输入张量的整数类型来确定向上转型后的数据类型
  ScalarType upcast_scalarType = dtype.value_or(at::isIntegralType(scalarType, /*includeBool=*/true) ? ScalarType::Long : scalarType);
  // 将输入张量 self 转换为指定数据类型的张量并返回
  return self.toType(upcast_scalarType);
}

// 定义维度掩码类型的别名
using DimMask = TensorIterator::DimMask;

// 根据可选的维度列表 opt_dims 和张量的维度数 ndim 创建维度向量
inline DimVector make_dim_vector(OptionalIntArrayRef opt_dims, int64_t ndim) {
  if (opt_dims.has_value()) {
    // 如果提供了有效的维度列表，则使用该列表创建 DimVector
    return DimVector(opt_dims.value());
  } else {
    // 如果未提供有效的维度列表，则创建一个包含所有维度索引的 DimVector
    std::vector<int64_t> all_dims(ndim);
    std::iota(all_dims.begin(), all_dims.end(), 0);
    return DimVector(all_dims);
  }
}

// 根据可选的维度列表 opt_dims 和张量的维度数 ndim 创建维度掩码
inline DimMask make_dim_mask(OptionalIntArrayRef opt_dims, int64_t ndim, bool allow_empty_dims=false) {
  // 初始化维度掩码
  DimMask mask;
  if (opt_dims.has_value()) {
    auto dims = opt_dims.value();
    if (dims.empty() && !allow_empty_dims) {
      // 如果维度列表为空且不允许空维度，则将掩码初始化为全部置位
      mask = DimMask().flip();
    } else {
      // 否则根据提供的维度列表转换为位掩码
      mask = at::dim_list_to_bitset(dims, ndim);
    }
  } else {
    // 如果未提供有效的维度列表，则将掩码初始化为全部置位
    mask = DimMask().flip();
  }
  // 返回生成的维度掩码
  return mask;
}

// 根据掩码 mask 和 keepdim 标志，从输入张量 self 推导出形状
inline DimVector shape_from_dim_mask(const Tensor& self, DimMask mask, bool keepdim) {
  // 获取输入张量 self 的形状
  auto shape = DimVector(self.sizes());
  // 从最后一个维度开始遍历形状
  for (int dim = shape.size() - 1; dim >= 0; dim--) {
    // 如果掩码对应位置为真
    if (mask[dim]) {
      // 如果 keepdim 为真，则将该维度设为1，否则从形状中删除该维度
      if (keepdim) {
        shape[dim] = 1;
      } else {
        shape.erase(shape.begin() + dim);
      }
    }
  }
  // 返回推导出的形状
  return shape;
}

// 调整输出张量 result 的大小以匹配 reduction 操作的要求
inline void resize_reduction_result(
    Tensor& result, const Tensor& self, DimMask mask, bool keepdim,
    ScalarType /*dtype*/)
{
  // 根据掩码 mask 和 keepdim 标志推导出形状
  auto shape = shape_from_dim_mask(self, mask, keepdim);
  // 检查结果张量是否已定义
  TORCH_CHECK(result.defined(), "Cannot create a new tensor inside a reduction op. You likely tried to call an operator with an out argument but the out argument was an undefined tensor.");
  // 调整结果张量的输出形状
  at::native::resize_output(result, shape);
}

// 创建 reduction 操作的结果张量
inline Tensor create_reduction_result(
  const Tensor& self, at::OptionalIntArrayRef dim, bool keepdim, ScalarType dtype
) {
  // 根据维度列表 dim 和输入张量 self 的维度数创建维度掩码
  DimMask mask = make_dim_mask(dim, self.dim());
  // 根据掩码 mask 和 keepdim 标志推导出形状
  auto shape = shape_from_dim_mask(self, mask, keepdim);
  // 创建指定数据类型 dtype 和推导出形状的空张量
  return at::empty(shape, self.options().dtype(dtype));
}

// 根据结果张量的形状和掩码 mask 进行 review 操作
inline Tensor review_reduce_result(const Tensor& result, int ndim, DimMask mask, bool keepdim) {
  // 如果 keepdim 为真，则直接返回结果张量
  if (keepdim) {
    return result;
  }
  // 否则进行 review 操作，调整形状和步长
  auto shape = DimVector(result.sizes());
  auto stride = DimVector(result.strides());
  for (const auto dim : c10::irange(ndim)) {
    // 如果掩码对应位置为真，则在该位置插入新维度和步长
    if (mask[dim]) {
      shape.insert(shape.begin() + dim, 1);
      stride.insert(stride.begin() + dim, 0);
    }
  }
  // 使用调整后的形状和步长创建新的张量并返回
  return result.as_strided(shape, stride);
}

// 创建一个张量迭代器以进行 reduction 操作
inline TensorIterator make_reduction(
    const char* name, Tensor& result, const Tensor& self,
    at::OptionalIntArrayRef dim_opt,
  // 检查如果提供了结果张量 `result`，则其数据类型必须与 `out_dtype` 一致
  TORCH_CHECK(
      !result.defined() || result.scalar_type() == out_dtype,
      name, ": provided dtype must match dtype of result. Got ",
      toString(result.scalar_type()),
      " and ",
      toString(out_dtype),
      ".");

  // 如果未提供维度 `dim_opt`，则将 `dim` 初始化为空数组引用
  IntArrayRef dim = dim_opt.value_or(IntArrayRef{});

  // 获取输入张量 `self` 的维度数量
  int64_t ndim = self.dim();

  // 根据给定的维度 `dim` 创建维度掩码 `mask`
  auto mask = make_dim_mask(dim, ndim);

  // 调整（或初始化）归约操作的结果张量 `result` 的大小和数据类型
  resize_reduction_result(result, self, mask, keepdim, out_dtype);

  // 根据维度掩码和参数 `keepdim` 重新视图化归约操作的结果张量 `result`
  auto viewed_result = review_reduce_result(result, ndim, mask, keepdim);

  // 根据命名推断，在归约操作后传播结果张量 `result` 的命名
  namedinference::propagate_names_for_reduction(result, self, dim, keepdim);

  // 如果输入张量 `self` 的数据类型等于 `in_dtype`，则直接使用张量迭代器进行归约操作
  if (self.scalar_type() == in_dtype) {
    return TensorIterator::reduce_op(viewed_result, self);
  }

  // 否则，将输入张量 `self` 转换为指定的数据类型 `in_dtype` 后再进行归约操作
  return TensorIterator::reduce_op(viewed_result, self.to(in_dtype));
// 返回一个 TensorIterator 对象，用于执行指定操作的张量降维运算
inline C10_UNUSED TensorIterator make_reduction(
    const char* name, Tensor& result, const Tensor& self,
    at::OptionalIntArrayRef dim, bool keepdim, ScalarType out_dtype) {
  // 对于混合精度的特殊情况，提高计算效率。
  // 不通用化到常见的输入/输出类型不匹配，以避免模板化内核启动的交叉产品。
  const bool gpu_lowp_to_f32 = (
    self.is_cuda() && (self.scalar_type() == kHalf || self.scalar_type() == kBFloat16) && out_dtype == kFloat);
  
  // 根据情况选择输入数据类型
  auto in_dtype = gpu_lowp_to_f32 ? self.scalar_type()
                   : self.is_complex() ? c10::toComplexType(out_dtype)
                                       : out_dtype;
  
  // 调用另一个重载的 make_reduction 函数，返回 TensorIterator 对象
  return make_reduction(name, result, self, dim, keepdim, in_dtype, out_dtype);
}

// 返回一个 TensorIterator 对象，用于执行指定操作的张量降维运算，同时处理两个结果张量
inline TensorIterator make_reduction(
    const char* name, Tensor& result1, Tensor& result2, const Tensor& self,
    at::OptionalIntArrayRef dim_opt, bool keepdim, ScalarType dtype1,
    ScalarType dtype2) {
  // 检查结果类型和数据类型是否匹配（如果已提供）
  TORCH_CHECK(
    (!result1.defined() || result1.scalar_type() == dtype1) && (!result2.defined() || result2.scalar_type() == dtype2),
    name, ": 提供的数据类型必须与结果的数据类型匹配。得到了 ",
    toString(result1.scalar_type()), toString(result2.scalar_type()),
    " 和 ",
    toString(dtype1), toString(dtype2),
    ".");
  
  // 如果未指定维度，则默认进行全局归约（all-reduce）
  auto dim = dim_opt.value_or(IntArrayRef{});
  int64_t ndim = self.dim();
  DimMask mask = make_dim_mask(dim, ndim);
  
  // 调整结果张量的尺寸以适应归约操作
  resize_reduction_result(result1, self, mask, keepdim, dtype1);
  auto viewed_result1 = review_reduce_result(result1, ndim, mask, keepdim);

  resize_reduction_result(result2, self, mask, keepdim, dtype2);
  auto viewed_result2 = review_reduce_result(result2, ndim, mask, keepdim);
  
  // 在归约操作中传播张量的命名属性
  namedinference::propagate_names_for_reduction(result1, self, dim, keepdim);
  namedinference::propagate_names_for_reduction(result2, self, dim, keepdim);
  
  // 对于混合精度的特殊情况，提高计算效率。
  // 不通用化到常见的输入/输出类型不匹配，以避免模板化内核启动的交叉产品。
  if (self.scalar_type() == dtype1 ||
      (self.is_cuda() && self.scalar_type() == kHalf && dtype1 == kFloat)) {
    return TensorIterator::reduce_op(viewed_result1, viewed_result2, self);
  }
  
  // 否则，将 self 张量转换为指定的数据类型并返回 TensorIterator 对象
  return TensorIterator::reduce_op(viewed_result1, viewed_result2, self.to(dtype1));
}

// 返回一个 TensorIterator 对象，用于执行指定操作的张量降维运算，处理相同的结果张量
inline C10_UNUSED TensorIterator make_reduction(
    const char* name, Tensor& result1, Tensor& result2, const Tensor& self,
    at::OptionalIntArrayRef dim, bool keepdim, ScalarType dtype) {
  // 调用前面定义的重载函数，指定相同的数据类型作为输入和输出
  return make_reduction(name, result1, result2, self, dim, keepdim, dtype, dtype);
}

// 当输入张量的维度为零时，抛出错误信息
inline void zero_numel_check_dims(const Tensor& self, const int64_t dim, const char *fn_name) {
  if (self.ndimension() == 0) {
    # 如果要求进行标量的缩减（reduction），检查维度（dim）是否为0或-1
    TORCH_CHECK_INDEX(dim == 0 || dim == -1, fn_name,
      ": Expected reduction dim -1 or 0 for scalar but got ", dim);
  }
  else {
    # 如果不是标量的缩减，检查指定维度（dim）的尺寸是否非零
    TORCH_CHECK_INDEX(self.size(dim) != 0, fn_name,
      ": Expected reduction dim ", dim, " to have non-zero size.");
  }
} // 结束 at::native 命名空间

namespace at::meta {

// 获取归约操作后的张量形状，根据给定的维度和是否保持维度信息
inline C10_UNUSED DimVector get_reduction_shape(
    const Tensor& self,
    IntArrayRef dims,
    bool keepdim,
    bool allow_empty_dims=false) {
  // 创建维度掩码，根据 dims 和 self 的维度，允许空维度标志
  auto mask = native::make_dim_mask(dims, self.dim(), allow_empty_dims);
  // 根据维度掩码和保持维度标志计算张量的形状
  return native::shape_from_dim_mask(self, mask, keepdim);
}

// 调整归约操作的结果元信息，根据输入张量、可选的维度列表、保持维度标志和输出数据类型
inline void resize_reduction(
    impl::MetaBase& meta,
    const Tensor& self,
    OptionalIntArrayRef opt_dims,
    bool keepdim,
    ScalarType out_dtype,
    # 定义函数 resize_reduction，用于执行张量的降维操作
    bool allow_empty_dims=false) {
  # 使用 opt_dims 和当前张量的维度信息创建一个维度向量 dims_
  DimVector dims_ = at::native::make_dim_vector(opt_dims, self.dim());
  # 根据当前张量的布局情况，可能调整维度向量 dims_
  maybe_wrap_dims(dims_, self.dim());
  # 根据给定的维度信息和参数，获取降维后的形状 shape
  auto shape = get_reduction_shape(self, dims_, keepdim, allow_empty_dims);
  # 如果当前张量的布局为 kStrided
  if (self.layout() == kStrided) {
    # 设置输出张量的元数据，使用原始步长布局，指定形状 shape，无额外步长信息，使用指定的输出数据类型 out_dtype
    meta.set_output_raw_strided(0, shape, {}, self.options().dtype(out_dtype));
  # 如果降维后的形状 shape 为空
  } else if (shape.size() == 0) {
    # 设置输出张量的元数据，使用原始步长布局，指定形状 shape，无额外步长信息，使用指定的输出数据类型 out_dtype，并设置布局为 kStrided
    meta.set_output_raw_strided(0, shape, {}, self.options().dtype(out_dtype).layout(kStrided));
  # 如果以上条件都不满足
  } else {
    # 抛出错误，提示不支持当前布局类型的输出
    TORCH_CHECK(false, "resize_reduction: support for output with ", self.layout(), " layout is not implemented yet");
  }
  # 根据降维操作后的结果，推断和传播输出张量的命名信息
  namedinference::propagate_names_for_reduction(
      meta.maybe_get_output(), self, dims_, keepdim);
}

// 通过给定的维度参数调整大小并生成相应的索引
inline void resize_reduction_with_indices(
    // 元数据基类，用于存储和操作张量的元信息
    impl::MetaBase& meta,
    // 待操作的张量
    const Tensor& self,
    // 要进行缩减的维度列表
    IntArrayRef dims,
    // 是否保持维度
    bool keepdim,
    // 输出张量的数据类型
    ScalarType out_dtype) {
  
  // 将 IntArrayRef 转换为 DimVector
  DimVector dims_(dims);
  // 调整维度列表，确保其有效性
  maybe_wrap_dims(dims_, self.dim());
  
  // 根据输入张量和维度参数，获取缩减操作后的形状
  auto shape = get_reduction_shape(self, dims_, keepdim);
  
  // 设置第一个输出张量的元信息，包括形状和数据类型
  meta.set_output_raw_strided(0, shape, {}, self.options().dtype(out_dtype));
  
  // 设置第二个输出张量的元信息，用于存储索引，形状与第一个输出相同，数据类型为长整型
  meta.set_output_raw_strided(1, shape, {}, self.options().dtype(kLong));
  
  // 根据缩减操作的输入张量、维度列表和保持维度标志，传播缩减操作的名称信息
  namedinference::propagate_names_for_reduction(
      meta.maybe_get_output(0), self, dims_, keepdim);
  
  // 类似地，传播第二个输出张量的名称信息
  namedinference::propagate_names_for_reduction(
      meta.maybe_get_output(1), self, dims_, keepdim);
}

// 根据输入张量和输出张量生成一个张量迭代器，用于执行缩减操作
inline TensorIterator make_reduction(
    // 输入张量
    const Tensor& self,
    // 输出张量
    const Tensor& result,
    // 可选的维度列表
    OptionalIntArrayRef opt_dims,
    // 是否保持维度
    bool keepdim,
    // 输入张量的数据类型
    ScalarType in_dtype) {
  
  // 获取输入张量的维度数量
  int64_t ndim = self.dim();
  
  // 根据可选的维度列表生成一个掩码，表示应用缩减操作的维度
  auto mask = at::native::make_dim_mask(opt_dims, ndim);
  
  // 生成一个视图结果张量，用于缩减操作的结果
  auto viewed_result =
      at::native::review_reduce_result(result, ndim, mask, keepdim);
  
  // 如果输入张量的数据类型与指定的输入数据类型相同，直接返回缩减操作迭代器
  if (self.scalar_type() == in_dtype) {
    return TensorIterator::reduce_op(viewed_result, self);
  }
  
  // 否则，将输入张量转换为指定的输入数据类型后返回缩减操作迭代器
  return TensorIterator::reduce_op(viewed_result, self.to(in_dtype));
}

// 根据输入张量、两个输出张量和维度列表生成一个张量迭代器，用于执行复杂的缩减操作
inline TensorIterator make_reduction(
    // 输入张量
    const Tensor& self,
    // 第一个输出张量
    const Tensor& result1,
    // 第二个输出张量
    const Tensor& result2,
    // 要进行缩减的维度列表
    IntArrayRef dims,
    // 是否保持维度
    bool keepdim,
    // 第一个输出张量的数据类型
    ScalarType dtype1,
    // 第二个输出张量的数据类型，此处未使用
    ScalarType /*dtype2*/) {
  
  // 获取输入张量的维度数量
  int64_t ndim = self.dim();
  
  // 根据维度列表生成一个掩码，表示应用缩减操作的维度
  auto mask = at::native::make_dim_mask(dims, ndim);
  
  // 生成两个视图结果张量，用于缩减操作的结果
  auto viewed_result1 = at::native::review_reduce_result(result1, ndim, mask, keepdim);
  auto viewed_result2 = at::native::review_reduce_result(result2, ndim, mask, keepdim);
  
  // 如果输入张量的数据类型与第一个输出张量的数据类型相同，或者是特殊的混合精度情况，直接返回缩减操作迭代器
  if (self.scalar_type() == dtype1 ||
      (self.is_cuda() && self.scalar_type() == kHalf && dtype1 == kFloat)) {
    return TensorIterator::reduce_op(viewed_result1, viewed_result2, self);
  }
  
  // 否则，将输入张量转换为指定的第一个输出数据类型后返回缩减操作迭代器
  return TensorIterator::reduce_op(viewed_result1, viewed_result2, self.to(dtype1));
}

// 根据输入张量和输出张量生成一个张量迭代器，用于执行缩减操作，并从输出张量类型推断输入数据类型
inline C10_UNUSED TensorIterator make_reduction_from_out_ty(
    // 输入张量
    const Tensor& self,
    // 输出张量
    const Tensor& result,
    // 可选的维度列表
    OptionalIntArrayRef opt_dims,
    // 是否保持维度
    bool keepdim,
    // 输出张量的数据类型
    ScalarType out_dtype) {
  
  // 特殊情况：在混合精度计算中进行类型提升，以提高计算效率
  const bool gpu_lowp_to_f32 =
      (self.is_cuda() &&
       (self.scalar_type() == kHalf || self.scalar_type() == kBFloat16) &&
       out_dtype == kFloat);
  
  // 根据情况确定输入数据类型
  auto in_dtype = gpu_lowp_to_f32 ? self.scalar_type() : out_dtype;
  
  // 返回根据确定的输入数据类型生成的缩减操作迭代器
  return make_reduction(self, result, opt_dims, keepdim, in_dtype);
}

} // namespace at::meta
```