# `.\pytorch\aten\src\ATen\native\ReduceOps.cpp`

```py
// 定义编译时使用的宏，仅启用方法操作符
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS

// 包含 ATen 库中的 ReduceOps.h 文件
#include <ATen/native/ReduceOps.h>

// 包含 ATen 核心 Tensor 类定义
#include <ATen/core/Tensor.h>

// 包含 ATen 库中的 AccumulateType 类型定义
#include <ATen/AccumulateType.h>

// 包含 ATen 分发机制相关头文件
#include <ATen/Dispatch.h>
#include <ATen/Dispatch_v2.h>

// 包含 ATen 并行处理相关功能
#include <ATen/Parallel.h>

// 包含 ATen 中用于处理维度的实用函数
#include <ATen/WrapDimUtils.h>
#include <ATen/WrapDimUtilsMulti.h>

// 包含 ATen 中的 Tensor 迭代器定义
#include <ATen/TensorIterator.h>

// 包含 ATen 中的张量操作函数定义
#include <ATen/TensorOperators.h>

// 包含 ATen 中用于命名张量的实用函数
#include <ATen/NamedTensorUtils.h>

// 包含 ATen 中的 ReduceOpsUtils 函数定义
#include <ATen/native/ReduceOpsUtils.h>

// 包含 ATen 中的 Resize 函数定义
#include <ATen/native/Resize.h>

// 包含 ATen 中的 TensorDimApply 函数定义
#include <ATen/native/TensorDimApply.h>

// 包含 ATen 中的梯度模式管理函数定义
#include <ATen/core/grad_mode.h>

// 包含 ATen 中用于张量子类的实用函数定义
#include <ATen/TensorSubclassLikeUtils.h>

// 如果未定义 AT_PER_OPERATOR_HEADERS 宏，则包含下列 ATen 和 NativeFunctions 头文件
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>

// 如果定义了 AT_PER_OPERATOR_HEADERS 宏，则包含下列操作相关的头文件
#else
#include <ATen/ops/_cummax_helper.h>
#include <ATen/ops/_cummax_helper_native.h>
#include <ATen/ops/_cummin_helper.h>
#include <ATen/ops/_cummin_helper_native.h>
#include <ATen/ops/_is_all_true_native.h>
#include <ATen/ops/_is_any_true_native.h>
#include <ATen/ops/_logcumsumexp.h>
#include <ATen/ops/_logcumsumexp_native.h>
#include <ATen/ops/_sparse_csr_sum.h>
#include <ATen/ops/_sparse_sum.h>
#include <ATen/ops/_sparse_sum_native.h>
#include <ATen/ops/_to_copy.h>
#include <ATen/ops/add.h>
#include <ATen/ops/all_meta.h>
#include <ATen/ops/all_native.h>
#include <ATen/ops/amax.h>
#include <ATen/ops/amax_meta.h>
#include <ATen/ops/amax_native.h>
#include <ATen/ops/amin_meta.h>
#include <ATen/ops/amin_native.h>
#include <ATen/ops/aminmax_meta.h>
#include <ATen/ops/aminmax_native.h>
#include <ATen/ops/any_meta.h>
#include <ATen/ops/any_native.h>
#include <ATen/ops/argmax_meta.h>
#include <ATen/ops/argmax_native.h>
#include <ATen/ops/argmin_meta.h>
#include <ATen/ops/argmin_native.h>
#include <ATen/ops/cat.h>
#include <ATen/ops/complex.h>
#include <ATen/ops/cummax.h>
#include <ATen/ops/cummax_native.h>
#include <ATen/ops/cummaxmin_backward_native.h>
#include <ATen/ops/cummin.h>
#include <ATen/ops/cummin_native.h>
#include <ATen/ops/cumprod.h>
#include <ATen/ops/cumprod_backward_native.h>
#include <ATen/ops/cumprod_meta.h>
#include <ATen/ops/cumprod_native.h>
#include <ATen/ops/cumsum.h>
#include <ATen/ops/cumsum_meta.h>
#include <ATen/ops/cumsum_native.h>
#include <ATen/ops/diff_native.h>
#include <ATen/ops/dist_native.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/equal_native.h>
#include <ATen/ops/exp.h>
#include <ATen/ops/gather.h>
#include <ATen/ops/gradient_native.h>
#include <ATen/ops/imag.h>
#include <ATen/ops/isnan_native.h>
#include <ATen/ops/linalg_vector_norm.h>
#include <ATen/ops/logcumsumexp.h>
#include <ATen/ops/logcumsumexp_native.h>
#include <ATen/ops/logical_xor.h>
#include <ATen/ops/logsumexp.h>
#include <ATen/ops/logsumexp_native.h>
#include <ATen/ops/mean.h>
#include <ATen/ops/mean_meta.h>
#include <ATen/ops/mean_native.h>
#include <ATen/ops/nanmean_native.h>
#include <ATen/ops/nansum.h>
#include <ATen/ops/nansum_native.h>
#include <ATen/ops/narrow.h>
#include <ATen/ops/native_norm.h>
#include <ATen/ops/ne.h>
#endif
// 包含 ATen 库中定义的一系列操作函数头文件

#include <ATen/ops/norm.h>
#include <ATen/ops/norm_meta.h>
#include <ATen/ops/norm_native.h>
#include <ATen/ops/ones.h>
#include <ATen/ops/prod.h>
#include <ATen/ops/prod_meta.h>
#include <ATen/ops/prod_native.h>
#include <ATen/ops/real.h>
#include <ATen/ops/slice.h>
#include <ATen/ops/special_logsumexp_native.h>
#include <ATen/ops/sqrt.h>
#include <ATen/ops/squeeze.h>
#include <ATen/ops/stack.h>
#include <ATen/ops/std.h>
#include <ATen/ops/std_mean.h>
#include <ATen/ops/std_mean_native.h>
#include <ATen/ops/std_native.h>
#include <ATen/ops/sub.h>
#include <ATen/ops/sum.h>
#include <ATen/ops/sum_meta.h>
#include <ATen/ops/sum_native.h>
#include <ATen/ops/trace_native.h>
#include <ATen/ops/value_selecting_reduction_backward_native.h>
#include <ATen/ops/var.h>
#include <ATen/ops/var_mean.h>
#include <ATen/ops/var_mean_native.h>
#include <ATen/ops/var_native.h>
#include <ATen/ops/zeros.h>
#include <ATen/ops/zeros_like.h>

// 包含 C10 库中定义的工具和数据结构头文件
#include <c10/util/irange.h>
#include <c10/util/SmallBuffer.h>

// 包含标准库头文件
#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <numeric>
#include <type_traits>
#include <utility>
#include <vector>

// 进入 ATen 命名空间的 meta 命名空间
namespace at::meta {

// 定义一个静态函数，推断张量的数据类型，根据可选参数 'opt_dtype' 和计算结果 'result'
static ScalarType infer_dtype_from_optional(
    const Tensor& self,
    const optional<ScalarType>& opt_dtype,
    const Tensor& result) {
  // 如果结果张量 'result' 已被定义
  if (result.defined()) {
    // 返回可选参数 'opt_dtype' 的值，如果没有则返回结果张量的数据类型
    return opt_dtype.value_or(result.scalar_type());
  } else {
    // 否则，获取自身张量 'self' 的数据类型
    // 如果自身张量是整数类型，则提升为 kLong 类型
    return at::native::get_dtype_from_self(self, opt_dtype, true);
  }
}

// 将 std::optional<int64_t> 转换为 IntArrayRef，如果未定义则返回空数组引用
static IntArrayRef optional_to_arrayref(const std::optional<int64_t>& opt) {
  return opt.has_value() ? opt.value() : IntArrayRef{};
}

// 获取结果张量或者 kByte 或 kBool 的数据类型，根据结果张量 'result' 是否定义和自身张量 'self' 的类型
static ScalarType get_result_or_bytebool_dtype(const Tensor& self, const Tensor& result) {
  // 参考 [all, any : uint8 compatibility]
  if (result.defined()) {
    // 如果结果张量已定义，返回其数据类型
    return result.scalar_type();
  } else {
    // 否则，如果自身张量类型是 kByte，则返回 kByte，否则返回 kBool
    return (self.scalar_type() == kByte) ? kByte : kBool;
  }
}

// 检查结果张量是否是 kByte 或 kBool 类型，否则报错
static void check_result_is_bytebool(const char* name, const Tensor& self, const Tensor& result) {
  if (result.defined()) {
    // 参考 [all, any : uint8 compatibility]
    TORCH_CHECK(
        result.scalar_type() == ScalarType::Bool ||
            result.scalar_type() == ScalarType::Byte,
        name, " only supports bool tensor for result, got: ",
        result.scalar_type());
  }
}

// 注释 [all, any : uint8 compatibility]：
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// 为了与 NumPy 兼容性，`all` 和 `any` 返回 dtype 为 `bool` 的张量。
// 然而，为了兼容性考虑，对于 `uint8` 类型，它们返回与输入类型相同的 `uint8` 张量。
// 参考：https://github.com/pytorch/pytorch/pull/47878#issuecomment-747108561
static void allany_meta(
    impl::MetaBase& meta,
    const char* name,
    const Tensor& self,
    OptionalIntArrayRef dims,
    bool keepdim) {
  // 获取操作的输出结果
  const auto& result = meta.maybe_get_output();
  // 检查输出结果是否为字节布尔类型
  check_result_is_bytebool(name, self, result);
  // 获取输出数据类型，或者使用字节布尔数据类型
  auto out_dtype = get_result_or_bytebool_dtype(self, result);
  // 调整张量的尺寸以进行缩减操作
  resize_reduction(meta, self, dims, keepdim, out_dtype, /*allow_empty_dims=*/true);
TORCH_META_FUNC2(all, dim)(const Tensor& self, int64_t dim, bool keepdim) {
    // 调用 allany_meta 函数处理 "all" 操作，传入 self 引用，维度 dim，是否保持维度信息
    allany_meta(*this, "all", self, dim, keepdim);
}

TORCH_META_FUNC2(all, dims)(const Tensor& self, OptionalIntArrayRef dim, bool keepdim) {
    // 调用 allany_meta 函数处理 "all" 操作，传入 self 引用，维度数组 dim，是否保持维度信息
    allany_meta(*this, "all", self, dim, keepdim);
}

TORCH_META_FUNC(all)(const Tensor& self) {
    // 调用 allany_meta 函数处理 "all" 操作，传入 self 引用，空维度数组，不保持维度信息
    allany_meta(*this, "all", self, {}, false);
}

TORCH_META_FUNC2(any, dim)(const Tensor& self, int64_t dim, bool keepdim) {
    // 调用 allany_meta 函数处理 "any" 操作，传入 self 引用，维度 dim，是否保持维度信息
    allany_meta(*this, "any", self, dim, keepdim);
}

TORCH_META_FUNC2(any, dims)(const Tensor& self, OptionalIntArrayRef dim, bool keepdim) {
    // 调用 allany_meta 函数处理 "any" 操作，传入 self 引用，维度数组 dim，是否保持维度信息
    allany_meta(*this, "any", self, dim, keepdim);
}

TORCH_META_FUNC(any)(const Tensor& self) {
    // 调用 allany_meta 函数处理 "any" 操作，传入 self 引用，空维度数组，不保持维度信息
    allany_meta(*this, "any", self, {}, false);
}

static void check_argmax_argmin(
    const char* name,
    const Tensor& self,
    const std::optional<int64_t>& dim) {
    // 检查是否指定了维度 dim，如果指定，则调整为有效维度
    if (dim.has_value()) {
        auto dim_ = maybe_wrap_dim(dim.value(), self.dim());
        // 对于零元素检查，使用指定的维度进行检查
        native::zero_numel_check_dims(self, dim_, name);
    } else {
        // 如果未指定维度并且张量元素数为零，则抛出错误
        TORCH_CHECK_INDEX(
            self.numel() != 0,
            name, ": Expected reduction dim to be specified for input.numel() == 0.");
    }
}

TORCH_META_FUNC(argmax)
(const Tensor& self, std::optional<int64_t> dim, bool keepdim) {
    // 调用 check_argmax_argmin 函数，检查并处理 "argmax" 操作的参数
    check_argmax_argmin("argmax()", self, dim);
    // 调用 resize_reduction 函数进行张量维度调整，以进行最大值计算
    resize_reduction(*this, self, optional_to_arrayref(dim), keepdim, kLong);
}

TORCH_META_FUNC(argmin)
(const Tensor& self, std::optional<int64_t> dim, bool keepdim) {
    // 调用 check_argmax_argmin 函数，检查并处理 "argmin" 操作的参数
    check_argmax_argmin("argmin()", self, dim);
    // 调用 resize_reduction 函数进行张量维度调整，以进行最小值计算
    resize_reduction(*this, self, optional_to_arrayref(dim), keepdim, kLong);
}

static void meta_func_cum_ops(
    impl::MetaBase& meta,
    const char* name,
    const Tensor& self,
    int64_t dim,
    std::optional<ScalarType> dtype) {
    // 检查维度 dim 的有效性
    maybe_wrap_dim(dim, self.dim());

    // 获取可能的输出结果
    const auto& result = meta.maybe_get_output();
    ScalarType out_dtype;

    // 根据输出结果是否已定义，确定输出的数据类型
    if (result.defined()) {
        out_dtype = dtype.value_or(result.scalar_type());
    } else {
        auto is_integral = at::isIntegralType(self.scalar_type(), /*includeBool=*/true);
        out_dtype = dtype.value_or(is_integral ? ScalarType::Long : self.scalar_type());
    }

    // 设置输出张量的类型和大小
    meta.set_output_raw_strided(0, self.sizes(), {}, self.options().dtype(out_dtype));
    // 将输出张量的命名信息传播到输入张量
    namedinference::propagate_names(result, self);
}

TORCH_META_FUNC(cumsum)
(const Tensor& self, int64_t dim, std::optional<ScalarType> dtype) {
    // 调用 meta_func_cum_ops 函数，处理 "cumsum" 操作的元信息
    meta_func_cum_ops(*this, "cumsum", self, dim, dtype);
}

TORCH_META_FUNC(cumprod)
(const Tensor& self, int64_t dim, std::optional<ScalarType> dtype) {
    // 调用 meta_func_cum_ops 函数，处理 "cumprod" 操作的元信息
    meta_func_cum_ops(*this, "cumprod", self, dim, dtype);
}

TORCH_META_FUNC2(sum, dim_IntList)
(const Tensor& self, OptionalIntArrayRef opt_dim, bool keepdim, optional<ScalarType> opt_dtype) {
    // 推断输出数据类型
    auto out_dtype = infer_dtype_from_optional(self, opt_dtype, maybe_get_output());
    // 调整张量以进行求和操作
    resize_reduction(*this, self, opt_dim, keepdim, out_dtype);
}

TORCH_META_FUNC2(prod, dim_int)
// 定义 TORCH_META_FUNC2(mean, dim) 函数，参数为输入张量 self、可选的维度 opt_dim、是否保持维度 keepdim 和可选的数据类型 opt_dtype
(const Tensor& self, OptionalIntArrayRef opt_dim, bool keepdim, optional<ScalarType> opt_dtype) {
  // 从输入张量 self 和 opt_dtype 推断输入的数据类型
  auto in_dtype = at::native::get_dtype_from_self(self, opt_dtype, true);

  // 如果输入数据类型既不是浮点类型也不是复数类型，则抛出错误
  if (!at::isFloatingType(in_dtype) && !at::isComplexType(in_dtype)) {
    std::string what = "Input";
    std::string dtype = toString(self.scalar_type());

    // 如果提供了 opt_dtype，则更新错误消息中的 what 和 dtype
    if (opt_dtype.has_value()) {
      what = "Optional";
      dtype = toString(opt_dtype.value());
    }

    // 抛出错误，说明无法推断输出数据类型
    TORCH_CHECK(
        false,
        "mean(): could not infer output dtype. ",
        what, " dtype must be either a floating point or complex dtype. ",
        "Got: ", dtype);
  }

  // 根据 opt_dtype 推断输出数据类型
  auto out_dtype = infer_dtype_from_optional(self, opt_dtype, maybe_get_output());
  // 调整张量维度以进行降维操作，保持维度 keepdim，使用推断得到的输出数据类型
  resize_reduction(*this, self, opt_dim, keepdim, out_dtype);
}
// 返回空白，不包含额外的内容
    }
  }



// 结束两个嵌套的代码块



  const auto options = self.options();



// 从对象 self 中获取选项信息，存储在变量 options 中



  this->set_output_raw_strided(0, shape, {}, options);
  this->set_output_raw_strided(1, shape, {}, options);



// 使用当前对象的方法 set_output_raw_strided 分别设置两个输出：
// - 第一个输出编号为 0，形状为 shape，空的步幅和选项信息为 options
// - 第二个输出编号为 1，形状为 shape，空的步幅和选项信息为 options
} // 结束 TORCH_META_FUNC(amax) 函数定义

TORCH_META_FUNC(amax)
(const Tensor& self, IntArrayRef dim, bool keepdim) {
  auto maybe_result = maybe_get_output();  // 尝试获取输出张量的可选结果
  if (maybe_result.defined()) {  // 如果结果已定义
    TORCH_CHECK(self.scalar_type() == maybe_result.scalar_type(), "Expected the dtype for input and out to match, but got ",
                self.scalar_type(), " for input's dtype and ",  maybe_result.scalar_type(), " for out's dtype.");
  }
  if (self.numel() == 0) {  // 如果输入张量元素数量为零
    at::native::zero_numel_check_dims(self, dim, "amax()");  // 执行零元素检查，针对指定维度
  }
  const ScalarType& out_dtype = maybe_result.defined() ? maybe_result.scalar_type() : self.scalar_type();  // 根据结果是否定义选择输出数据类型
  resize_reduction(*this, self, dim, keepdim, out_dtype);  // 调整尺寸以进行缩减操作
} // 结束 TORCH_META_FUNC(amax) 函数定义

TORCH_META_FUNC(amin)
(const Tensor& self, IntArrayRef dim, bool keepdim) {
  auto maybe_result = maybe_get_output();  // 尝试获取输出张量的可选结果
  if (maybe_result.defined()) {  // 如果结果已定义
    TORCH_CHECK(self.scalar_type() == maybe_result.scalar_type(), "Expected the dtype for input and out to match, but got ",
                self.scalar_type(), " for input's dtype and ",  maybe_result.scalar_type(), " for out's dtype.");
  }
  if (self.numel() == 0) {  // 如果输入张量元素数量为零
    at::native::zero_numel_check_dims(self, dim, "amin()");  // 执行零元素检查，针对指定维度
  }
  const ScalarType& out_dtype = maybe_result.defined() ? maybe_result.scalar_type() : self.scalar_type();  // 根据结果是否定义选择输出数据类型
  resize_reduction(*this, self, dim, keepdim, out_dtype);  // 调整尺寸以进行缩减操作
} // 结束 TORCH_META_FUNC(amin) 函数定义

} // 结束命名空间 at::meta

namespace at::native {

DEFINE_DISPATCH(aminmax_stub);  // 定义 aminmax_stub 分发函数
DEFINE_DISPATCH(aminmax_allreduce_stub);  // 定义 aminmax_allreduce_stub 分发函数

TORCH_IMPL_FUNC(aminmax_out)
(const Tensor& self,
 std::optional<int64_t> dim_opt,
 bool keepdim,
 const Tensor& min,
 const Tensor& max) {
  auto mutable_min = const_cast<Tensor&>(min);  // 将 min 张量转换为可变形式
  auto mutable_max = const_cast<Tensor&>(max);  // 将 max 张量转换为可变形式
  if (dim_opt.has_value()) {  // 如果维度参数有值
    aminmax_stub(
        self.device().type(),
        self,
        maybe_wrap_dim(dim_opt.value(), self.ndimension()),  // 对维度进行包装
        keepdim,
        mutable_min,
        mutable_max);
  } else {
    aminmax_allreduce_stub(self.device().type(), self.contiguous(), mutable_min, mutable_max);  // 执行全局范围内的 aminmax 操作
  }
} // 结束 TORCH_IMPL_FUNC(aminmax_out) 函数定义

DEFINE_DISPATCH(sum_stub);  // 定义 sum_stub 分发函数
DEFINE_DISPATCH(nansum_stub);  // 定义 nansum_stub 分发函数
DEFINE_DISPATCH(std_var_stub);  // 定义 std_var_stub 分发函数
DEFINE_DISPATCH(prod_stub);  // 定义 prod_stub 分发函数
DEFINE_DISPATCH(norm_stub);  // 定义 norm_stub 分发函数
DEFINE_DISPATCH(mean_stub);  // 定义 mean_stub 分发函数
DEFINE_DISPATCH(and_stub);  // 定义 and_stub 分发函数
DEFINE_DISPATCH(or_stub);  // 定义 or_stub 分发函数
DEFINE_DISPATCH(min_values_stub);  // 定义 min_values_stub 分发函数
DEFINE_DISPATCH(max_values_stub);  // 定义 max_values_stub 分发函数
DEFINE_DISPATCH(argmax_stub);  // 定义 argmax_stub 分发函数
DEFINE_DISPATCH(argmin_stub);  // 定义 argmin_stub 分发函数
DEFINE_DISPATCH(cumsum_stub);  // 定义 cumsum_stub 分发函数
DEFINE_DISPATCH(cumprod_stub);  // 定义 cumprod_stub 分发函数
DEFINE_DISPATCH(logcumsumexp_stub);  // 定义 logcumsumexp_stub 分发函数

Tensor _logcumsumexp_cpu(const Tensor& self, int64_t dim) {
  Tensor result = at::empty_like(self, MemoryFormat::Contiguous);  // 创建与输入张量相同尺寸的空张量
  return _logcumsumexp_out_cpu(self, dim, result);  // 调用 _logcumsumexp_out_cpu 函数计算对数累积和
}

Tensor& _logcumsumexp_out_cpu(const Tensor& self, int64_t dim, Tensor& result) {
  logcumsumexp_stub(self.device().type(), result, self, dim);  // 使用分发函数计算对数累积和
  return result;  // 返回结果张量的引用
}

Tensor logcumsumexp(const Tensor& self, int64_t dim) {
  auto result = [&]() {
    NoNamesGuard guard;  // 创建无名称保护
    return at::_logcumsumexp(self, dim);  // 计算对数累积和
  }();
  namedinference::propagate_names(result, self);  // 根据输入张量传播名称
  return result;  // 返回结果张量
}
// 检查结果张量和输入张量的标量类型、设备和布局是否相等
Tensor& logcumsumexp_out(const Tensor& self, int64_t dim, Tensor& result) {
  check_scalar_type_device_layout_equal(result, self);
  {
    // 禁用名称传播以防止函数内部操作影响结果张量的命名
    NoNamesGuard guard;
    // 调用底层的 logcumsumexp 函数，将结果存储在 result 张量中
    at::_logcumsumexp_out(result, self.toType(result.scalar_type()), dim);
  }
  // 将输入张量的命名信息传播到输出结果张量
  namedinference::propagate_names(result, self);
  // 返回计算后的结果张量
  return result;
}

// 实现累积操作的内部函数模板
template <class Stub>
void impl_func_cum_ops(
    const Tensor& self,
    int64_t dim,
    const Tensor& result,
    Stub& stub) {
  // 禁用名称传播以防止函数内部操作影响结果张量的命名
  NoNamesGuard guard;
  if (self.dim() == 0) {
    // 如果输入张量是标量，则将结果张量填充为输入张量的值
    result.fill_(self);
  } else if (self.numel() == 0) {
    // 如果输入张量为空，则将结果张量置零
    result.zero_();
  } else {
    // 对于非标量和非空输入张量，根据指定维度进行累积操作
    dim = maybe_wrap_dim(dim, self.dim());
    // 调用传入的 stub 函数，执行具体的累积操作
    stub(self.device().type(), result, self.to(result.scalar_type()), dim);
  }
}

// 实现累积和操作的具体函数（累加）
TORCH_IMPL_FUNC(cumsum_out)
(const Tensor& self,
 int64_t dim,
 std::optional<ScalarType> dtype,
 const Tensor& result) {
  // 调用内部实现函数处理累积操作
  impl_func_cum_ops(self, dim, result, cumsum_stub);
}

// 实现累积乘积操作的具体函数
TORCH_IMPL_FUNC(cumprod_out)
(const Tensor& self,
 int64_t dim,
 std::optional<ScalarType> dtype,
 const Tensor& result) {
  // 调用内部实现函数处理累积乘积操作
  impl_func_cum_ops(self, dim, result, cumprod_stub);
}

// 对给定张量在指定维度上进行反向累积求和操作
static Tensor reversed_cumsum(const Tensor& w, int64_t dim) {
  // 首先对张量在指定维度上进行翻转，然后计算累积和，最后再次翻转得到结果
  return w.flip(dim).cumsum(dim).flip(dim);
}

// 计算累积乘积的反向传播梯度
Tensor cumprod_backward(const Tensor& grad, const Tensor& input, int64_t dim, const Tensor& output) {
  /*
    以下是如何推导出对于任意输入的 O(n) 梯度公式。
    通过链式法则结合不同情况的基本应用来得到，我们假设 x 是 n 维向量，y = cumprod(x)。
    在实际实现中，我们需要通过一些掩码技巧来实现这些针对张量的公式。

    首先我们推导出 x[i] != 0 的情况下的公式。

    对于 F : R^n -> R 的损失函数（稍后我们将考虑更复杂的情况），
    我们有

    dF / dx_k = sum_j (dF / dy_j) * (dy_j / dx_k)   (1)

    其中 dF / dy_j 简单地是 grad_output[j]（假设一切都是一维的）。

    项 (dy_j / dx_k) 很容易看到是

    若 j >= k
      dy_j / dx_k = prod_{1 <= i <= j, i != k} x_i
    否则:
      dy_j / dx_k = 0

    注意，指示器 (j>=k) 可以通过将求和替换为 k <= j <= n 来消除。

    因此，
    dF / dx_k = sum_{k <= j <= n} grad_output[j] * (dy_j / dx_k)

    其中
    dy_j / dx_k = prod_{1 <= i <= j, i != k} x_i     (2)

    最后一个术语实际上就是累积乘积，省略了 k。因此，如果 x_k（输入）不为零，我们可以简单地表示为

    dy_j / dx_k = (prod_{1 <= i <= j} x_i) / x_k
                = y_j / x_k

    因此，

    dF / dx_k = sum_{k <= j <= n} grad_output[j] * y_j / x_k

    这个公式在输入每个 i 都不为零时才成立。

    现在假设输入中至少存在一个零。
    用 z1 表示第一个元素 1 <= z1 <= n，其中 input[z1] = 0，
    用 z2 表示第二个元素 z1 < z2 <= n，其中 input[z2] = 0，
  */
}
  /*
    (or z2 = n if there is just one zero in input)
    如果输入中只有一个零，则 z2 = n。

    We have three cases.
    我们有三种情况。

    k > z1:
    k > z1 时：
    Looking at (2), we see that dy_j / dx_k = 0, for j >= k, as these terms
    all include a x_{z1} which is zero. As such, dF / dx_k = 0 in this case
    根据（2），我们可以看到对于 j >= k，dy_j / dx_k = 0，因为这些项都包括一个值为零的 x_{z1}。因此，在这种情况下，dF / dx_k = 0。

    k < z1:
    k < z1 时：
    Reasoning as in the previous case, we see that for these elements we have that
    dF / dx_k = sum_{k <= j < z1} grad_output[j] * (dy_j / dx_k)
    与前一种情况的推理类似，对于这些元素，我们有 dF / dx_k = sum_{k <= j < z1} grad_output[j] * (dy_j / dx_k)

    as the terms of the sum for j in z1 <= j <= n are all zero
    因为对于 j 属于 z1 <= j <= n 的和项都是零。

    k = z1:
    k = z1 时：
    Similar to the case k < z1, we have that
    与 k < z1 的情况类似，我们有
    dF / dx_z1 = sum_{z1 <= j < z2} grad_output[j] * (dy_j / dx_z1)

    This case has a subtlety though. To compute (dy_j / dx_z1), we cannot use the formula
    但这种情况有一个微妙之处。计算 (dy_j / dx_z1) 时，我们不能使用公式
    dy_j / dx_z1 = y_j / x_z1

    as, y_j = x_z1 = 0 for j >= z1. We need to compute it with the formula for its derivative,
    因为对于 j >= z1，y_j = x_z1 = 0。我们需要使用其导数的公式来计算它，
    that is:
    即：

    dy_j / dx_z1 = prod(x[:z1]) * (grad_output[z1] + sum(grad_output[z1+1:z2] * cumprod(x[z1+1:z2])))
    dy_j / dx_z1 = prod(x[:z1]) * (grad_output[z1] + sum(grad_output[z1+1:z2] * cumprod(x[z1+1:z2])))

    When the inputs are complex, this is map is holomorphic. As such, to compute
    当输入是复数时，这个映射是全纯的。因此，要计算它的反向传播只是常规反向传播的共轭。这简化为共轭输入。我们也可以重用输出，因为映射是全纯的，
    cumprod(input.conj()) = cumprod(input).conj()

  */

  if (input.sym_numel() <= 1) {
    return grad;
  }
  如果输入的元素数量小于等于1，则直接返回梯度 grad。

  dim = at::maybe_wrap_dim(dim, input.dim());
  使用 maybe_wrap_dim 函数确保维度 dim 在合理范围内。

  const int64_t dim_size = input.sym_sizes()[dim].guard_int(__FILE__, __LINE__);
  获取指定维度的符号尺寸，并进行边界检查。

  if (dim_size == 1) {
    如果指定维度的尺寸为1，则直接返回梯度 grad。
    return grad;
  }

  // To enable complex support.
  // 开启复数支持。

  // From this line on `input_conj` and output_conj`
  // are interchangeable with `input` and `output`.
  从这一行开始，`input_conj` 和 `output_conj` 可以与 `input` 和 `output` 互换使用。

  auto input_conj = input.conj();
  创建输入的共轭 input_conj。

  auto output_conj = output.conj();
  创建输出的共轭 output_conj。

  // For Composite Compliance, we always choose the slower but composite compliant path.
  // 为了符合复合兼容性，我们总是选择较慢但符合复合标准的路径。

  bool are_inputs_tensors_sublcass = areAnyTensorSubclassLike({input, grad, output});
  检查输入、梯度和输出是否为任何张量子类。

  const auto w = output_conj * grad;
  计算 w = output_conj * grad。

  const auto is_zero = input == 0;
  检查 input 是否为零的元素。

  if (!are_inputs_tensors_sublcass) {
    如果输入不是张量子类，
    if (is_zero.any().item<uint8_t>() == 0) {
      如果输入中存在非零元素，
      return reversed_cumsum(w, dim).div(input_conj);
      返回 w 沿指定维度的反向累加和除以 input_conj。
    }
  }

  // If we are not computing a second order gradient, we can use an
  // O(n) implementation. The derivative of this implementation is _not_
  // the second derivative of cumprod. As such, we fallback to a less efficient
  // O(n^2) implementation when at::GradMode::is_enabled().
  如果不需要计算二阶梯度，并且未启用梯度模式，
  // n.b. This could probably be implemented much faster with a kernel
  注意：这可能通过内核实现得更快。

  // From here on we need to use some mask gymnastics to
  从这里开始，我们需要使用一些掩码技巧来

  // account for the tensorial dimensions
  考虑张量的维度。

  // We do a cumsum of the zeros along the dimension.
  在维度上对零进行累加和。

  // For a vector is_zero = [False, True, False, True, False]
  对于向量 is_zero = [False, True, False, True, False]

  // we would have cumsum = [0, 1, 1, 2, 2]
  我们会得到累加和 cumsum = [0, 1, 1, 2, 2]

  // As such we have (in python code for simplicity)
  因此，我们有（为简化起见使用 Python 代码）

  // The mask for the range [0, z1):
  // 对于范围 [0, z1) 的掩码

  // cumsum == 0
  cumsum == 0

  // The indices of the first zero z1 and zeros when
  第一个零 z1 的索引以及当零时的索引
  // there is no first zero:
  没有第一个零时：
    // indices = (cumsum == 1).max(dim, keepdim=True).indices
    // 确定第一个零的位置：
    // 首先计算在 dim 维度上累积和等于 1 的位置索引
    const auto indices = (cumsum == 1).max(dim, /*keepdim*/ true).indices;
    // 生成一个与 indices 形状相同的全零张量作为掩码
    // 然后在 dim 维度上，将 indices 对应位置填充为 1，并与 cumsum == 1 逻辑与
    // 这一步骤处理了当没有第一个零的情况
    const auto mask = at::zeros_like(indices).scatter_(dim, indices, /*src*/ 1.).logical_and_(cumsum == 1);

    // 初始化 grad_input 为与输入相同大小的零张量
    Tensor grad_input = at::zeros_like(input.sym_sizes(), grad.options());
    // 取出 cumsum 中为 0 的部分对应的 deriv[grad] 并填充到 grad_input 的相应位置
    grad_input.masked_scatter_(mask,
        reversed_cumsum(w.masked_fill(~mask, 0.), dim).div_(input_conj).masked_select(mask));

    // 选取从第一个零到第二个零之间的部分 [z1, z2)
    mask = cumsum == 1;

    // 当 k = z1 时，选择第一个零 [z1]
    // 通过 max 函数确定第一个零的索引，再通过 index_fill_ 生成对应的掩码
    // 当该切片中不存在零时，max 函数将返回索引 0
    const auto first_zero_index = std::get<1>(mask.max(dim, /*keepdim*/ true));
    const auto first_zero_mask = at::zeros_like(mask)
                                  .scatter_(dim, first_zero_index, /*src*/ 1)
                                  .logical_and_(mask);

    // 选择第一个零到第二个零之间的部分 (z1, z2)
    mask &= ~first_zero_mask;

    // 计算 dy_j / dx_z1 = sum(cumprod(input[z1+1:z2] * grad[z1+1:z2])) * prod(output[z1-1])
    // 注意，由于 gather 不支持负索引，因此需要对 (first_zero_index - 1) 进行 relu_ 处理
    // 将计算结果填充到 grad_input 的相应位置 grad_input[z1]
    grad_input.masked_scatter_(first_zero_mask,
                               input_conj.masked_fill(~mask, 1.).cumprod(dim)
                                    .mul_(grad.masked_fill(cumsum != 1, 0.))
                                    .sum(dim, /*keepdim*/ true)
                                    .mul_(at::gather(output_conj, dim, (first_zero_index - 1).relu_())
                                          .masked_fill_(first_zero_index == 0, 1.))
                                    .masked_select(first_zero_mask));

    return grad_input;
  } else { // GradMode::enabled()
    /*
    如果输入非零，我们需要使用公式 (2) 计算 dy_j / dx_k，
    通过调用代码 omitted_products 实现。

    代码中计算方式简单地通过以下方式：

    prod_{1 <= i <= j, i != k} x_i
        = (prod_{1 <= i <= k} x_i) * (prod_{k + 1 <= i <= j} x_i)

    第一个项作为 prods_until_k 计算，由于不依赖于 j，易于向量化。

    第二项（由 j 索引）是 x_{k+1}, x_{k+2}, ..., x_n 的累积乘积，称为 prods_from_k_plus_1，
    通过 cumprod 计算得到。

    为了正确向量化这一过程，需要将其添加到...
    */
    // 定义梯度输入张量
    Tensor grad_input;
    // 对于复合一致性，我们将在梯度片段上使用 at::stack，因此使用向量存储梯度输入
    std::vector<Tensor> grad_inputs;
    // 如果输入是张量子类，则预留空间以提高效率
    if (are_inputs_tensors_sublcass) {
      grad_inputs.reserve(dim_size);
    } else {
      // 否则，创建与输入相同大小的零张量作为梯度输入
      grad_input = at::zeros(input.sizes(), grad.options());
    }
    // 创建大小为 ones_size 的张量，将维度 dim 上的尺寸设置为 1
    auto ones_size = input.sym_sizes().vec();
    ones_size[dim] = 1;
    const Tensor ones = at::ones({1}, grad.options()).expand_symint(ones_size);
    // 定义用于存储从 k+1 开始的累积乘积的张量
    Tensor prods_from_k_plus_1;
    // 定义用于存储省略的乘积的张量
    Tensor omitted_products;
    // 遍历维度 dim 的范围
    for (const auto k : c10::irange(dim_size)) {
      // 如果 k 为 0，计算从 k+1 开始的累积乘积，并包括 ones，形成省略的乘积
      if (k == 0) {
        prods_from_k_plus_1 = at::cumprod(input_conj.slice(dim, k + 1), dim);
        omitted_products = at::cat({ones, std::move(prods_from_k_plus_1)}, dim);
      // 如果 k 为 dim_size - 1，计算直到 k 的乘积，形成省略的乘积
      } else if (k == dim_size - 1) {
        const Tensor prods_until_k = at::prod(input_conj.slice(dim, 0, k), dim, true);
        omitted_products = prods_until_k;
      // 对于其他情况，计算直到 k 的乘积和从 k+1 开始的累积乘积，并形成省略的乘积
      } else {
        const Tensor prods_until_k = at::prod(input_conj.slice(dim, 0, k), dim, true);
        prods_from_k_plus_1 = at::cumprod(input_conj.slice(dim, k+1), dim);
        omitted_products = prods_until_k.expand_as(prods_from_k_plus_1) * prods_from_k_plus_1;
        omitted_products = at::cat({prods_until_k, omitted_products}, dim);
      }

      // 检查省略的乘积在维度 dim 上的符号尺寸是否为 dim_size - k
      TORCH_CHECK(omitted_products.sym_size(dim) == dim_size - k);

      // 计算在维度 dim 上的梯度片段乘以省略的乘积后的和
      auto grad_slice = at::sum(grad.slice(dim, k) * omitted_products, dim);
      // 如果输入是张量子类，则将计算的梯度片段添加到向量中
      if (are_inputs_tensors_sublcass) {
        grad_inputs.push_back(grad_slice);
      // 否则，将计算的梯度片段复制到 grad_input 的相应位置
      } else {
        grad_input.select(dim, k).copy_(grad_slice);
      }
    }

    // 返回结果，如果输入是张量子类，则使用 at::stack 将所有梯度片段堆叠在一起，否则返回 grad_input
    return are_inputs_tensors_sublcass ? at::stack(grad_inputs, dim) : std::move(grad_input);
}

// 实现 std::is_nan<IntegralType> 用于 MSVC。
namespace {
#ifdef _MSC_VER
// 如果 T 是整数类型，返回 false
template<typename T>
inline typename std::enable_if<std::is_integral<T>::value, bool>::type isnan_(T x) {
  return false;
}
// 如果 T 不是整数类型，调用 std::isnan 检查是否为 NaN
template<typename T>
inline typename std::enable_if<!std::is_integral<T>::value, bool>::type isnan_(T x) {
  return std::isnan(x);
}
#else
// 非 MSVC 平台上的通用实现，调用 std::isnan 检查是否为 NaN
template<typename T>
inline bool isnan_(T x) {
  return std::isnan(x);
}
#endif
}

// 辅助函数模板 cummax_cummin_helper
template<typename T1, typename T2, typename Operation>
void cummax_cummin_helper(const T1* self_data, T1* values_data, T2* indices_data,
          int self_dim_size, int self_stride, int values_stride, int indices_stride) {
      // 创建操作对象
      Operation op;
      // 初始化输出值为 self_data 的第一个元素
      T1 out = c10::load(self_data);
      // 初始化索引为 0
      int idx = 0;
      // 遍历 self_data 的每个元素
      for (const auto i : c10::irange(self_dim_size)) {
        // 加载当前元素
        T1 curr_elem = c10::load(&self_data[i*self_stride]);
        // 如果当前元素是 NaN 或者 (输出值不是 NaN 且当前元素满足操作条件)
        if(isnan_(curr_elem) || (!isnan_(out) && op(curr_elem, out))) {
            // 更新输出值为当前元素
            out = curr_elem;
            // 更新索引为当前位置 i
            idx = i;
        }
        // 将输出值写入 values_data 的相应位置
        values_data[i*values_stride] = out;
        // 将索引写入 indices_data 的相应位置
        indices_data[i*indices_stride] = idx;
      }
}

// CPU 版本的 cummax 辅助函数
void cummax_helper_cpu(const Tensor& self, Tensor& values, Tensor& indices, int64_t dim) {
  // 根据 self 的数据类型进行分发处理
  AT_DISPATCH_ALL_TYPES_AND3(kBool, kBFloat16, kHalf,
    self.scalar_type(), "cummax_cpu",
    [&] {
      // 调用 tensor_dim_apply3 函数，处理 self, values, indices，使用 std::greater_equal 进行操作
      at::native::tensor_dim_apply3<scalar_t, int64_t>(self, values, indices, dim, cummax_cummin_helper<scalar_t, int64_t, std::greater_equal<scalar_t>>);
    });
}

// cummax 的输出版本，修改 values 和 indices 的值并返回它们
std::tuple<Tensor&, Tensor&> cummax_out(const Tensor& self, int64_t dim, Tensor& values, Tensor& indices) {
  // 检查 values 和 self 的标量类型、设备和布局是否相同
  check_scalar_type_device_layout_equal(values, self);
  // 检查 indices 和一个空张量的标量类型是否相同
  check_scalar_type_device_layout_equal(indices, at::empty({0}, self.options().dtype(at::kLong)));
  {
    // 未命名保护，可能是一个上下文保护对象
    NoNamesGuard guard;
    // 调整 values 和 indices 的输出尺寸为 self 的尺寸
    at::native::resize_output(values, self.sizes());
    at::native::resize_output(indices, self.sizes());
    // 如果 self 的维度数为 0
    if(self.dim() == 0) {
      // 将 values 填充为 self 的值
      values.fill_(self);
      // 将 indices 填充为 0
      indices.fill_(0);
    } else if(self.numel() != 0) {
      // 否则，处理维度 dim 上的累积最大值操作
      dim = maybe_wrap_dim(dim, self.dim());
      at::_cummax_helper(self, values, indices, dim);
    }
  }
  // 传播 values 和 indices 的名称属性
  namedinference::propagate_names(values, self);
  namedinference::propagate_names(indices, self);
  // 返回 values 和 indices
  return std::forward_as_tuple(values, indices);
}

// 计算张量 self 沿维度 dim 的累积最大值
std::tuple<Tensor, Tensor> cummax(const Tensor& self, int64_t dim) {
  // 创建空张量 values 和 indices
  auto values = at::empty(self.sizes(), self.options());
  auto indices = at::empty(self.sizes(), self.options().dtype(at::kLong));
  // 调用 cummax_out 函数，计算累积最大值并将结果存储在 values 和 indices 中
  at::cummax_out(values, indices, self, dim);
  // 返回 values 和 indices 的元组
  return std::make_tuple(values, indices);
}

// CPU 版本的 cummin 辅助函数
void cummin_helper_cpu(const Tensor& self, Tensor& values, Tensor& indices, int64_t dim) {
  // 根据 self 的数据类型进行分发处理
  AT_DISPATCH_ALL_TYPES_AND3(kBool, kBFloat16, kHalf,
    self.scalar_type(), "cummin_cpu",
    [&] {
      // 调用 tensor_dim_apply3 函数，处理 self, values, indices，使用 std::less_equal 进行操作
      at::native::tensor_dim_apply3<scalar_t, int64_t>(self, values, indices, dim, cummax_cummin_helper<scalar_t, int64_t, std::less_equal<scalar_t>>);
    });
}
// 返回一个 tuple，包含对给定张量按指定维度进行累积最小值操作后的结果张量和索引张量
std::tuple<Tensor&, Tensor&> cummin_out(const Tensor& self, int64_t dim, Tensor& values, Tensor& indices) {
  // 检查 values 和 indices 张量的数据类型、设备和布局是否与 self 张量相同
  check_scalar_type_device_layout_equal(values, self);
  check_scalar_type_device_layout_equal(indices, at::empty({0}, self.options().dtype(at::kLong)));
  {
    // 创建 NoNamesGuard 对象，用于处理无命名情况
    NoNamesGuard guard;
    // 调整 values 和 indices 张量的大小以匹配 self 张量的大小
    at::native::resize_output(values, self.sizes());
    at::native::resize_output(indices, self.sizes());
    if(self.dim() == 0) {
      // 如果 self 张量是零维，则将 values 填充为 self，indices 填充为 0
      values.fill_(self);
      indices.fill_(0);
    } else if(self.numel() != 0) {
      // 否则，根据指定的维度 dim 对 self 进行累积最小值操作，并将结果存储在 values 和 indices 中
      dim = maybe_wrap_dim(dim, self.dim());
      at::_cummin_helper(self, values, indices, dim);
    }
  }
  // 传播 values 和 indices 张量的命名信息，以保持命名一致性
  namedinference::propagate_names(values, self);
  namedinference::propagate_names(indices, self);
  // 返回 values 和 indices 的 tuple
  return std::forward_as_tuple(values, indices);
}

// 返回一个 tuple，包含对给定张量按指定维度进行累积最小值操作后的结果张量和索引张量
std::tuple<Tensor, Tensor> cummin(const Tensor& self, int64_t dim) {
  // 创建 values 和 indices 张量，用于存储累积最小值操作的结果
  auto values = at::empty(self.sizes(), self.options());
  auto indices = at::empty(self.sizes(), self.options().dtype(at::kLong));
  // 调用 cummin_out 函数，对 self 进行累积最小值操作，并返回 values 和 indices 的 tuple
  at::cummin_out(values, indices, self, dim);
  return std::make_tuple(values, indices);
}

// 计算给定梯度、输入张量和索引张量在指定维度上的累积最大最小值的反向传播结果
Tensor cummaxmin_backward(const Tensor& grad, const Tensor& input, const Tensor& indices, int64_t dim) {
  if (input.sym_numel() == 0) {
    // 如果 input 的符号元素数为 0，则直接返回 input 张量
    return input;
  }
  // 创建与 input 形状相同的零张量 result
  auto result = at::zeros_symint(input.sym_sizes(), input.options());

  // 如果 indices 或 grad 是张量子类的情况下，使用 out-of-place 的 scatter_add 变体
  if (areAnyTensorSubclassLike({indices, grad})) {
    // 返回在 dim 维度上使用 scatter_add 方法得到的结果
    return result.scatter_add(dim, indices, grad);
  }
  // 否则，在 dim 维度上使用 inplace 的 scatter_add_ 方法得到结果并返回
  return result.scatter_add_(dim, indices, grad);
}

// 在指定维度上，处理 diff 函数中的前置和后置操作的辅助函数
static Tensor prepend_append_on_dim(const Tensor& self, const std::optional<Tensor>& prepend, const std::optional<Tensor>& append, int64_t dim) {
  // 断言：要求至少有一个 prepend 或 append 有值
  TORCH_INTERNAL_ASSERT(prepend.has_value() || append.has_value(), "either prepend or append must be have value");
  if (!prepend.has_value() && append.has_value()) {
    // 如果只有 append 有值，则在 dim 维度上将 append 与 self 连接
    return at::cat({self, append.value()}, dim);
  } else if (prepend.has_value() && !append.has_value()) {
    // 如果只有 prepend 有值，则在 dim 维度上将 prepend 与 self 连接
    return at::cat({prepend.value(), self}, dim);
  } else {
    // 如果都有值，则在 dim 维度上将 prepend、self 和 append 连接
    return at::cat({prepend.value(), self, append.value()}, dim);
  }
}

// 在 diff 函数中，检查待连接张量的形状是否与输入张量兼容的辅助函数
static inline void diff_check_compatible_shape(const Tensor& self, const std::optional<Tensor>& other, int64_t dim) {
  // 如果 other 有值，则检查其维度是否与 self 相同
  if (other.has_value()) {
    // 对 dim 进行可能的维度包装，并进行维度匹配检查
    int64_t wrapped_dim = maybe_wrap_dim(dim, self.dim(), false);

    TORCH_CHECK(
        other.value().dim() == self.dim(),
        "diff expects prepend or append to be the same dimension as input");
    // 对于指定范围内的每个索引 i，在循环中执行以下操作
    for (const auto i : c10::irange(other.value().dim())) {
      // 检查其他张量的符号尺寸与当前张量的符号尺寸相等，或者当前索引 i 等于 wrapped_dim
      TORCH_CHECK(
          other.value().sym_size(i) == self.sym_size(i) || i == wrapped_dim,
          // 如果不相等，则输出错误信息，指示预期的尺寸
          "diff expects the shape of tensor to prepend or append to match that of"
          " input except along the differencing dimension;"
          // 输出当前输入张量的尺寸信息
          " input.size(", i, ") = ", self.sym_size(i), ", but got"
          // 输出其他张量的尺寸信息
          " tensor.size(", i, ") = ", other.value().sym_size(i));
    }
}

// 这是一个静态内联函数，用于检查 diff 函数的参数是否有效
static inline void diff_check(const Tensor& self, int64_t n, int64_t dim, const std::optional<Tensor>& prepend, const std::optional<Tensor>& append) {
  // 检查输入张量是否至少为一维
  TORCH_CHECK(
      self.dim() >= 1,
      "diff expects input to be at least one-dimensional");

  // 检查 n 是否为非负数
  TORCH_CHECK(
      n >= 0,
      "order must be non-negative but got ", n);

  // 检查 prepend 和 append 张量是否与主张量在指定维度上的形状兼容
  diff_check_compatible_shape(self, prepend, dim);
  diff_check_compatible_shape(self, append, dim);
}

// 这是 diff 函数的辅助函数，用于执行差分操作
static inline Tensor diff_helper(const Tensor& self, int64_t n, int64_t dim) {
  // 如果 n 为 0，则直接返回与 self 形状相同的零张量
  if (n == 0) {
    auto result = at::zeros_like(self);
    result.copy_(self);
    return result;
  }

  // 计算输出张量在指定维度上的长度
  auto out_len = self.sym_size(dim) - 1;
  auto result = self;
  bool is_kBool = (self.dtype() == at::kBool);

  // 如果 n 大于指定维度上的长度，则将 n 限制为指定维度上的长度
  n = n > self.sym_size(dim) ? self.sym_size(dim).guard_int(__FILE__, __LINE__) : n;

  // 执行 n 次差分操作
  for (C10_UNUSED const auto i : c10::irange(n)) {
    if (is_kBool) {
      // 如果张量类型为布尔型，则使用逻辑异或进行差分计算
      result = at::logical_xor(
        at::narrow_symint(result, dim, 1, out_len),
        at::narrow_symint(result, dim, 0, out_len)
      );
    } else {
      // 否则，使用普通的数值减法进行差分计算
      result = at::narrow_symint(result, dim, 1, out_len) - at::narrow_symint(result, dim, 0, out_len);
    }
    out_len = out_len - 1;
  }

  return result;
}

// 这是 diff 函数的实现，用于处理主函数调用和条件分支
Tensor diff(const Tensor& self, int64_t n, int64_t dim, const std::optional<Tensor>& prepend, const std::optional<Tensor>& append) {
  // 执行参数有效性检查
  diff_check(self, n, dim, prepend, append);

  // 如果没有前置和后置张量，或者 n 为 0，则调用 diff_helper 函数进行差分计算
  if ((!prepend.has_value() && !append.has_value()) || n == 0) {
    return diff_helper(self, n, dim);
  } else {
    // 否则，调用 prepend_append_on_dim 函数处理前置和后置张量，并调用 diff_helper 函数进行差分计算
    auto a = prepend_append_on_dim(self, prepend, append, dim);
    return diff_helper(a, n, dim);
  }
}

// 这是 diff_out 函数的辅助函数，用于在输出张量已存在的情况下执行差分操作
static inline Tensor& diff_out_helper(const Tensor& self, int64_t n, int64_t dim, Tensor& result) {
  // 如果 n 为 0，则将结果张量调整为与 self 相同形状的零张量，并将 self 复制到结果张量
  if (n == 0) {
    // 如果需要，调整输出张量为与 self 形状相同的零张量
    if (resize_output_check_symint(result, self.sym_sizes())) {
      result.resize__symint(self.sym_sizes());
    }
    // 检查并保证输出张量与 self 具有相同的标量类型、设备和布局
    check_scalar_type_device_layout_equal(result, self);
    // 将 self 的内容复制到结果张量中
    return result.copy_(self);
  }

  // 如果 n 大于指定维度上的长度，则将 n 限制为指定维度上的长度
  n = n > self.sym_size(dim) ? self.sym_size(dim).guard_int(__FILE__, __LINE__) : n;
  const auto out_len = self.sym_size(dim) - n;
  auto prev_result = self;

  // 如果 n 大于 1，则先计算 n-1 次差分结果作为 prev_result
  if (n > 1) {
    prev_result = diff_helper(self, n - 1, dim);
  }

  // 根据张量类型执行差分计算，并将结果写入输出张量 result 中
  if (self.dtype() == at::kBool) {
    at::logical_xor_out(
      result,
      at::narrow_symint(prev_result, dim, 1, out_len),
      at::narrow_symint(prev_result, dim, 0, out_len)
    );
  } else {
    at::sub_out(
      result,
      at::narrow_symint(prev_result, dim, 1, out_len),
      at::narrow_symint(prev_result, dim, 0, out_len)
    );
  }

  return result;
}

// 这是 diff_out 函数的实现，用于处理主函数调用和条件分支
Tensor& diff_out(const Tensor& self, int64_t n, int64_t dim, const std::optional<Tensor>& prepend, const std::optional<Tensor>& append, Tensor& result) {
  // 执行参数有效性检查
  diff_check(self, n, dim, prepend, append);

  // 如果没有前置和后置张量，或者 n 为 0，则调用 diff_out_helper 函数进行差分计算
  if ((!prepend.has_value() && !append.has_value()) || n == 0) {
    return diff_out_helper(self, n, dim, result);
  } else {
    // 否则，调用 prepend_append_on_dim 函数处理前置和后置张量，并调用 diff_out_helper 函数进行差分计算
    auto a = prepend_append_on_dim(self, prepend, append, dim);
    return diff_out_helper(a, n, dim, result);
  }
}
    return diff_out_helper(a, n, dim, result);



// 调用名为 diff_out_helper 的函数，传入参数 a, n, dim, result，并返回其返回值
return diff_out_helper(a, n, dim, result);


这行代码是一个函数返回语句，调用了名为 `diff_out_helper` 的函数，并将参数 `a`, `n`, `dim`, `result` 传递给它。函数 `diff_out_helper` 可能用于计算或处理某些数据，并返回一个值，这个值被这个函数返回语句所返回。
static void pre_check_gradient(const Tensor& self, std::optional<int64_t> spacing_size, at::OptionalIntArrayRef dim,  int64_t edge_order) {
  // Helper for gradient function to make sure input data satisfies prerequisites

  // 检查输入张量类型是否为 uint8，因为 torch.gradient 不支持 uint8 输入
  TORCH_CHECK(self.scalar_type() != ScalarType::Byte, "torch.gradient does not support uint8 input.");

  // 如果 spacing_size 有值而 dim 没有值
  if (spacing_size.has_value() && !dim.has_value()) {
    // 如果 spacing 被指定为标量，调用该函数的调用者会创建一个预期大小的间距向量，这个检查将通过
    TORCH_CHECK(spacing_size.value() == self.dim(),
      "torch.gradient expected spacing to be unspecified, a scalar, or a list ",
      "of length equal to 'self.dim() = ", self.dim(), "', since dim argument ",
      "was not given, but got a list of length ", spacing_size.value());
  }

  // 如果 spacing_size 和 dim 都有值
  if (spacing_size.has_value() && dim.has_value()) {
    // 检查 spacing 是否未指定，或者是标量，或者其 spacing 和 dim 参数具有相同的长度
    TORCH_CHECK(spacing_size.value() == static_cast<int64_t>(dim.value().size()),
      "torch.gradient expected spacing to be unspecified, a scalar or it's spacing and dim arguments to have the same length, but got a spacing argument of length ", spacing_size.value(), " and a dim argument of length ", dim.value().size(), "." );
  }

  // 检查 edge_order 是否为 1 或 2，因为 torch.gradient 仅支持这两个值
  TORCH_CHECK(edge_order == 1 || edge_order == 2, "torch.gradient only supports edge_order=1 and edge_order=2.");

  // 如果 dim 有值
  if (dim.has_value()) {
    // 调用 dim_list_to_bitset 函数来检查 dim 参数是否满足先决条件
    // 该函数的输出不用于梯度的计算
    dim_list_to_bitset(dim.value(), self.dim());

    // 遍历 dim 参数的每个维度
    for (const auto i : c10::irange(dim.value().size())) {
      // 检查每个维度大小是否至少为 edge_order+1
      TORCH_CHECK(self.size(dim.value()[i]) >= edge_order + 1, "torch.gradient expected each dimension size to be at least edge_order+1");
    }
  } else {
    // 如果 dim 没有值，则对输入张量的每个维度进行检查
    for (const auto i : c10::irange(self.dim())) {
      // 检查每个维度大小是否至少为 edge_order+1
      TORCH_CHECK(self.size(i) >= edge_order + 1, "torch.gradient expected each dimension size to be at least edge_order+1");
    }
  }
}
    auto b = ( (dx2-dx1) / (dx1*dx2)       ).reshape(shape);
    // 计算系数 b，并将结果 reshape 成指定的 shape

    auto c = (    dx1    / (dx2*(dx1+dx2)) ).reshape(shape);
    // 计算系数 c，并将结果 reshape 成指定的 shape

    auto center = a * at::slice(self, direction, 0, -2) + b * at::slice(self, direction , 1, -1) + c * at::slice(self, direction, 2);
    // 计算中心值，使用给定的系数 a, b, c，分别乘以不同位置的 self tensor 切片，以计算中心值

    if (edge_order == 1) {
        // 如果边缘顺序为 1
        prepend = (at::slice(self, direction, 1, 2  ) - at::slice(self, direction, 0, 1   )) / ax_dx[0]  ;
        // 计算前置值，使用给定的 ax_dx 数组计算

        append  = (at::slice(self, direction, -1    ) - at::slice(self, direction, -2, -1 )) / ax_dx[-1] ;
        // 计算后置值，使用给定的 ax_dx 数组计算

    } else if (edge_order == 2) {
        // 如果边缘顺序为 2
        a =-(2.0 * ax_dx[0] + ax_dx[1]) / (ax_dx[0] * (ax_dx[0] + ax_dx[1])) ;
        // 计算系数 a，用于边缘顺序为 2 的情况

        b = (      ax_dx[0] + ax_dx[1]) / (ax_dx[0] * ax_dx[1])       ;
        // 计算系数 b，用于边缘顺序为 2 的情况

        c = (     -ax_dx[0]           ) / (ax_dx[1] * (ax_dx[0] + ax_dx[1]));
        // 计算系数 c，用于边缘顺序为 2 的情况

        prepend = a * at::slice(self, direction, 0, 1) + b * at::slice(self, direction, 1, 2) + c * at::slice(self, direction, 2, 3);
        // 计算前置值，使用给定的系数 a, b, c 和 self tensor 的切片

        a = (    ax_dx[-1]            ) / (ax_dx[-2] * (ax_dx[-1] + ax_dx[-2]));
        // 计算系数 a，用于边缘顺序为 2 的情况

        b =-(    ax_dx[-1] + ax_dx[-2]) / (ax_dx[-1] * ax_dx[-2]);
        // 计算系数 b，用于边缘顺序为 2 的情况

        c = (2 * ax_dx[-1] + ax_dx[-2]) / (ax_dx[-1] * (ax_dx[-1] + ax_dx[-2]));
        // 计算系数 c，用于边缘顺序为 2 的情况

        append = a * at::slice(self, direction, -3, -2) + b * at::slice(self, direction, -2, -1) + c * at::slice(self, direction, -1);
        // 计算后置值，使用给定的系数 a, b, c 和 self tensor 的切片
    }

    result.emplace_back(prepend_append_on_dim(center, prepend, append, direction));
    // 将计算得到的 center, prepend, append 和 direction 参数传递给函数 prepend_append_on_dim，并将结果添加到 result 的末尾
  }
  return result;
// 返回最终的 result 结果
}

// 辅助函数：计算浮点数场的梯度
static std::vector<Tensor> gradient_helper_float(const Tensor& self, ArrayRef<Scalar> spacing, IntArrayRef dim, int64_t edge_order) {
  // 结果向量，用于存储计算得到的梯度张量
  std::vector<Tensor> result;
  // 遍历维度列表
  for (const auto i : c10::irange(dim.size())) {
      // 获取可能包裹后的维度方向
      int64_t direction = maybe_wrap_dim(dim[i], self.dim());
      // 获取当前维度对应的间隔
      const auto& ax_dx = spacing[i];
      // 前导和后续梯度张量
      Tensor prepend, append;
      // 计算中心梯度，这里的`slice`函数用于对张量进行切片操作
      auto center  = (at::slice(self,direction, 2   ) - at::slice(self, direction, 0, -2 ) ) / ax_dx;
      // 根据边缘顺序计算前导和后续梯度张量
      if (edge_order==1) {
        prepend = (at::slice(self,direction, 1, 2) - at::slice(self, direction, 0, 1  ) ) / ax_dx;
        append  = (at::slice(self,direction, -1  ) - at::slice(self, direction, -2, -1) ) / ax_dx ;
      } else if (edge_order==2) {
        prepend = (-1.5 * at::slice(self, direction, 0, 1) + 2 * at::slice(self, direction, 1, 2)   - 0.5 * at::slice(self, direction, 2, 3))/ ax_dx;
        append = (0.5 * at::slice(self, direction, -3, -2) - 2 * at::slice(self, direction, -2, -1) + 1.5 * at::slice(self, direction, -1))  / ax_dx;
      }

      // 将计算得到的梯度张量加入结果向量
      result.emplace_back(prepend_append_on_dim(center/2, prepend, append, direction));
  }
  // 返回计算得到的梯度张量向量
  return result;
}

// 辅助函数：处理梯度计算的维度预处理
static std::vector<int64_t> gradient_dim_preprocess(const Tensor& self, std::optional<int64_t> dim) {
  // 如果梯度维度作为整数提供，则仅在此方向上计算梯度
  // 如果根本没有提供，则对所有方向都计算梯度
  // 如果梯度维度作为整数向量提供，则此函数不会被调用
  if (dim.has_value()) {
    return std::vector<int64_t>{dim.value()};
  }

  // 生成包含所有维度的索引向量
  std::vector<int64_t> axis(self.dim());
  std::iota(axis.begin(), axis.end(), 0);
  return axis;
}

// 计算梯度函数：处理整数维度的情况
std::vector<Tensor> gradient(const Tensor& self, TensorList coordinates, IntArrayRef dim, int64_t edge_order) {
    // 预检查梯度计算参数
    pre_check_gradient(self,
                       std::optional<int64_t>(coordinates.size()),
                       at::OptionalIntArrayRef(dim),
                       edge_order);
    // 调用梯度计算的具体实现函数
    return gradient_helper(self, coordinates, dim, edge_order);
}

// 计算梯度函数：处理可选整数维度的情况
std::vector<Tensor> gradient(const Tensor& self, TensorList coordinates, std::optional<int64_t> dim, int64_t edge_order) {
  // 预处理梯度计算维度
  const auto processed_dim = gradient_dim_preprocess(self, dim);
  // 预检查梯度计算参数
  pre_check_gradient(self,
                     std::optional<int64_t>(coordinates.size()),
                     dim.has_value() ? at::OptionalIntArrayRef(processed_dim) : c10::nullopt,
                     edge_order);
  // 调用梯度计算的具体实现函数
  return gradient_helper(self, coordinates, processed_dim, edge_order);
}

// 计算梯度函数：处理浮点数场的情况
std::vector<Tensor> gradient(const Tensor& self, c10::ArrayRef<Scalar> spacing, IntArrayRef dim, int64_t edge_order) {
  // 预检查梯度计算参数
  pre_check_gradient(self,
                     std::optional<int64_t>(spacing.size()),
                     at::OptionalIntArrayRef(dim),
                     edge_order);
  // 调用浮点数场梯度计算的具体实现函数
  return gradient_helper_float(self, spacing, dim, edge_order);
}
// 计算张量的梯度
std::vector<Tensor> gradient(const Tensor& self, ArrayRef<Scalar> spacing, std::optional<int64_t> dim, int64_t edge_order) {
  // 预处理梯度计算的维度参数，处理可能的空值情况
  const auto processed_dim = gradient_dim_preprocess(self, dim);
  // 执行梯度计算前的检查，验证参数合法性
  pre_check_gradient(self,
                     std::optional<int64_t>(spacing.size()),
                     dim.has_value() ? at::OptionalIntArrayRef(processed_dim) : c10::nullopt,
                     edge_order);
  // 调用梯度计算的辅助函数，返回梯度张量数组
  return gradient_helper_float(self, spacing, processed_dim, edge_order);
}

// 计算张量的梯度
std::vector<Tensor> gradient(const Tensor& self, const Scalar& unit_size, IntArrayRef dim, int64_t edge_order) {
  // 当间距以标量形式给出时，为每个给定维度元素的间距值设置单位大小
  std::vector<Scalar> spacing(dim.size(), unit_size);
  // 执行梯度计算前的检查，验证参数合法性
  pre_check_gradient(self,
                     std::optional<int64_t>(spacing.size()),
                     at::OptionalIntArrayRef(dim),
                     edge_order);
  // 调用梯度计算的辅助函数，返回梯度张量数组
  return gradient_helper_float(self, spacing, dim, edge_order);
}

// 计算张量的梯度
std::vector<Tensor> gradient(const Tensor& self, const std::optional<Scalar>& unit_size, std::optional<int64_t> dim, int64_t edge_order) {
  // 预处理梯度计算的维度参数，处理可能的空值情况
  const auto processed_dim = gradient_dim_preprocess(self, dim);
  // 当未提供单位大小时，默认为1
  // 当维度具有整数值时，意味着我们希望在特定方向上计算梯度；但未提供时，表示我们希望在所有方向上计算梯度
  std::vector<Scalar> spacing(dim.has_value() ? 1 : self.dim(),
                              unit_size.has_value() ? unit_size.value() : 1.0);
  // 执行梯度计算前的检查，验证参数合法性
  pre_check_gradient(self,
                     unit_size.has_value() ? std::optional<int64_t>(spacing.size()) : c10::nullopt,
                     dim.has_value() ? at::OptionalIntArrayRef(processed_dim) : c10::nullopt,
                     edge_order);
  // 调用梯度计算的辅助函数，返回梯度张量数组
  return gradient_helper_float(self, spacing, processed_dim, edge_order);
}

// 计算张量的梯度
std::vector<Tensor> gradient(const Tensor& self, IntArrayRef dim, int64_t edge_order) {
  // 默认单位大小为1.0，间距数组为给定维度大小
  std::vector<Scalar> spacing(dim.size(), 1.0);
  // 执行梯度计算前的检查，验证参数合法性
  pre_check_gradient(self,
                     std::optional<int64_t>(spacing.size()),
                     at::OptionalIntArrayRef(dim),
                     edge_order);
  // 调用梯度计算的辅助函数，返回梯度张量数组
  return gradient_helper_float(self, spacing, dim, edge_order);
}

// 决定是否使用累加缓冲区的内联函数
inline bool should_use_acc_buffer(at::TensorIterator& iter) {
  // 获取迭代器的维度数
  const auto ndim = iter.ndim();
  // 如果设备不是 CPU 或者输出张量数不为 1，则不使用累加缓冲区
  if (!iter.device().is_cpu() || iter.noutputs() != 1) {
    return false;
  }
  // 如果公共数据类型不是浮点类型，则不使用累加缓冲区
  if (!at::isReducedFloatingType(iter.common_dtype())) {
    return false;
  }
  // 如果维度小于 2，则不使用累加缓冲区
  if (ndim < 2) {
    return false;
  }
  // 检查输出张量的前两个维度步长，如果不为 0，则不使用累加缓冲区
  auto out_strides = iter.strides(0);
  for (const auto dim : c10::irange(0, 2)) {
      if (out_strides[dim] != 0) {
        return false;
      }
  }
  // 符合所有条件，使用累加缓冲区
  return true;
}

// 实现 sum_out 函数
TORCH_IMPL_FUNC(sum_out)
(const Tensor& self,
 OptionalIntArrayRef opt_dim,
 bool keepdim,
 optional<ScalarType> opt_dtype,
 const Tensor& result) {
  // 从输入和输出创建一个迭代器，该迭代器用于执行降维操作
  auto iter = meta::make_reduction_from_out_ty(self, result, opt_dim, keepdim, result.scalar_type());
  // 如果输入张量为空，则将输出张量置零
  if (iter.numel() == 0) {
    result.zero_();
  } else {
    // 如果应该使用累加缓冲区，则创建临时输出张量，并使用高精度计算
    if (should_use_acc_buffer(iter)) {
      auto tmp_output = at::empty(result.sizes(), result.options().dtype(kFloat));
      // 执行将输入张量转换为浮点型后的求和操作
      at::sum_outf(self.to(ScalarType::Float), opt_dim, keepdim, /*dtype=*/c10::nullopt, tmp_output);
      // 将临时输出复制到结果张量中
      result.copy_(tmp_output);
    } else {
      // 否则直接使用底层的求和函数进行计算
      sum_stub(iter.device_type(), iter);
    }
  }
}

// 对不带维度名称的输入张量执行求和操作
Tensor sum(const Tensor &self, std::optional<ScalarType> dtype) {
  return at::sum(self, IntArrayRef{}, false, dtype);
}

// 对带有维度名称的输入张量执行求和操作
Tensor sum(const Tensor& self, DimnameList dim, bool keepdim, std::optional<ScalarType> dtype) {
  return at::sum(self, dimnames_to_positions(self, dim), keepdim, dtype);
}

// 将求和结果输出到指定的结果张量中
Tensor& sum_out(const Tensor& self, DimnameList dim,
                bool keepdim, optional<ScalarType> opt_dtype, Tensor& result) {
  return at::sum_out(result, self, dimnames_to_positions(self, dim), keepdim, opt_dtype);
}

// 对输入张量执行带有NaN处理的求和操作，并将结果输出到指定的结果张量中
Tensor& nansum_out(const Tensor& self, at::OptionalIntArrayRef dim,
                       bool keepdim, optional<ScalarType> opt_dtype, Tensor& result) {
  // 如果输入张量在CPU上，且是复数类型，则不支持nansum操作
  if (self.device().is_cpu()) {
    TORCH_CHECK(!c10::isComplexType(self.scalar_type()), "nansum does not support complex inputs");
  }

  // 对于整数类型，直接使用普通的求和函数，因为整数类型不存在NaN
  if (c10::isIntegralType(self.scalar_type(), true)) {
    return at::sum_out(result, self, dim, keepdim, opt_dtype);
  }

  // 否则，根据输入张量的类型和参数创建一个迭代器，并执行带有NaN处理的求和操作
  ScalarType dtype = get_dtype_from_result(result, opt_dtype);
  auto iter = make_reduction("nansum", result, self, dim, keepdim, dtype);
  // 如果输入张量为空，则将结果张量置零
  if (iter.numel() == 0) {
    result = result.zero_();
  } else {
    // 否则调用底层的nansum函数进行计算
    nansum_stub(iter.device_type(), iter);
  }
  return result;
}

// 对输入张量执行带有NaN处理的求和操作
Tensor nansum(const Tensor& self, at::OptionalIntArrayRef dim, bool keepdim, std::optional<ScalarType> opt_dtype) {
  // 根据输入张量和参数创建求和结果的张量
  ScalarType dtype = get_dtype_from_self(self, opt_dtype, true);
  Tensor result = create_reduction_result(self, dim, keepdim, dtype);
  // 调用nansum_out函数执行带有NaN处理的求和，并将结果返回
  return at::native::nansum_out(self, dim, keepdim, dtype, result);
}

namespace {
// 模板函数：将累加结果设置到结果张量中
template<typename scalar_t, typename accscalar_t = at::acc_type<scalar_t, false>>
void inline set_result(Tensor& result, accscalar_t sum)
{
    if constexpr (std::is_integral_v<accscalar_t>) {
        // 如果 accscalar_t 是整数类型，将所有整数类型提升为 int64_t
        *result.data_ptr<int64_t>() = sum;
    } else {
        // 如果 accscalar_t 不是整数类型，直接赋值给对应类型的指针
        *result.data_ptr<scalar_t>() = sum;
    }
// } 末尾标记函数 trace_cpu 的结尾
// } 末尾标记匿名函数 AT_DISPATCH_ALL_TYPES_AND_COMPLEX
// NOTE: this could be implemented via diag and sum, but this has perf problems,
// see https://github.com/pytorch/pytorch/pull/47305,
// 定义了一个名为 trace_cpu 的函数，接受一个名为 self 的 Tensor 参数
Tensor trace_cpu(const Tensor& self) {
  Tensor result;
  // 返回 self 张量的 ScalarType，如果是非整数类型张量，则返回其 ScalarType
  // 在 self 是整数类型张量的情况下，返回 at::kLong，因为 promote_integers 被设置为 true
  ScalarType dtype = get_dtype_from_self(self, c10::nullopt, true);
  // 创建一个空的标量 Tensor，使用 self 张量的选项和指定的 dtype
  result = at::empty({}, self.options().dtype(dtype));
  // 使用宏展开，分派给所有类型和复数类型的元素进行操作
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX(self.scalar_type(), "trace", [&] {
    // 定义了 accscalar_t 类型为 scalar_t 的精度类型，不用做四舍五入
    using accscalar_t = at::acc_type<scalar_t, false>;
    // 初始化 sum 为 0
    accscalar_t sum = 0;
    // 获取 self 张量的常量数据指针，并转换为 scalar_t 类型的指针
    const auto* t_data = self.const_data_ptr<scalar_t>();

    int64_t t_stride_0, t_stride_1, t_diag_size;

    // 检查 self 张量的维度是否为 2，如果不是则抛出错误信息
    TORCH_CHECK(self.dim() == 2, "trace: expected a matrix, but got tensor with dim ", self.dim());

    // 获取 self 张量的步长值
    t_stride_0 = self.stride(0);
    t_stride_1 = self.stride(1);

    // 计算对角线元素的数量，取 self 张量第一维和第二维大小的最小值
    t_diag_size = std::min(self.size(0), self.size(1));
    // 遍历对角线元素，累加到 sum 中
    for (const auto i : c10::irange(t_diag_size)) {
      sum += t_data[i * (t_stride_0 + t_stride_1)];
    }
    // 将 sum 设置到 result 中
    set_result<scalar_t>(result, sum);

  });

  // 返回计算结果的 Tensor
  return result;
}

// 定义了一个静态函数 impl_func_prod，实现对 self 张量进行指定维度的乘积计算
static void impl_func_prod(
    const Tensor& self,
    IntArrayRef dims,
    bool keepdim,
    std::optional<ScalarType> dtype,
    const Tensor& result) {
  // 根据 self 和 result 张量的类型和维度创建元数据迭代器
  auto iter = meta::make_reduction_from_out_ty(self, result, dims, keepdim, result.scalar_type());
  // 如果迭代器中的元素数量为 0，则将 result 张量填充为 1
  if (iter.numel() == 0) {
    result.fill_(1);
  } else {
    // 否则调用 prod_stub 对迭代器进行乘积计算
    prod_stub(iter.device_type(), iter);
  }
}

// TORCH_IMPL_FUNC(prod_out) 定义了名为 prod_out 的函数，实现了对 self 张量指定维度的乘积计算
TORCH_IMPL_FUNC(prod_out)
(const Tensor& self,
 int64_t dim,
 bool keepdim,
 std::optional<ScalarType> dtype,
 const Tensor& result) {
  // 调用 impl_func_prod 对 self 张量进行指定维度的乘积计算
  impl_func_prod(self, dim, keepdim, dtype, result);
}

// 定义了一个函数 prod，计算 self 张量所有元素的乘积
Tensor prod(const Tensor &self, std::optional<ScalarType> opt_dtype) {
  // 获取 self 张量的 dtype
  auto dtype = get_dtype_from_self(self, opt_dtype, true);
  // 根据 self 张量的形状获取 reduction 的 shape
  auto shape = meta::get_reduction_shape(self, {}, false);
  // 创建一个空的 Tensor result，使用 self 张量的 dtype 和指定的 shape
  Tensor result = at::empty(shape, self.options().dtype(dtype));
  // 调用 impl_func_prod 对 self 张量所有元素进行乘积计算
  impl_func_prod(self, {}, false, dtype, result);
  // 返回计算结果的 Tensor
  return result;
}

// 定义了一个函数 prod，计算 self 张量在指定维度 dim 上的乘积
Tensor prod(const Tensor& self, Dimname dim, bool keepdim, std::optional<ScalarType> dtype) {
  // 调用 at::prod 函数，将 Dimname 转换为维度位置
  return at::prod(self, dimname_to_position(self, dim), keepdim, dtype);
}

// 定义了一个函数 prod_out，计算 self 张量在指定 Dimname 维度上的乘积，并将结果存储到 result 张量中
Tensor& prod_out(const Tensor& self, Dimname dim,
                 bool keepdim, optional<ScalarType> opt_dtype, Tensor& result) {
  // 调用 at::prod_out 函数，将 Dimname 转换为维度位置，计算乘积并存储到 result 中
  return at::prod_out(result, self, dimname_to_position(self, dim), keepdim, opt_dtype);
}

// TORCH_IMPL_FUNC(mean_out) 定义了名为 mean_out 的函数，实现了对 self 张量的均值计算
TORCH_IMPL_FUNC(mean_out)
(const Tensor& self,
 OptionalIntArrayRef opt_dim,
 bool keepdim,
 std::optional<ScalarType> opt_dtype,
 const Tensor& result) {
  // 获取 result 张量的 ScalarType
  ScalarType dtype = result.scalar_type();
  // 如果 self 张量在 CPU 设备上
  if (self.device().is_cpu()) {
    // 计算指定维度的元素乘积
    int64_t dim_prod = 1;
    // 如果未提供维度选项、维度选项为空或者输入张量没有维度（ndimension() == 0），则计算整个张量的元素数量作为维度乘积
    if (!opt_dim.has_value() || opt_dim.value().empty() || self.ndimension() == 0) {
      dim_prod = self.numel();
    } else {
      // 否则，使用给定的维度选项计算维度乘积
      auto dim = opt_dim.value();
      for (auto d : dim) {
        dim_prod *= self.size(d);
      }
    }
    // 获取可变引用的结果张量（result）
    auto& result_mut = const_cast<Tensor&>(result);
    // 对于精度要求，BF16/FP16 的均值计算应该通过以下方法进行：
    //  cast_fp32 -> sum -> div -> cast_bf16_or_fp16
    //
    // 这种方法是必要的，因为如果我们选择与 FP32 相同的方法来处理 BF16/FP16，那么会导致以下代码流程 -
    // cast_fp32 -> sum -> cast_bf16 -> cast_fp32 -> div -> cast_bf16，
    // 这样不会产生准确的结果。
    bool is_half_type = (dtype == kHalf || dtype == kBFloat16);
    auto sum_out_dtype = is_half_type ? ScalarType::Float : dtype;
    // 如果 dtype 是 FP16 或 BF16，则将结果张量 result_mut 先转换为 sum_out_dtype 类型
    result_mut = is_half_type ? result_mut.to(sum_out_dtype) : result_mut;
    // 如果 dtype 是 FP16 或 BF16，在求和和除法之后，将结果张量 result_mut 转换回原始的 dtype 类型
    result_mut = is_half_type ? result_mut.to(dtype) : result_mut;
  } else {
    // 如果设备不是 CPU
    // 根据输出结果 result 创建一个迭代器 iter
    auto iter = at::meta::make_reduction_from_out_ty(
        self, result, opt_dim, keepdim, dtype);
    // 如果迭代器中元素数量为 0，则将结果张量 result 填充为 NaN
    if (iter.numel() == 0) {
      result.fill_(std::numeric_limits<double>::quiet_NaN());
    } else {
      // 否则，调用均值计算的函数 mean_stub
      mean_stub(iter.device_type(), iter);
    }
  }
}

// 结束一个代码块，该代码块包含了 logsumexp_out_impl 函数的定义

Tensor mean(const Tensor &self, optional<ScalarType> dtype) {
  // 调用 at::mean 函数计算张量 self 的平均值，不指定维度，默认使用整个张量
  return at::mean(self, IntArrayRef{}, false, dtype);
}

Tensor mean(const Tensor& self, DimnameList dim, bool keepdim, optional<ScalarType> dtype) {
  // 调用 at::mean 函数计算张量 self 沿指定维度 dim 的平均值，根据 DimnameList 转换为维度索引
  return at::mean(self, dimnames_to_positions(self, dim), keepdim, dtype);
}

Tensor& mean_out(const Tensor& self, DimnameList dim,
                 bool keepdim, std::optional<ScalarType> opt_dtype, Tensor& result) {
  // 调用 at::mean_out 函数计算张量 self 沿指定维度 dim 的平均值，并将结果存入 result 张量中
  return at::mean_out(result, self, dimnames_to_positions(self, dim), keepdim, opt_dtype);
}

// TODO(@heitorschueroff) implement custom kernels for nanmean
Tensor& nanmean_out(
    const Tensor& self,
    at::OptionalIntArrayRef dim,
    bool keepdim,
    std::optional<ScalarType> opt_dtype,
    Tensor& result) {
  // 检查输入张量 self 是否为浮点数或复数类型
  TORCH_CHECK(
      self.is_floating_point() || self.is_complex(),
      "nanmean(): expected input to have floating point or complex dtype but got ",
      self.scalar_type());
  // 计算不是 NaN 的元素个数，并在结果张量 result 中执行 nansum 后再除以这个个数
  const auto factor = at::native::isnan(self).logical_not_().sum(dim, keepdim);
  at::native::nansum_out(self, dim, keepdim, opt_dtype, result).div_(factor);
  return result;
}

Tensor nanmean(
    const Tensor& self,
    at::OptionalIntArrayRef dim,
    bool keepdim,
    optional<ScalarType> opt_dtype) {
  // 检查输入张量 self 是否为浮点数或复数类型
  TORCH_CHECK(
      self.is_floating_point() || self.is_complex(),
      "nanmean(): expected input to have floating point or complex dtype but got ",
      self.scalar_type());
  // 计算不是 NaN 的元素个数，并在返回值中执行 nansum 后再除以这个个数
  const auto factor =
      at::native::isnan(self.detach()).logical_not_().sum(dim, keepdim);
  return at::nansum(self, dim, keepdim, opt_dtype).div(factor);
}

static Tensor& logsumexp_out_impl(Tensor& result, const Tensor& self, IntArrayRef dims, bool keepdim) {
  // 如果输入张量 self 不为空，则执行 logsumexp 运算
  if (self.numel() != 0) {
    // 计算指定维度 dims 上的最大值，并在保持维度的情况下得到最大值张量
    auto maxes = at::amax(self, dims, true);
    auto maxes_squeezed = (keepdim ? maxes : at::squeeze(maxes, dims));
    // 将最大值张量中的无穷值位置设为 0
    maxes_squeezed.masked_fill_(maxes_squeezed.abs() == INFINITY, 0);
    // 执行 exp 函数后减去最大值，并对结果求和，然后再取对数，加上最大值张量的值
    at::sum_out(result, (self - maxes).exp_(), dims, keepdim);
    result.log_().add_(maxes_squeezed);
  } else {
    // 输入张量为空时，执行 exp 函数后取对数操作
    at::sum_out(result, at::exp(self), dims, keepdim);
    result.log_();
  }
  return result;
}

Tensor& logsumexp_out(const Tensor& self, IntArrayRef dims, bool keepdim, Tensor& result) {
  // 检查结果张量 result 的类型是否为浮点数类型
  TORCH_CHECK(at::isFloatingType(result.scalar_type()),
              "logsumexp(): Expected floating point type for result tensor, but got: ",
              result.scalar_type());
  {
    NoNamesGuard guard;
    // 如果输入张量 self 的类型为整数类型（包括布尔类型），则提升为默认的浮点数类型
    if (at::isIntegralType(self.scalar_type(), /*includeBool=*/true)) {
      auto default_dtype = at::typeMetaToScalarType(c10::get_default_dtype());
      logsumexp_out_impl(result, self.to(default_dtype), dims, keepdim);
    } else {
      logsumexp_out_impl(result, self, dims, keepdim);
    }
  }
  // 在执行张量的减少操作后，传播命名信息到结果张量
  namedinference::propagate_names_for_reduction(result, self, dims, keepdim);
  return result;
}
// 计算给定张量在指定维度上的 logsumexp 操作，返回结果张量
Tensor logsumexp(const Tensor& self, IntArrayRef dims, bool keepdim) {
  // 确定结果张量的选项，即数据类型，默认情况下是浮点数类型
  TensorOptions result_options;
  if (at::isIntegralType(self.scalar_type(), /*includeBool=*/true)) {
    auto default_dtype = at::typeMetaToScalarType(c10::get_default_dtype());
    result_options = self.options().dtype(default_dtype);
  } else {
    result_options = self.options();
  }
  // 创建一个空的结果张量
  auto result = at::empty({0}, result_options);
  // 调用 logsumexp_outf 函数计算 logsumexp 操作，并将结果存入预先创建的结果张量中
  return at::logsumexp_outf(self, dims, keepdim, result);
}

// 计算给定张量在指定命名维度上的 logsumexp 操作，返回结果张量
Tensor logsumexp(const Tensor& self, DimnameList dims, bool keepdim) {
  // 将命名维度转换为位置维度后调用 logsumexp 函数进行计算
  return at::logsumexp(self, dimnames_to_positions(self, dims), keepdim);
}

// 计算给定张量在指定命名维度上的 logsumexp 操作，结果存入指定的输出张量中
Tensor& logsumexp_out(const Tensor& self, DimnameList dims, bool keepdim, Tensor& result) {
  return at::logsumexp_out(result, self, dimnames_to_positions(self, dims), keepdim);
}

// 特殊情况下的 logsumexp 操作，与 logsumexp 等效
Tensor special_logsumexp(const Tensor& self, IntArrayRef dims, bool keepdim) {
  return self.logsumexp(dims, keepdim);
}

// 特殊情况下的 logsumexp_out 操作，与 logsumexp_out 等效
Tensor& special_logsumexp_out(const Tensor& self, IntArrayRef dims, bool keepdim, Tensor& result) {
  return at::logsumexp_out(result, self, dims, keepdim);
}

// 实现计算给定张量的范数操作的具体函数，将结果存入指定的输出张量中
static void impl_func_norm(
    const Tensor& self,
    const OptionalScalarRef& opt_p,
    IntArrayRef dim,
    bool keepdim,
    optional<ScalarType> opt_dtype,
    const Tensor& result) {
  // 选择范数值 p 的默认值，如果未提供则默认为 2.0
  auto p = opt_p.has_value() ? opt_p.get() : Scalar(2.0).to<double>();
  // 调用 linalg_vector_norm_out 函数进行范数计算，并将结果存入输出张量中
  at::linalg_vector_norm_out(const_cast<Tensor&>(result), self, p, dim, keepdim, opt_dtype);
}

// 使用实现函数 impl_func_norm 计算给定张量的范数操作，并将结果存入指定的输出张量中
TORCH_IMPL_FUNC(norm_out)
(const Tensor& self,
 const OptionalScalarRef p,
 IntArrayRef dim,
 bool keepdim,
 const Tensor& result) {
  impl_func_norm(self, p, dim, keepdim, c10::nullopt, result);
}

// 使用实现函数 impl_func_norm 计算给定张量的范数操作，并将结果存入指定的输出张量中，指定输出数据类型
TORCH_IMPL_FUNC(norm_dtype_out)
(const Tensor& self,
 const OptionalScalarRef p,
 IntArrayRef dim,
 bool keepdim,
 ScalarType dtype,
 const Tensor& result) {
  impl_func_norm(self, p, dim, keepdim, dtype, result);
}

// 计算稀疏张量在指定维度上的范数操作，返回结果张量
Tensor sparse_norm(
    const Tensor& self,
    const optional<Scalar>& p,
    IntArrayRef dim,
    bool keepdim) {
  return at::native_norm(self, p, dim, keepdim, c10::nullopt);
}

// 计算稀疏张量在指定维度上的范数操作，结果存入指定的输出张量中，指定输出数据类型
Tensor sparse_dtype_norm(
    const Tensor& self,
    const optional<Scalar>& p,
    IntArrayRef dim,
    bool keepdim,
    ScalarType dtype) {
  return at::native_norm(self, p, dim, keepdim, dtype);
}

// 计算给定张量的范数操作，指定输出数据类型，默认在所有维度上进行操作
Tensor norm(const Tensor& self, const optional<Scalar>& p, ScalarType dtype) {
  return at::norm(self, p, IntArrayRef{}, false, dtype);
}

// 计算给定张量的范数操作，指定范数值 p，默认在所有维度上进行操作
Tensor norm(const Tensor& self, const Scalar& p) {
  return at::norm(self, p, IntArrayRef{}, false);
}

// 获取张量迭代器，用于 allany 函数的操作，指定操作维度和是否保持维度信息
inline TensorIterator get_allany_iter(
    const Tensor& self,
    const Tensor& result,
    OptionalIntArrayRef dims,
    bool keepdim) {
  if (self.is_cuda()) {
    // 因为 CUDA 支持动态类型转换，使用这个重载版本
    // TODO: 确定这里的实现逻辑，以确保 CUDA 上的正确性
    // 返回一个根据输入不进行类型转换的 `make_reduction` 函数，即使用当前对象的标量类型。
    // 如果需要额外的操作来转换输入为 kBool 类型，则使用下面的重载版本。
    return meta::make_reduction(self, result, dims, keepdim, self.scalar_type());
  }
  // 使用输出结果的标量类型，返回一个带有输出类型的 `make_reduction_from_out_ty` 函数。
  return meta::make_reduction_from_out_ty(
      self, result, dims, keepdim, result.scalar_type());
}

template <int identity, typename Stub>
// 定义模板函数 allany_impl，用于执行 all 和 any 操作的实现
inline void allany_impl(
    const Tensor& self,                    // 输入张量 self
    const Tensor& result,                  // 输出张量 result
    OptionalIntArrayRef dims,              // 可选的维度数组引用 dims
    bool keepdim,                          // 是否保持维度的布尔值
    Stub& stub) {                          // 存根对象的引用
  if (self.numel() == 0) {
    result.fill_(identity);                // 如果输入张量为空，则将输出张量填充为给定的 identity 值
  } else if (self.numel() == 1) {
    result.copy_(self.view_as(result).to(at::kBool));  // 如果输入张量只有一个元素，将其复制到输出张量并转换为布尔类型
  } else {
    auto iter = get_allany_iter(self, result, dims, keepdim);  // 否则，获取 allany 迭代器对象
    stub(iter.device_type(), iter);        // 调用存根对象的操作符函数，传递设备类型和迭代器对象
  }
}

TORCH_IMPL_FUNC(all_out)
(const Tensor& self, int64_t dim, bool keepdim, const Tensor& result) {
  allany_impl<1>(self, result, dim, keepdim, and_stub);  // 实现 all 操作的函数，调用 allany_impl 模板函数
}

TORCH_IMPL_FUNC(all_dims_out)
(const Tensor& self, OptionalIntArrayRef dim, bool keepdim, const Tensor& result) {
  allany_impl<1>(self, result, dim, keepdim, and_stub);  // 实现带有指定维度的 all 操作的函数，调用 allany_impl 模板函数
}

TORCH_IMPL_FUNC(all_all_out)(const Tensor& self, const Tensor& result) {
  allany_impl<1>(self, result, {}, false, and_stub);     // 实现对所有维度执行 all 操作的函数，调用 allany_impl 模板函数
}

TORCH_IMPL_FUNC(any_out)
(const Tensor& self, int64_t dim, bool keepdim, const Tensor& result) {
  allany_impl<0>(self, result, dim, keepdim, or_stub);   // 实现 any 操作的函数，调用 allany_impl 模板函数
}

TORCH_IMPL_FUNC(any_dims_out)
(const Tensor& self, OptionalIntArrayRef dim, bool keepdim, const Tensor& result) {
  allany_impl<0>(self, result, dim, keepdim, or_stub);   // 实现带有指定维度的 any 操作的函数，调用 allany_impl 模板函数
}

TORCH_IMPL_FUNC(any_all_out)(const Tensor& self, const Tensor& result) {
  allany_impl<0>(self, result, {}, false, or_stub);      // 实现对所有维度执行 any 操作的函数，调用 allany_impl 模板函数
}

template <bool is_all>
Tensor allany_dims_default(const Tensor &self, OptionalIntArrayRef dim, bool keepdim) {
  // 默认实现，基于全局减少或单维减少操作
  if (!dim) {
    Tensor out;
    if constexpr (is_all) {
      out = self.all();                    // 如果维度未指定且为 all 操作，则对整个张量执行 all 操作
    } else {
      out = self.any();                    // 如果维度未指定且为 any 操作，则对整个张量执行 any 操作
    }

    if (keepdim) {
      DimVector out_shape(self.dim(), 1);  // 如果需要保持维度，则将输出张量扩展为与输入张量相同的形状
      return out.expand(out_shape);
    }
    return out;
  }

  if (dim->size() == 0) {
    if (self.scalar_type() == kByte) {
      // 转换为 1 或 0 的掩码
      auto out = at::empty_like(self);     // 如果维度为空且输入张量类型为 kByte，则创建一个与输入张量相同形状的空张量
      return at::ne_outf(self, 0, out);    // 使用不等于运算符创建掩码
    } else {
      return at::_to_copy(self, kBool);    // 否则，将输入张量转换为布尔类型
    }
  }

  Tensor out = self;
  for (auto d : *dim) {
    if constexpr (is_all) {
      out = out.all(d, /*keepdim=*/true);  // 对指定维度执行 all 操作，保持维度
    } else {
      out = out.any(d, /*keepdim=*/true);  // 对指定维度执行 any 操作，保持维度
    }
  }
  return keepdim ? out : out.squeeze(*dim); // 如果需要保持维度，则返回输出张量；否则，挤压指定的维度
}

Tensor all_dims_default(const Tensor &self, OptionalIntArrayRef dim, bool keepdim) {
  return allany_dims_default<true>(self, dim, keepdim);   // 默认执行 all 操作的函数
}

Tensor any_dims_default(const Tensor &self, OptionalIntArrayRef dim, bool keepdim) {
  return allany_dims_default<false>(self, dim, keepdim);  // 默认执行 any 操作的函数
}

Tensor& all_dims_out_default(
    const Tensor &self, OptionalIntArrayRef dim, bool keepdim, Tensor &result) {
  TORCH_CHECK(self.device() == result.device(), "all.dims: output must be on the same device as input");
  auto tmp = all_dims_default(self, dim, keepdim);
  at::native::resize_output(result, tmp.sizes());         // 调整输出张量的大小以匹配临时张量的大小
  return result.copy_(tmp);                               // 将临时张量的内容复制到输出张量
}

Tensor& any_dims_out_default(
    // 检查输出张量和输入张量是否在相同的设备上，否则抛出错误信息
    TORCH_CHECK(self.device() == result.device(), "any.dims: output must be on the same device as input");

    // 调用 any_dims_default 函数计算任意维度上的结果
    auto tmp = any_dims_default(self, dim, keepdim);

    // 调用 resize_output 函数，将结果张量 result 调整为和 tmp 相同的尺寸
    at::native::resize_output(result, tmp.sizes());

    // 将 tmp 张量的数据复制到结果张量 result 中，并返回 result
    return result.copy_(tmp);
}

TORCH_IMPL_FUNC(amin_out) (const Tensor& self, IntArrayRef dim, bool keepdim, const Tensor& result) {
  auto iter =
      meta::make_reduction(self, result, dim, keepdim, self.scalar_type());
  // 创建一个迭代器，用于执行对输入张量 self 在指定维度 dim 上的归约操作，并将结果存入 result 张量中
  if (iter.numel() != 0) {
    // 如果迭代器非空（即张量 self 在指定维度上有元素）
    min_values_stub(iter.device_type(), iter);
    // 调用 min_values_stub 函数，执行对迭代器 iter 所指定的设备类型上的最小值计算操作
  }
}

TORCH_IMPL_FUNC(amax_out) (const Tensor& self, IntArrayRef dim, bool keepdim, const Tensor& result) {
  auto iter =
      meta::make_reduction(self, result, dim, keepdim, self.scalar_type());
  // 创建一个迭代器，用于执行对输入张量 self 在指定维度 dim 上的归约操作，并将结果存入 result 张量中
  if (iter.numel() != 0) {
    // 如果迭代器非空（即张量 self 在指定维度上有元素）
    max_values_stub(iter.device_type(), iter);
    // 调用 max_values_stub 函数，执行对迭代器 iter 所指定的设备类型上的最大值计算操作
  }
}

template <class Stub>
void argmax_argmin_impl(
    const Tensor& self,
    std::optional<int64_t> dim,
    bool keepdim,
    const Tensor& result,
    Stub& stub) {
  c10::MaybeOwned<Tensor> in;
  DimVector dims;
  int64_t _dim = 0;

  if (dim.has_value()) {
    _dim = maybe_wrap_dim(dim.value(), self.dim());
    auto sizes = self.sizes();

    if (sizes[_dim] == 1) {
      result.fill_(0);
      return;
    }

    dims = IntArrayRef(_dim);
    in = c10::MaybeOwned<Tensor>::borrowed(self);
  } else {
    in = c10::MaybeOwned<Tensor>::owned(self.reshape({-1}));
    keepdim = false;
  }

  auto iter =
      meta::make_reduction(*in, result, dims, keepdim, self.scalar_type());
  // 创建一个迭代器，用于执行对输入张量 self 在指定维度 dim 上的归约操作，并将结果存入 result 张量中
  if (iter.numel() != 0) {
    // 如果迭代器非空（即张量 self 在指定维度上有元素）
    stub(iter.device_type(), iter);
    // 调用传入的 stub 函数，执行对迭代器 iter 所指定的设备类型上的操作（根据具体情况是 argmax 或 argmin）
  }
}

TORCH_IMPL_FUNC(argmax_out)
(const Tensor& self,
 std::optional<int64_t> dim,
 bool keepdim,
 const Tensor& result) {
  // 调用 argmax_argmin_impl 函数，执行对输入张量 self 进行 argmax 操作
  argmax_argmin_impl(self, dim, keepdim, result, argmax_stub);
}

TORCH_IMPL_FUNC(argmin_out)
(const Tensor& self,
 std::optional<int64_t> dim,
 bool keepdim,
 const Tensor& result) {
  // 调用 argmax_argmin_impl 函数，执行对输入张量 self 进行 argmin 操作
  argmax_argmin_impl(self, dim, keepdim, result, argmin_stub);
}

static double std_var_all_cpu(const Tensor& self, double correction, bool take_sqrt) {
  const auto dtype = self.scalar_type();
  // 检查张量的数据类型是否是 kDouble 或 kFloat，如果不是抛出错误信息
  TORCH_CHECK(dtype == kDouble || dtype == kFloat,
              "std_var_all: Unsupported dtype ", dtype);

  // 计算张量 self 的均值并转换为 double 类型
  auto mean = self.mean().item<double>();
  // 根据 TensorIteratorConfig 配置创建迭代器 iter，用于处理输入张量 self
  auto iter = TensorIteratorConfig()
      .add_const_input(self)
      .build();

  // 定义一个 lambda 函数 reduction，用于在指定的范围内计算方差或标准差
  auto reduction = [&](int64_t begin, int64_t end, double thread_sum) {
    // 根据张量的数据类型调度不同的操作
    AT_DISPATCH_FLOATING_TYPES(iter.common_dtype(), "std_var_all_cpu", [&] {
      // 使用 iter.serial_for_each 方法对迭代器 iter 中的每个元素执行特定操作
      iter.serial_for_each([&] (char** data, const int64_t* strides, int64_t size0, int64_t size1) {
        // 获取均值的本地副本
        const double local_mean = mean;
        // 获取内部和外部步幅
        const int64_t inner_stride = strides[0];
        const int64_t outer_stride = strides[1];

        // 初始化本地和为 0.0
        double local_sum = 0.0;
        // 遍历处理每个元素
        for (const auto i : c10::irange(size1)) {
          const char* row_ptr = data[0] + outer_stride * i;
          for (const auto j : c10::irange(size0)) {
            const auto ptr = reinterpret_cast<const scalar_t*>(row_ptr + inner_stride * j);
            auto dx = (static_cast<double>(*ptr) - local_mean);
            local_sum += dx * dx;
          }
        }
        // 线程和增加本地和
        thread_sum += local_sum;
      }, {begin, end});
    });
    // 返回线程内部计算的总和
    return thread_sum;
  };

  // 计算 ((x - mean)**2).sum()
  // 使用并行化方法计算总和，使用 std::plus<> 进行求和操作
  const double sum_dx2 = at::parallel_reduce(
      0, iter.numel(), at::internal::GRAIN_SIZE, 0.0, reduction, std::plus<>{});

  // 计算方差 var，忽略除以零的浮点数异常
  const auto var = [&] () __ubsan_ignore_float_divide_by_zero__ {
    return sum_dx2 / std::max(0.0, self.numel() - correction);
  }();
  
  // 根据 take_sqrt 变量决定是否对 var 进行平方根运算
  const auto result = take_sqrt ? std::sqrt(var) : var;

  // 如果 dtype 是 kFloat 类型，则将 result 转换为 float 类型
  if (dtype == kFloat) {
    // 如果结果超出 float 类型的范围，转换为无穷大
    // 这里的转换可以避免后续的 checked_convert 失败
    return static_cast<float>(result);
  }
  
  // 返回计算的结果
  return result;
} // 结束匿名命名空间

namespace { // 定义一个匿名命名空间，用于限定warn_invalid_degrees_of_freedom函数的作用域

  // 在给定函数名、迭代器和修正值的情况下，警告自由度无效
  inline void warn_invalid_degrees_of_freedom(const char* fname, const TensorIterator& iter, double correction) {
    // 计算每个元素上的归约比率是否小于等于0
    int64_t reducing_over_num_elements = iter.num_output_elements() == 0 ? 0 : iter.numel() / iter.num_output_elements();
    if (reducing_over_num_elements - correction <= 0) {
      // 如果自由度小于等于0，则发出警告信息
      TORCH_WARN(fname, "(): degrees of freedom is <= 0. Correction should be strictly less than the reduction factor (input numel divided by output numel).");
    }
  }
} // 结束匿名命名空间

static Tensor& std_var_out(
    const char* fname, Tensor& result, const Tensor& self,
    at::OptionalIntArrayRef dim, const std::optional<Scalar>& correction_opt,
    bool keepdim, bool take_sqrt) {
  // 检查张量所在设备是否为CPU或CUDA
  TORCH_CHECK(self.device().is_cpu() || self.device().is_cuda(),
              "std and var only supports tensors on a CPU or CUDA device, but got: ",
              self.device().type());
  // 检查张量是否采用连续布局
  TORCH_CHECK(self.layout() == Layout::Strided,
              "std and var only supports strided layout, got: ", self.layout());
  // 检查张量是否为浮点类型或复数类型
  TORCH_CHECK(at::isFloatingType(self.scalar_type()) || at::isComplexType(self.scalar_type()),
              "std and var only support floating point and complex dtypes");

  if (at::isComplexType(self.scalar_type())) {
    // 对于复数类型，分别计算实部和虚部的方差，然后相加得到总方差
    ScalarType dtype = c10::toRealValueType(get_dtype_from_result(result, {}));
    // 提取张量的实部
    Tensor real_in = at::real(self);
    // 创建与实部相同尺寸的空张量，用于存储实部的方差
    Tensor real_out = at::empty({0}, self.options().dtype(dtype));
    // 递归调用std_var_out处理实部
    std_var_out(
        fname,
        real_out,
        real_in,
        dim,
        correction_opt,
        keepdim,
        /*take_sqrt=*/false);

    // 提取张量的虚部
    Tensor imag_in = at::imag(self);
    // 创建与虚部相同尺寸的空张量，用于存储虚部的方差
    Tensor imag_out = at::empty({0}, self.options().dtype(dtype));
    // 递归调用std_var_out处理虚部
    std_var_out(
        fname,
        imag_out,
        imag_in,
        dim,
        correction_opt,
        keepdim,
        /*take_sqrt=*/false);

    // 将实部和虚部的方差相加得到总方差
    at::add_out(result, real_out, imag_out);
    // 如果指定需要进行平方根处理，则对结果应用平方根
    if (take_sqrt) {
      at::sqrt_out(result, result);
    }
    return result;
  }

  // 对于浮点类型的计算
  const auto correction = correction_opt.value_or(1).toDouble();
  ScalarType dtype = get_dtype_from_result(result, {});
  // 创建一个张量迭代器进行归约计算
  auto iter = make_reduction(fname, result, self, dim, keepdim, dtype);
  // 检查是否可以将输入张量类型转换为期望输出类型
  TORCH_CHECK(at::canCast(self.scalar_type(), result.scalar_type()),
              "result type ", self.scalar_type(), " can't be cast to the "
              "desired output type ", result.scalar_type());
  // 警告自由度无效
  warn_invalid_degrees_of_freedom(fname, iter, correction);

  if (iter.numel() == 0) {
    // 如果张量元素个数为0，则结果填充为NaN
    result.fill_(std::numeric_limits<double>::quiet_NaN());
    return result;
  } else if (
      result.numel() == 1 && iter.device_type() == kCPU &&
      iter.common_dtype() != kBFloat16 && iter.common_dtype() != kHalf) {
    // 注意：当尝试移植到ATen时，CPU性能显著退化，因此全局归约有一个自定义实现。
    // 如果是单个元素的简单归约操作，且迭代器的设备类型为CPU，并且公共数据类型不是BFloat16或Half
    // 这里可能是对性能问题的注释和说明
  // 如果条件满足，则使用特定函数计算标准差和方差，并填充结果张量
  result.fill_(std_var_all_cpu(self, correction, take_sqrt));
} else {
  // 如果条件不满足，则调用特定的函数处理标准差和方差计算，该函数根据设备类型选择不同的实现
  std_var_stub(iter.device_type(), iter, correction, take_sqrt);
}
// 返回计算得到的结果张量
return result;
}

static std::tuple<Tensor&, Tensor&> std_var_mean_out(
    const char* fname, Tensor& result1, Tensor& result2, const Tensor& self,
    at::OptionalIntArrayRef dim, const std::optional<Scalar>& correction_opt,
    bool keepdim, bool take_sqrt) {
  // 断言结果张量已定义且有效
  AT_ASSERT(result1.defined() && result2.defined());
  // 检查张量是否在 CPU 或 CUDA 设备上，否则抛出错误
  TORCH_CHECK(self.device().is_cpu() || self.is_cuda(),
              fname, " only supports tensors on a CPU or CUDA device, got: ",
              self.device().type());
  // 检查张量布局是否为步进布局，否则抛出错误
  TORCH_CHECK(self.layout() == Layout::Strided,
              fname, " only supports strided layout, got: ", self.layout());
  // 检查张量是否为浮点类型或复数类型，否则抛出错误
  TORCH_CHECK(at::isFloatingType(self.scalar_type()) || at::isComplexType(self.scalar_type()),
              fname, " only support floating point and complex dtypes");
  // 检查结果1张量是否为实数类型且与结果2张量精度匹配，否则抛出错误
  TORCH_CHECK(result1.scalar_type() == c10::toRealValueType(result2.scalar_type()),
              fname, " expected result1 to be real and match the precision of result2. Got ",
              result1.scalar_type(), " and ", result2.scalar_type(), ".");

  if (at::isComplexType(self.scalar_type())) {
    // 对于复数，分别计算实部和虚部的方差和均值
    // 方差 = var_real + var_imag
    // 均值 = mean_real + j * mean_imag
    ScalarType dtype = c10::toRealValueType(get_dtype_from_result(result1, {}));
    Tensor real_in = at::real(self);
    // 创建空张量以存储实部的方差和均值
    Tensor real_out_var = at::empty({0}, self.options().dtype(dtype));
    Tensor real_out_mean = at::empty({0}, self.options().dtype(dtype));
    // 调用递归函数计算实部的方差和均值
    std_var_mean_out(
        fname,
        real_out_var,
        real_out_mean,
        real_in,
        dim,
        correction_opt,
        keepdim,
        /*take_sqrt=*/false);

    Tensor imag_in = at::imag(self);
    // 创建空张量以存储虚部的方差和均值
    Tensor imag_out_var = at::empty({0}, self.options().dtype(dtype));
    Tensor imag_out_mean = at::empty({0}, self.options().dtype(dtype));
    // 调用递归函数计算虚部的方差和均值
    std_var_mean_out(
        fname,
        imag_out_var,
        imag_out_mean,
        imag_in,
        dim,
        correction_opt,
        keepdim,
        /*take_sqrt=*/false);

    // 将实部和虚部的方差相加到结果1张量
    at::add_out(result1, real_out_var, imag_out_var);
    // 如果指定要进行平方根操作，则在结果1张量上应用平方根函数
    if (take_sqrt) {
      at::sqrt_out(result1, result1);
    }
    // 组合实部和虚部的均值到结果2张量
    at::complex_out(result2, real_out_mean, imag_out_mean);
    return std::tuple<Tensor&, Tensor&>(result1, result2);
  }

  // 对于浮点类型的计算
  const auto correction = correction_opt.value_or(1).toDouble();
  ScalarType dtype = get_dtype_from_result(result1, {});
  // 创建迭代器以进行降维操作
  auto iter =
      make_reduction(fname, result1, result2, self, dim, keepdim, dtype);
  // 检查自由度是否有效，否则发出警告
  warn_invalid_degrees_of_freedom(fname, iter, correction);

  if (iter.numel() == 0) {
    // 如果迭代器为空，填充结果1和结果2张量为 NaN
    result1.fill_(std::numeric_limits<double>::quiet_NaN());
    result2.fill_(std::numeric_limits<double>::quiet_NaN());
  } else {
    // 否则，调用标准方差函数处理迭代器数据
    std_var_stub(iter.device_type(), iter, correction, take_sqrt);
  }
  return std::tuple<Tensor&, Tensor&>(result1, result2);
}
    // 计算输入张量 self 沿指定维度 dim 的方差和均值。
    const Tensor& self, at::OptionalIntArrayRef dim, bool unbiased, bool keepdim) {
        // 调用 at::var_mean 函数计算方差和均值
        return at::var_mean(
            self, /*dim=*/at::OptionalIntArrayRef(dim),
            // 根据 unbiased 参数确定是否进行修正
            /*correction=*/c10::make_optional<Scalar>(unbiased ? 1 : 0),
            // keepdim 参数指定是否保持输出张量的维度
            keepdim);
    }
}

// 计算张量 self 沿指定维度 dim 的标准差和均值
std::tuple<Tensor, Tensor> std_mean(
    const Tensor& self, at::OptionalIntArrayRef dim, bool unbiased, bool keepdim) {
  return at::std_mean(
      self, /*dim=*/at::OptionalIntArrayRef(dim),
      /*correction=*/c10::make_optional<Scalar>(unbiased ? 1 : 0),
      keepdim);
}

// 计算张量 self 的标准差和均值，不指定维度
std::tuple<Tensor, Tensor> std_mean(const Tensor& self, bool unbiased) {
  return at::std_mean(
      self, /*dim=*/c10::nullopt,
      /*correction=*/c10::make_optional<Scalar>(unbiased ? 1 : 0));
}

// 计算张量 self 的方差和均值，不指定维度
std::tuple<Tensor, Tensor> var_mean(const Tensor& self, bool unbiased) {
  return at::var_mean(
      self, /*dim=*/c10::nullopt,
      /*correction=*/c10::make_optional<Scalar>(unbiased ? 1 : 0));
}

// 计算张量 self 沿指定维度 dim 的方差和均值，并输出到 result1 和 result2
std::tuple<Tensor&, Tensor&> var_mean_out(
    Tensor& result1, Tensor& result2, const Tensor& self, IntArrayRef dim,
    int64_t correction, bool keepdim) {
  return std_var_mean_out(
      "var_mean", result1, result2, self, dim, correction, keepdim, false);
}

// 将张量选项 opts 转换为对应的数值类型选项
static TensorOptions options_to_value_type(TensorOptions opts) {
  auto scalar_type = typeMetaToScalarType(opts.dtype());
  return opts.dtype(c10::toRealValueType(scalar_type));
}

// 计算张量 self 沿指定维度 dim 的方差和均值，支持指定修正值 correction
std::tuple<Tensor, Tensor> var_mean(
    const Tensor& self, at::OptionalIntArrayRef dim,
    const std::optional<Scalar>& correction, bool keepdim) {
  Tensor result1 = at::empty({0}, options_to_value_type(self.options()));
  Tensor result2 = at::empty({0}, self.options());
  return std_var_mean_out(
      "var_mean", result1, result2, self, dim, correction, keepdim, false);
}

// 计算张量 self 沿指定维度 dim 的标准差和均值，支持指定修正值 correction
std::tuple<Tensor, Tensor> std_mean(
    const Tensor& self, at::OptionalIntArrayRef dim,
    const std::optional<Scalar>& correction, bool keepdim) {
  Tensor result1 = at::empty({0}, options_to_value_type(self.options()));
  Tensor result2 = at::empty({0}, self.options());
  return std_var_mean_out(
      "std_mean", result1, result2, self, dim, correction, keepdim, true);
}

// 计算张量 self 的方差，不指定维度
Tensor var(const Tensor& self, bool unbiased) {
  return at::var(
      self, /*dim=*/c10::nullopt,
      /*correction=*/c10::make_optional<Scalar>(unbiased ? 1 : 0));
}

// 计算张量 self 沿指定维度 dim 的方差，支持指定修正值 correction
Tensor var(const Tensor& self, at::OptionalIntArrayRef dim, bool unbiased, bool keepdim) {
  return at::var(
      self, /*dim=*/at::OptionalIntArrayRef(dim),
      /*correction=*/c10::make_optional<Scalar>(unbiased ? 1 : 0),
      keepdim);
}

// 计算张量 self 的标准差，不指定维度
Tensor std(const Tensor& self, bool unbiased) {
  return at::std(
      self, /*dim=*/c10::nullopt, /*correction=*/c10::make_optional<Scalar>(unbiased ? 1 : 0));
}

// 计算张量 self 沿指定维度 dim 的标准差，支持指定修正值 correction
Tensor std(const Tensor& self, at::OptionalIntArrayRef dim, bool unbiased, bool keepdim) {
  return at::std(self, dim,
                 /*correction=*/c10::make_optional<Scalar>(unbiased ? 1 : 0), keepdim);
}
Tensor& std_out(const Tensor& self, at::OptionalIntArrayRef opt_dim, bool unbiased, bool keepdim, Tensor& result) {
    // 调用at命名空间中的std_out函数，计算标准差，并将结果存储到result中
    return at::std_out(result, self, opt_dim,
                       /*correction=*/c10::make_optional<Scalar>(unbiased ? 1 : 0), keepdim);
}

Tensor std(const Tensor& self, at::OptionalIntArrayRef dim,
           const std::optional<Scalar>& correction, bool keepdim) {
    // 创建一个空的张量result，用于存储标准差计算的结果
    Tensor result = at::empty({0}, options_to_value_type(self.options()));
    // 调用std_var_out函数，计算标准差，并返回结果
    return std_var_out("std", result, self, dim, correction, keepdim, true);
}

Tensor& std_out(
    const Tensor& self, at::OptionalIntArrayRef dim,
    const std::optional<Scalar>& correction, bool keepdim, Tensor& result) {
    // 调用std_var_out函数，计算标准差，并将结果存储到result中
    return std_var_out("std", result, self, dim, correction, keepdim, true);
}

Tensor& var_out(
    const Tensor& self, at::OptionalIntArrayRef dim,
    const std::optional<Scalar>& correction, bool keepdim, Tensor& result) {
    // 调用std_var_out函数，计算方差，并将结果存储到result中
    return std_var_out("var", result, self, dim, correction, keepdim, false);
}

Tensor var(
    const Tensor& self, at::OptionalIntArrayRef dim,
    const std::optional<Scalar>& correction, bool keepdim) {
    // 创建一个空的张量result，用于存储方差计算的结果
    Tensor result = at::empty({0}, options_to_value_type(self.options()));
    // 调用std_var_out函数，计算方差，并返回结果
    return std_var_out("var", result, self, dim, correction, keepdim, false);
}

Tensor std(const Tensor& self, DimnameList dim, bool unbiased, bool keepdim) {
    // 调用at命名空间中的std函数，计算标准差，并返回结果
    return at::std(self, dimnames_to_positions(self, dim), unbiased, keepdim);
}

Tensor& std_out(const Tensor& self, DimnameList dim, bool unbiased, bool keepdim, Tensor& result) {
    // 调用at命名空间中的std_out函数，计算标准差，并将结果存储到result中
    return at::std_out(result, self, dimnames_to_positions(self, dim), unbiased, keepdim);
}

Tensor var(const Tensor& self, DimnameList dim, bool unbiased, bool keepdim) {
    // 调用at命名空间中的var函数，计算方差，并返回结果
    return at::var(self, dimnames_to_positions(self, dim), unbiased, keepdim);
}

Tensor& var_out(const Tensor& self, DimnameList dim, bool unbiased, bool keepdim, Tensor& result) {
    // 调用at命名空间中的var_out函数，计算方差，并将结果存储到result中
    return at::var_out(
        result, self, dimnames_to_positions(self, dim), unbiased, keepdim);
}

std::tuple<Tensor,Tensor> var_mean(const Tensor& self, DimnameList dim, bool unbiased, bool keepdim) {
    // 调用at命名空间中的var_mean函数，同时返回方差和均值的元组
    return at::var_mean(self, dimnames_to_positions(self, dim), unbiased, keepdim);
}

std::tuple<Tensor,Tensor> std_mean(const Tensor& self, DimnameList dim, bool unbiased, bool keepdim) {
    // 调用at命名空间中的std_mean函数，同时返回标准差和均值的元组
    return at::std_mean(self, dimnames_to_positions(self, dim), unbiased, keepdim);
}

Tensor std(const Tensor& self, DimnameList dim, const std::optional<Scalar>& correction, bool keepdim) {
    // 调用at命名空间中的std函数，计算标准差，并返回结果
    return at::std(self, dimnames_to_positions(self, dim), correction, keepdim);
}

Tensor& std_out(const Tensor& self, DimnameList dim, const std::optional<Scalar>& correction,
                bool keepdim, Tensor& result) {
    // 调用at命名空间中的std_out函数，计算标准差，并将结果存储到result中
    return at::std_out(result, self, dimnames_to_positions(self, dim), correction, keepdim);
}

Tensor var(const Tensor& self, DimnameList dim, const std::optional<Scalar>& correction, bool keepdim) {
    // 调用at命名空间中的var函数，计算方差，并返回结果
    return at::var(self, dimnames_to_positions(self, dim), correction, keepdim);
}
Tensor& var_out(const Tensor& self, DimnameList dim, const std::optional<Scalar>& correction,
                bool keepdim, Tensor& result) {
  return at::var_out(
      result, self, dimnames_to_positions(self, dim), correction, keepdim);
}


# 计算输入张量沿指定维度的方差，并将结果写入预分配的输出张量中
Tensor& var_out(const Tensor& self, DimnameList dim, const std::optional<Scalar>& correction,
                bool keepdim, Tensor& result) {
  return at::var_out(
      result, self, dimnames_to_positions(self, dim), correction, keepdim);
}



std::tuple<Tensor,Tensor> var_mean(const Tensor& self, DimnameList dim,
                                   const std::optional<Scalar>& correction, bool keepdim) {
  return at::var_mean(self, dimnames_to_positions(self, dim), correction, keepdim);
}


# 计算输入张量沿指定维度的方差和均值
std::tuple<Tensor,Tensor> var_mean(const Tensor& self, DimnameList dim,
                                   const std::optional<Scalar>& correction, bool keepdim) {
  return at::var_mean(self, dimnames_to_positions(self, dim), correction, keepdim);
}



std::tuple<Tensor,Tensor> std_mean(const Tensor& self, DimnameList dim,
                                   const std::optional<Scalar>& correction, bool keepdim) {
  return at::std_mean(self, dimnames_to_positions(self, dim), correction, keepdim);
}


# 计算输入张量沿指定维度的标准差和均值
std::tuple<Tensor,Tensor> std_mean(const Tensor& self, DimnameList dim,
                                   const std::optional<Scalar>& correction, bool keepdim) {
  return at::std_mean(self, dimnames_to_positions(self, dim), correction, keepdim);
}



Tensor& norm_out(const Tensor& self, const optional<Scalar>& p, DimnameList dim, bool keepdim, ScalarType dtype, Tensor& result) {
  return at::norm_out(result, self, p, dimnames_to_positions(self, dim), keepdim, dtype);
}


# 计算输入张量沿指定维度的范数，并将结果写入预分配的输出张量中
Tensor& norm_out(const Tensor& self, const optional<Scalar>& p, DimnameList dim, bool keepdim, ScalarType dtype, Tensor& result) {
  return at::norm_out(result, self, p, dimnames_to_positions(self, dim), keepdim, dtype);
}



Tensor& norm_out(const Tensor& self, const optional<Scalar>& p, DimnameList dim, bool keepdim, Tensor& result) {
  return at::norm_out(result, self, p, dimnames_to_positions(self, dim), keepdim);
}


# 计算输入张量沿指定维度的范数，并将结果写入预分配的输出张量中
Tensor& norm_out(const Tensor& self, const optional<Scalar>& p, DimnameList dim, bool keepdim, Tensor& result) {
  return at::norm_out(result, self, p, dimnames_to_positions(self, dim), keepdim);
}



Tensor norm(const Tensor& self, const optional<Scalar>& p, DimnameList dim, bool keepdim, ScalarType dtype) {
  return at::norm(self, p, dimnames_to_positions(self, dim), keepdim, dtype);
}


# 计算输入张量沿指定维度的范数
Tensor norm(const Tensor& self, const optional<Scalar>& p, DimnameList dim, bool keepdim, ScalarType dtype) {
  return at::norm(self, p, dimnames_to_positions(self, dim), keepdim, dtype);
}



Tensor norm(const Tensor& self, const optional<Scalar>& p, DimnameList dim, bool keepdim) {
  return at::norm(self, p, dimnames_to_positions(self, dim), keepdim);
}


# 计算输入张量沿指定维度的范数
Tensor norm(const Tensor& self, const optional<Scalar>& p, DimnameList dim, bool keepdim) {
  return at::norm(self, p, dimnames_to_positions(self, dim), keepdim);
}



Tensor any(const Tensor& self, Dimname dim, bool keepdim) {
  reportNYIDimnameOverload("any");
}


# 报告未实现的 Dimname 版本的 any 操作
Tensor any(const Tensor& self, Dimname dim, bool keepdim) {
  reportNYIDimnameOverload("any");
}



Tensor& any_out(const Tensor &self, Dimname dim, bool keepdim, Tensor& result) {
  reportNYIDimnameOverload("any");
}


# 报告未实现的 Dimname 版本的 any_out 操作
Tensor& any_out(const Tensor &self, Dimname dim, bool keepdim, Tensor& result) {
  reportNYIDimnameOverload("any");
}



Tensor all(const Tensor& self, Dimname dim, bool keepdim) {
  reportNYIDimnameOverload("all");
}


# 报告未实现的 Dimname 版本的 all 操作
Tensor all(const Tensor& self, Dimname dim, bool keepdim) {
  reportNYIDimnameOverload("all");
}



Tensor& all_out(const Tensor &self, Dimname dim, bool keepdim, Tensor& result) {
  reportNYIDimnameOverload("all");
}


# 报告未实现的 Dimname 版本的 all_out 操作
Tensor& all_out(const Tensor &self, Dimname dim, bool keepdim, Tensor& result) {
  reportNYIDimnameOverload("all");
}



Tensor _is_all_true(const Tensor& self) {
  TORCH_INTERNAL_ASSERT(self.scalar_type() == at::kBool);
  return self.all();
}


# 检查输入张量是否为布尔类型，如果是则计算其所有元素的逻辑与
Tensor _is_all_true(const Tensor& self) {
  TORCH_INTERNAL_ASSERT(self.scalar_type() == at::kBool);
  return self.all();
}



Tensor _is_any_true(const Tensor& self) {
  TORCH_INTERNAL_ASSERT(self.scalar_type() == at::kBool);
  return self.any();
}


# 检查输入张量是否为布尔类型，如果是则计算其所有元素的逻辑或
Tensor _is_any_true(const Tensor& self) {
  TORCH_INTERNAL_ASSERT(self.scalar_type() == at::kBool);
  return self.any();
}



Tensor logcumsumexp(const Tensor& self, Dimname dim) {
  return at::logcumsumexp(self, dimname_to_position(self, dim));
}


# 计算输入张量沿指定维度的对数累积和指数函数
Tensor logcumsumexp(const Tensor& self, Dimname dim) {
  return at::logcumsumexp(self, dimname_to_position(self, dim));
}



Tensor& logcumsumexp_out(const Tensor& self, Dimname dim, Tensor& result) {
  return at::logcumsumexp_out(result, self, dimname_to_position(self, dim));
}


# 计算输入张量沿指定维度的对数累积和指数函数，并将结果写入预分配的输出张量中
Tensor& logcumsumexp_out(const Tensor& self, Dimname dim, Tensor& result) {
  return at::logcumsumexp_out(result, self, dimname_to_position(self, dim));
}



Tensor cumsum(const Tensor& self, Dimname dim, std::optional<ScalarType> dtype) {
  return at::cumsum(self, dimname_to_position(self, dim), dtype);
}


# 计算输入张量沿指定维度的累积和
Tensor cumsum(const Tensor& self, Dimname dim, std::optional<ScalarType> dtype) {
  return at::cumsum(self, dimname_to_position(self, dim), dtype);
}



Tensor& cumsum_(Tensor& self, Dimname dim, std::optional<ScalarType> dtype) {
  return at::cumsum_out(self, self, dimname_to_position(self, dim), dtype);
}


# 计算输入张量沿指定维度的累积和，并将结果写入原张量
Tensor& cumsum_(Tensor& self, Dimname dim, std::optional<ScalarType> dtype) {
  return at::cumsum_out(self, self, dimname_to_position(self, dim), dtype);
}
``
// 计算给定维度上的累积乘积，并返回新的张量
Tensor cumprod(const Tensor& self, Dimname dim, std::optional<ScalarType> dtype) {
  return at::cumprod(self, dimname_to_position(self, dim), dtype);
}

// 在给定维度上计算累积乘积，并将结果存储在输入张量中
Tensor& cumprod_(Tensor& self, Dimname dim, std::optional<ScalarType> dtype) {
  return at::cumprod_out(self, self, dimname_to_position(self, dim), dtype);
}

// 在给定维度上计算累积乘积，并将结果存储在指定的结果张量中
Tensor& cumprod_out(const Tensor& self, Dimname dim, std::optional<ScalarType> dtype, Tensor& result) {
  return at::cumprod_out(result, self, dimname_to_position(self, dim), dtype);
}

// 计算给定维度上的累积最大值和对应索引，并返回结果元组
std::tuple<Tensor, Tensor> cummax(const Tensor& self, Dimname dim) {
  return at::cummax(self, dimname_to_position(self, dim));
}

// 在给定维度上计算累积最大值和对应索引，并将结果存储在指定的值和索引张量中
std::tuple<Tensor&, Tensor&> cummax_out(const Tensor& self, Dimname dim, Tensor& values, Tensor& indices) {
  return at::cummax_out(values, indices, self, dimname_to_position(self, dim));
}

// 计算给定维度上的累积最小值和对应索引，并返回结果元组
std::tuple<Tensor, Tensor> cummin(const Tensor& self, Dimname dim) {
  return at::cummin(self, dimname_to_position(self, dim));
}

// 在给定维度上计算累积最小值和对应索引，并将结果存储在指定的值和索引张量中
std::tuple<Tensor&, Tensor&> cummin_out(const Tensor& self, Dimname dim, Tensor& values, Tensor& indices) {
  return at::cummin_out(values, indices, self, dimname_to_position(self, dim));
}

// 计算两个张量之间的距离，使用指定的范数 p
Tensor dist(const Tensor &self, const Tensor& other, const Scalar& p){
  return at::norm(self - other, p);
}

// 检查两个 CPU 上的张量是否相等
bool cpu_equal(const Tensor& self, const Tensor& other) {
  // 检查张量的命名是否相同
  if (!at::namedinference::are_names_equal(
        self.unsafeGetTensorImpl(), other.unsafeGetTensorImpl())) {
    return false;
  }
  // 禁用名称保护以进行设备检查
  at::NoNamesGuard guard;
  // 检查张量是否在相同设备上
  TORCH_CHECK(self.device() == other.device(), "Cannot compare two tensors on "
              "different devices. Got: ", self.device(), " and ", other.device());
  // 检查张量是否具有相同的大小
  if (!self.is_same_size(other)) {
    return false;
  }
  // 快速路径：比较存储、偏移、数据类型等是否完全相同
  if (self.is_alias_of(other)
      && self.storage_offset() == other.storage_offset()
      && self.dtype() == other.dtype()
      && self.is_contiguous() == other.is_contiguous()
      && self.strides().equals(other.strides())
      && self.layout() == other.layout()
      && self.is_neg() == other.is_neg()
      && self.is_conj() == other.is_conj()) {
    if (c10::isIntegralType(self.scalar_type(), /*includeBool=*/true)) {
      return true;
    }
    // 如果是整数类型，直接返回 true
    std::atomic<bool> result{true};
    // 构建张量迭代器
    auto iter = TensorIteratorConfig().add_const_input(self).build();
    // 继续执行更复杂的检查
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(kHalf, kBFloat16, iter.input_dtype(), "equal_notnan_cpu", [&] {
      iter.for_each([&](char** data, const int64_t *strides, int64_t dim_size) {
        // 如果 result 为 false，直接返回
        if (!result) {
            return;
        }
        // 获取自身数据的指针
        char* self_data = data[0];
        // 遍历维度大小
        for (C10_UNUSED const auto i : c10::irange(dim_size)) {
          // 如果当前元素是 NaN，则将 result 置为 false，并返回
          if (isnan_(c10::load<scalar_t>(self_data))) {
            result = false;
            return;
          }
          // 更新自身数据指针，根据步长 strides[0]
          self_data += strides[0];
        }
      });
    });
    // 返回 result 的当前值
    return result.load();
  }

  // 初始化一个原子布尔变量 result，并设为 true
  std::atomic<bool> result{true};
  // 根据 self 和 other 构建一个迭代器 iter
  auto iter = TensorIteratorConfig()
    .add_const_input(self)
    .add_const_input(other)
    .allow_cpu_scalars(true)
    .promote_inputs_to_common_dtype(true)
    .build();

  // 根据 iter 的输入数据类型进行分发，执行以下操作
  AT_DISPATCH_V2(iter.input_dtype(), "equal_cpu", AT_WRAP([&] {
    iter.for_each([&](char** data, const int64_t *strides, int64_t dim_size) {
      // 如果 result 为 false，直接返回
      if (!result) {
          return;
      }
      // 获取自身和其他数据的指针
      char* self_data = data[0];
      char* other_data = data[1];
      // 遍历维度大小
      for (C10_UNUSED const auto i : c10::irange(dim_size)) {
        // 如果自身和其他数据的当前元素不相等，则将 result 置为 false，并返回
        if (c10::load<scalar_t>(self_data) != c10::load<scalar_t>(other_data)) {
          result = false;
          return;
        }
        // 更新自身和其他数据指针，根据各自的步长 strides[0] 和 strides[1]
        self_data += strides[0];
        other_data += strides[1];
      }
    });
  }), kBool, kBFloat16, kHalf, AT_EXPAND(AT_ALL_TYPES_AND_COMPLEX), AT_EXPAND(AT_BAREBONES_UNSIGNED_TYPES));
  // 返回 result 的当前值
  return result.load();
// `}` 结束了 `at::native` 命名空间的定义

// max(dim), min(dim), topk(dim), mode(dim) 是选择值的归约函数示例，
// value_selecting_reduction_backward 是这些操作符的反向函数；
// 它将梯度传播到`indices`指定的具体值位置。
Tensor value_selecting_reduction_backward_symint(const Tensor& grad, int64_t dim, const Tensor& indices, c10::SymIntArrayRef sizes, bool keepdim) {
  // 定义一个 lambda 函数 inplace_scatter_if_not_tensor_subclass，用于根据条件原地散布梯度
  auto inplace_scatter_if_not_tensor_subclass =
      [&](const Tensor& grad_out, const Tensor& indices_) {
        // 创建一个与 grad_out 具有相同大小的零张量 grad_in
        auto grad_in = at::zeros_symint(sizes, grad_out.options());
        // 如果 grad 或 indices 中有任何张量子类，则返回散布后的结果
        if (areAnyTensorSubclassLike({grad, indices})) {
          return grad_in.scatter(dim, indices_, grad_out);
        }
        // 否则原地散布并返回结果
        return grad_in.scatter_(dim, indices_, grad_out);
      };

  // 如果不保留维度并且 sizes 非空
  if (!keepdim && !sizes.empty()) {
    // 在维度 dim 上对 grad 和 indices 进行unsqueeze操作
    auto grad_ = grad.unsqueeze(dim);
    auto indices_ = indices.unsqueeze(dim);
    // 调用原地散布函数 inplace_scatter_if_not_tensor_subclass
    return inplace_scatter_if_not_tensor_subclass(grad_, indices_);
  }
  // 否则直接调用原地散布函数 inplace_scatter_if_not_tensor_subclass
  return inplace_scatter_if_not_tensor_subclass(grad, indices);
}

// 对于稀疏张量的 CSR 格式，计算其值的和
Tensor sum_csr(const Tensor &self, std::optional<ScalarType> dtype) {
  // 调用 self.values() 返回的张量值的和，可选指定数据类型 dtype
  return self.values().sum(dtype);
}

// 对于 COO 格式的稀疏张量，计算其值的和
Tensor sum_coo(const Tensor &self, std::optional<ScalarType> dtype) {
  // 调用 self._values() 返回的稀疏张量值的和，可选指定数据类型 dtype
  return self._values().sum(dtype);
}

// 对 COO 格式的稀疏张量，按维度 dim 计算其值的和
Tensor sum_sparse_coo(const Tensor& self, at::OptionalIntArrayRef dim, bool keepdim, std::optional<ScalarType> dtype) {
  Tensor result;
  // 如果 dim 有值
  if (dim.has_value()) {
    // 如果指定了 dtype
    if (dtype.has_value()) {
      // 调用 at::_sparse_sum 函数计算稀疏张量 self 在维度 dim 上的和，使用指定的 dtype
      result = at::_sparse_sum(self, *dim, *dtype);
    } else {
      // 如果 self 的标量类型是整型，则使用 Long 类型计算和
      if (c10::isIntegralType(self.scalar_type(), true)) {
        result = at::_sparse_sum(self, *dim, at::kLong);
      } else {
        // 否则使用默认的数据类型计算和
        result = at::_sparse_sum(self, *dim);
      }
    }
  } else {
    // 如果 dim 没有值，则调用 sum_coo 函数计算 COO 格式的稀疏张量 self 的和
    result = sum_coo(self, dtype);
  }
  // 如果保留维度，则根据 dim_mask 对结果进行 unsqueeze 操作
  if (keepdim) {
    auto dim_mask = make_dim_mask(dim, self.dim());
    for (int dim = 0; dim < self.dim(); dim++) {
      if (dim_mask[dim]) {
        result = result.unsqueeze(dim);
      }
    }
  }
  // 返回计算得到的结果
  return result;
}

// 对压缩稀疏格式的稀疏张量进行求和计算
Tensor sum_sparse_compressed(
    const Tensor& self,
    at::OptionalIntArrayRef dim,
    bool keepdim,
    std::optional<ScalarType> dtype) {
  // TODO: sum.dim_IntList 和 _sparse_csr_sum.dim_dtype 的签名有点不同，这导致了 `dim` 的转换问题
  // 在第二个参数上调用 `_sparse_csr_sum` 应该是更好的选择。将签名对齐将是更好的选择。
  // 检查 dim 是否有值，否则抛出错误信息
  TORCH_CHECK(
      dim.has_value(), "dim has no value, cannot be used in sum.dim_IntList");
  // 检查 self 的布局是否为 kSparseCsr，否则抛出错误信息
  auto layout = self.layout();
  TORCH_CHECK(
      layout == kSparseCsr,
      "Currently the only compressed sparse format supported for sum.dim_IntList is CSR, but got layout ",
      layout)
  // 调用 at::_sparse_csr_sum 函数计算 CSR 格式的稀疏张量 self 在指定维度 dim 上的和
  return at::_sparse_csr_sum(self, *dim, keepdim, dtype);
}
```