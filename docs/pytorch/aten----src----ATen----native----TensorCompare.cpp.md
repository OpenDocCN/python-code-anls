# `.\pytorch\aten\src\ATen\native\TensorCompare.cpp`

```
// 定义宏，用于限制仅包含方法运算符
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS

// 包含相关的头文件，用于张量操作和计算
#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/ScalarOps.h>
#include <ATen/TensorIndexing.h>
#include <ATen/TensorMeta.h>
#include <ATen/TensorOperators.h>
#include <ATen/WrapDimUtils.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <ATen/native/Resize.h>
#include <ATen/native/TensorCompare.h>
#include <ATen/native/TypeProperties.h>
#include <ATen/TensorSubclassLikeUtils.h>

// 包含标准输入输出流的头文件
#include <iostream>

// 包含 C10 库中异常处理的头文件
#include <c10/util/Exception.h>

// 如果未定义 AT_PER_OPERATOR_HEADERS 宏，则包含以下头文件
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
// 如果定义了 AT_PER_OPERATOR_HEADERS 宏，则包含以下头文件
#else
#include <ATen/ops/_aminmax_native.h>
#include <ATen/ops/_assert_async_native.h>
#include <ATen/ops/_functional_assert_async_native.h>
#include <ATen/ops/_print_native.h>
#include <ATen/ops/_assert_scalar_native.h>
#include <ATen/ops/_functional_assert_scalar_native.h>
#include <ATen/ops/_make_per_tensor_quantized_tensor.h>
#include <ATen/ops/_unique.h>
#include <ATen/ops/allclose_native.h>
#include <ATen/ops/aminmax.h>
#include <ATen/ops/argsort_native.h>
#include <ATen/ops/cat.h>
#include <ATen/ops/clamp.h>
#include <ATen/ops/clamp_max.h>
#include <ATen/ops/clamp_max_native.h>
#include <ATen/ops/clamp_min.h>
#include <ATen/ops/clamp_min_native.h>
#include <ATen/ops/clamp_native.h>
#include <ATen/ops/clip_native.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/eq.h>
#include <ATen/ops/fill.h>
#include <ATen/ops/imag.h>
#include <ATen/ops/index.h>
#include <ATen/ops/is_nonzero_native.h>
#include <ATen/ops/isclose.h>
#include <ATen/ops/isclose_native.h>
#include <ATen/ops/isfinite.h>
#include <ATen/ops/isfinite_native.h>
#include <ATen/ops/isin.h>
#include <ATen/ops/isin_native.h>
#include <ATen/ops/isinf.h>
#include <ATen/ops/isinf_native.h>
#include <ATen/ops/isnan_native.h>
#include <ATen/ops/isneginf_native.h>
#include <ATen/ops/isposinf_native.h>
#include <ATen/ops/isreal_native.h>
#include <ATen/ops/max.h>
#include <ATen/ops/max_native.h>
#include <ATen/ops/min.h>
#include <ATen/ops/min_native.h>
#include <ATen/ops/mode.h>
#include <ATen/ops/mode_native.h>
#include <ATen/ops/ne.h>
#include <ATen/ops/ones_like.h>
#include <ATen/ops/real.h>
#include <ATen/ops/result_type_native.h>
#include <ATen/ops/scalar_tensor.h>
#include <ATen/ops/where.h>
#include <ATen/ops/where_native.h>
#include <ATen/ops/zeros_like.h>
#endif

// 命名空间定义，命名空间为 at::meta
namespace at::meta {

// 静态内联函数，检查不支持的 isin 函数的数据类型
static inline void check_for_unsupported_isin_dtype(const ScalarType type) {
    // 检查是否为不支持的数据类型，如果是则抛出异常
    TORCH_CHECK(type != ScalarType::Bool &&
        type != ScalarType::BFloat16 &&
        type != ScalarType::ComplexFloat &&
        type != ScalarType::ComplexDouble,
        "Unsupported input type encountered for isin(): ", type);
}

// TORCH_META_FUNC 宏，定义了名为 clamp 的元函数
TORCH_META_FUNC(clamp) (
    // self 参数，表示要执行 clamp 操作的张量
    const Tensor& self,
    // min 参数，可选的最小值标量引用
    const OptionalScalarRef min,
    // 最大值标量引用
    const OptionalScalarRef max
    ) {
const OptionalScalarRef max) {
  // 检查 'min' 和 'max' 必须至少有一个不为 None
  if (!min && !max) {
    TORCH_CHECK(false, "torch.clamp: At least one of 'min' or 'max' must not be None");
  }
  // 手动进行类型提升，因为标量必须参与其中
  ScalarType result_type = self.scalar_type();
  // 检查是否为复数类型，clamp 不支持复数类型
  TORCH_CHECK(!isComplexType(result_type), "clamp is not supported for complex types");
  // 浮点型是支持的最高类型
  if (!isFloatingType(result_type)) {
    // 更新结果类型状态
    at::native::ResultTypeState state = {};
    state = at::native::update_result_type_state(self, state);

    if (min) {
      state = at::native::update_result_type_state(min.get(), state);
    }
    if (max) {
      state = at::native::update_result_type_state(max.get(), state);
    }
    // 获取最终的结果类型
    result_type = at::native::result_type(state);
    // 禁止原地操作中的类型提升
    TORCH_CHECK((result_type == self.scalar_type()) ||
       (!(maybe_get_output().defined()) || !(maybe_get_output().is_same(self))),
       "result type ", result_type, " can't be cast to the desired output type ",
       self.dtype());
  }
  // 再次确保结果类型不是复数类型
  TORCH_CHECK(!isComplexType(result_type), "clamp is not supported for complex types");
  // 构建一元操作，将 self 张量转换为 result_type 类型
  build_unary_op(maybe_get_output(), self.to(result_type));
}

TORCH_META_FUNC2(clamp, Tensor) (
const Tensor& self,
const OptionalTensorRef min,
const OptionalTensorRef max) {
  // 至少需要一个 'min' 或 'max' 不为 None
  TORCH_CHECK(min || max, "torch.clamp: At least one of 'min' or 'max' must not be None");
  // 检查是否为复数类型，clamp 不支持复数类型
  TORCH_CHECK(!isComplexType(self.scalar_type()), "clamp is not supported for complex types");
  // 定义宏 CLAMP_CONFIG，用于设置张量迭代器的配置
  #define CLAMP_CONFIG()                    \
    TensorIteratorConfig()                  \
      .set_check_mem_overlap(true)          \
      .add_output(maybe_get_output())       \
      .add_const_input(self)                \
      .promote_inputs_to_common_dtype(true) \
      .cast_common_dtype_to_outputs(true)   \
      .enforce_safe_casting_to_output(true)

  if (min && max) {
    // 如果同时提供了 'min' 和 'max'，使用两者作为输入构建张量迭代器
    build(CLAMP_CONFIG().add_const_input(*min).add_const_input(*max));
  } else if (min) {
    // 如果只提供了 'min'，使用 'min' 作为输入构建张量迭代器
    build(CLAMP_CONFIG().add_const_input(*min));
  } else if (max) {
    // 如果只提供了 'max'，使用 'max' 作为输入构建张量迭代器
    build(CLAMP_CONFIG().add_const_input(*max));
  }
}

TORCH_META_FUNC(clamp_max) (
  const Tensor& self,
  const Scalar& max
) {
  // 可以将 max 包装成张量并发送到张量重载，但是为了性能和一致性的原因，通过更快但正确的方法实现
  ScalarType result_type = self.scalar_type();
  // 检查是否为复数类型，clamp 不支持复数类型
  TORCH_CHECK(!isComplexType(result_type), "clamp is not supported for complex types");
  // 检查 max 不是复数类型
  TORCH_CHECK(!max.isComplex(), "clamp is not supported for complex types");
  // 浮点型是支持的最高类型
  if (!isFloatingType(result_type)) {
    // 获取结果类型
    auto result_type = at::native::result_type(self, max);
    // 禁止类型提升
    TORCH_CHECK((result_type == self.scalar_type()) ||
       (!(maybe_get_output().defined()) || !(maybe_get_output().is_same(self))),
       "result type ", result_type, " can't be cast to the desired output type ",
       self.dtype());
    // 如果条件成立，执行以下代码块
    build_unary_op(maybe_get_output(), self.to(result_type));
  } else {
    // 如果条件不成立，执行以下代码块
    build_borrowing_unary_op(maybe_get_output(), self);
  }
TORCH_META_FUNC2(clamp_max, Tensor) (
  const Tensor& self,  // 第一个参数：输入张量 self
  const Tensor& max    // 第二个参数：最大值张量 max
) {
  build_borrowing_binary_op(maybe_get_output(), self, max);  // 调用函数构建一个二元操作，结果可能存储在 maybe_get_output() 中
}


TORCH_META_FUNC(clamp_min) (
  const Tensor& self,   // 第一个参数：输入张量 self
  const Scalar& min     // 第二个参数：最小标量值 min
) {
  ScalarType result_type = self.scalar_type();  // 获取输入张量的数据类型
  TORCH_CHECK(!isComplexType(result_type), "clamp is not supported for complex types");  // 检查是否为复数类型
  TORCH_CHECK(!min.isComplex(), "clamp is not supported for complex types");  // 检查最小值是否为复数
  //Floating is the highest supported
  if (!isFloatingType(result_type)) {  // 如果数据类型不是浮点类型
    auto result_type = at::native::result_type(self, min);  // 获取输入张量和最小标量值的结果类型
    TORCH_CHECK((result_type == self.scalar_type() ||
       !(maybe_get_output().defined()) || !(maybe_get_output().is_same(self))),
       "result type ", result_type, " can't be cast to the desired output type ",
       self.dtype());  // 检查结果类型是否能够转换为期望的输出类型
    build_unary_op(maybe_get_output(), self.to(result_type));  // 调用函数构建一个一元操作，结果存储在 maybe_get_output() 中
  } else {
    build_borrowing_unary_op(maybe_get_output(), self);  // 调用函数构建一个借用的一元操作，结果存储在 maybe_get_output() 中
  }
}

TORCH_META_FUNC2(clamp_min, Tensor) (
  const Tensor& self,   // 第一个参数：输入张量 self
  const Tensor& min     // 第二个参数：最小值张量 min
) {
  build_borrowing_binary_op(maybe_get_output(), self, min);  // 调用函数构建一个二元操作，结果可能存储在 maybe_get_output() 中
}

TORCH_META_FUNC2(isin, Tensor_Tensor) (
  const Tensor& elements,  // 第一个参数：元素张量 elements
  const Tensor& test_elements,  // 第二个参数：测试元素张量 test_elements
  bool /*assume_unique*/,  // 忽略的布尔参数，假设唯一
  bool /*invert*/          // 忽略的布尔参数，反转
) {
  check_for_unsupported_isin_dtype(elements.scalar_type());  // 检查元素张量数据类型是否支持
  check_for_unsupported_isin_dtype(test_elements.scalar_type());  // 检查测试元素张量数据类型是否支持
  // 设置输出张量的形状、步长和数据类型为布尔类型，基于测试元素张量的设备
  set_output_raw_strided(0, elements.sizes(), {}, TensorOptions(elements.device()).dtype(ScalarType::Bool));
}

TORCH_META_FUNC2(isin, Tensor_Scalar) (
  const Tensor& elements,   // 第一个参数：元素张量 elements
  const c10::Scalar& test_elements,  // 第二个参数：测试标量值 test_elements
  bool /*assume_unique*/,   // 忽略的布尔参数，假设唯一
  bool /*invert*/           // 忽略的布尔参数，反转
) {
  check_for_unsupported_isin_dtype(elements.scalar_type());  // 检查元素张量数据类型是否支持
  check_for_unsupported_isin_dtype(test_elements.type());    // 检查测试标量值数据类型是否支持
  // 设置输出张量的形状、步长和数据类型为布尔类型，基于元素张量的设备
  set_output_raw_strided(0, elements.sizes(), {}, TensorOptions(elements.device()).dtype(ScalarType::Bool));
}

TORCH_META_FUNC2(isin, Scalar_Tensor) (
  const c10::Scalar& elements,   // 第一个参数：元素标量值 elements
  const Tensor& test_elements,   // 第二个参数：测试元素张量 test_elements
  bool /*assume_unique*/,        // 忽略的布尔参数，假设唯一
  bool /*invert*/                // 忽略的布尔参数，反转
) {
  check_for_unsupported_isin_dtype(elements.type());  // 检查元素标量值数据类型是否支持
  check_for_unsupported_isin_dtype(test_elements.scalar_type());  // 检查测试元素张量数据类型是否支持
  // 设置输出张量的形状为空、步长为空，数据类型为布尔类型，基于测试元素张量的设备
  set_output_raw_strided(0, {0}, {}, TensorOptions(test_elements.device()).dtype(ScalarType::Bool));
}

TORCH_META_FUNC(isposinf) (const Tensor& self) {
  TORCH_CHECK(!self.is_complex(), "isposinf does not support complex inputs.");  // 检查输入张量是否为复数类型
  TORCH_CHECK(maybe_get_output().defined() ? maybe_get_output().dtype() == at::kBool : true,
              "isposinf does not support non-boolean outputs.");  // 检查输出是否为布尔类型
  // 调用函数构建一个强制布尔值的借用一元操作，结果存储在 maybe_get_output() 中
  build_borrowing_unary_force_boolean_op(maybe_get_output(), self);
}

TORCH_META_FUNC(isneginf) (const Tensor& self) {
  TORCH_CHECK(!self.is_complex(), "isneginf does not support complex inputs.");  // 检查输入张量是否为复数类型
  TORCH_CHECK(maybe_get_output().defined() ? maybe_get_output().dtype() == at::kBool : true,
              "isneginf does not support non-boolean outputs.");  // 检查输出是否为布尔类型
  // 调用函数构建一个强制布尔值的借用一元操作，结果存储在 maybe_get_output() 中
  build_borrowing_unary_force_boolean_op(maybe_get_output(), self);
}
// 检查输入张量是否包含复数，若包含则抛出错误信息
static void check_unsupported_complex(const char* name, const Tensor& self) {
  TORCH_CHECK(!self.is_complex(), name, ": does not support complex input");
}

// 预计算函数：max(dim)
TORCH_PRECOMPUTE_META_FUNC2(max, dim)(const Tensor& self, int64_t dim, bool keepdim) {
  // 将维度 dim 包装至合法范围内
  dim = maybe_wrap_dim(dim, self.dim());
  // 检查零维张量的异常情况
  at::native::zero_numel_check_dims(self, dim, "max()");
  // 检查输入张量是否包含复数，若包含则抛出错误信息
  check_unsupported_complex("max()", self);
  // 调整用于索引和尺寸约简的张量大小
  resize_reduction_with_indices(*this, self, dim, keepdim, self.scalar_type());
  // 返回预计算结构 max(dim)，设置合法化的维度
  return TORCH_PRECOMPUTE_STRUCT2(max, dim)()
      .set_dim(maybe_wrap_dim(dim, self.dim()));
}

// 预计算函数：min(dim)
TORCH_PRECOMPUTE_META_FUNC2(min, dim)(const Tensor& self, int64_t dim, bool keepdim) {
  // 将维度 dim 包装至合法范围内
  dim = maybe_wrap_dim(dim, self.dim());
  // 检查零维张量的异常情况
  at::native::zero_numel_check_dims(self, dim, "min()");
  // 检查输入张量是否包含复数，若包含则抛出错误信息
  check_unsupported_complex("min()", self);
  // 调整用于索引和尺寸约简的张量大小
  resize_reduction_with_indices(*this, self, dim, keepdim, self.scalar_type());
  // 返回预计算结构 min(dim)，设置合法化的维度
  return TORCH_PRECOMPUTE_STRUCT2(min, dim)()
      .set_dim(maybe_wrap_dim(dim, self.dim()));
}

} // namespace at::meta

namespace at::native {

// 定义分派函数：where_kernel
DEFINE_DISPATCH(where_kernel); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
// 定义分派函数：max_stub
DEFINE_DISPATCH(max_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
// 定义分派函数：min_stub
DEFINE_DISPATCH(min_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
// 定义分派函数：isposinf_stub
DEFINE_DISPATCH(isposinf_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
// 定义分派函数：isneginf_stub
DEFINE_DISPATCH(isneginf_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
// 定义分派函数：mode_stub
DEFINE_DISPATCH(mode_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
// 定义分派函数：clamp_stub
DEFINE_DISPATCH(clamp_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
// 定义分派函数：clamp_scalar_stub
DEFINE_DISPATCH(clamp_scalar_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
// 定义分派函数：clamp_min_scalar_stub
DEFINE_DISPATCH(clamp_min_scalar_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
// 定义分派函数：clamp_max_scalar_stub
DEFINE_DISPATCH(clamp_max_scalar_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
// 定义分派函数：isin_default_stub
DEFINE_DISPATCH(isin_default_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)

// 检查张量 self 和 other 是否全部接近
bool allclose(const Tensor& self, const Tensor& other, double rtol, double atol, bool equal_nan) {
  // 调用 at::isclose 判断张量 self 和 other 是否接近，返回布尔值
  return at::isclose(self, other, rtol, atol, equal_nan).all().item<uint8_t>();
}

// Note [closeness]
// 数字 A 接近于 B 当且仅当：
//
// (1) A 等于 B，且当 equal_nan 为 true 时 NaN 值也视为相等。
// (2) 误差 abs(A - B) 是有限的，并且小于最大误差 (atol + abs(rtol * B))。
//
// 注意这与 NumPy 的 isclose 一致，但与 Python 的 isclose 不同，
// 后者计算的最大误差为 max(rtol * max(abs(A), abs(B)), atol)。
// TODO: 一旦添加位操作符重载，重新审视复数输入和 equal_nan=true 的情况
// TODO: 在 https://github.com/numpy/numpy/issues/15959 解决后重新审视
// 定义函数 `isclose`，用于检查两个张量是否在给定的相对和绝对误差范围内相等
Tensor isclose(const Tensor& self, const Tensor& other, double rtol, double atol, bool equal_nan) {
  // 检查输入张量的数据类型是否匹配
  TORCH_CHECK(self.scalar_type() == other.scalar_type(), self.scalar_type(), " did not match ", other.scalar_type());
  // 检查输入张量是否为量化张量，量化张量不支持 `isclose`
  TORCH_CHECK(!(self.is_quantized() || other.is_quantized()),
    "isclose is not supported for quantized inputs.");

  // 检查相对误差 `rtol` 和绝对误差 `atol` 是否为非负数
  // 注意：与 Python 的 `isclose` 一致，但与 NumPy 不同，NumPy 允许负的 `atol` 和 `rtol`
  TORCH_CHECK(rtol >= 0, "rtol must be greater than or equal to zero, but got ", rtol);
  TORCH_CHECK(atol >= 0, "atol must be greater than or equal to zero, but got ", atol);

  // 计算是否相等的张量 `close`
  Tensor close = self == other;
  if (equal_nan && (self.is_floating_point() || self.is_complex())) {
    // 如果 `equal_nan` 为真且输入张量为浮点或复数类型，则处理 NaN 值的情况
    if (isTensorSubclassLike(other)) {
      close.__ior__(self.isnan().bitwise_and(other.isnan()));
    } else {
      close.__ior__(self.isnan().__iand__(other.isnan()));
    }
  }

  // 当 `rtol` 和 `atol` 均为零时，直接返回 `close`，避免误报
  if (rtol == 0 && atol == 0){
      return close;
  }

  // 计算允许的误差和实际误差
  // 将 `self` 和 `other` 张量转换为默认的数据类型，以便进行计算
  Tensor cast_self, cast_other;
  cast_self = self.scalar_type() == at::kBool ? self.to(at::get_default_dtype()) : self;
  if (c10::isIntegralType(self.scalar_type(), /*includeBool=*/true)) {
    cast_other = other.to(at::get_default_dtype());
  } else {
    cast_other = other;
  }

  // 计算允许的误差范围和实际误差
  Tensor allowed_error = atol + (rtol * cast_other).abs();
  Tensor actual_error = (cast_self - cast_other).abs();

  // 计算有限数的相等性，更新 `close` 张量
  close.__ior__(at::isfinite(actual_error).__iand__(actual_error <= allowed_error));

  // 返回最终的相等性结果 `close`
  return close;
}

// 定义函数 `isnan`，用于检查张量中的 NaN 值
Tensor isnan(const Tensor& self) {
  return self != self;  // 返回一个张量，标记出输入张量中的 NaN 值
}
Tensor isreal(const Tensor& self) {
  // Note: Integral and Floating tensor values are always real
  // 检查张量是否是整数类型或浮点数类型，这些类型的张量值始终是实数
  if (c10::isIntegralType(self.scalar_type(), /*includeBool=*/true) ||
      c10::isFloatingType(self.scalar_type())) {
    // 创建一个与输入张量相同形状的布尔类型张量，所有元素初始化为1
    return at::ones_like(self, at::kBool, at::MemoryFormat::Preserve);
  }

  // 对于复数张量，判断其虚部是否为零来确定是否是实数
  return at::imag(self) == 0;
}


#if !defined(C10_MOBILE)
#define _AT_DISPATCH_INF_TYPES(TYPE, NAME, ...)                          \
        AT_DISPATCH_FLOATING_TYPES_AND3( kHalf, kBFloat16, kFloat8_e5m2, \
            TYPE, NAME, __VA_ARGS__)
#else
#define _AT_DISPATCH_INF_TYPES(TYPE, NAME, ...)           \
        AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16, \
            TYPE, NAME, __VA_ARGS__)
#endif


Tensor isinf(const Tensor &self) {
  // Note: Integral tensor values are never infinite
  // 检查张量是否是整数类型，这些类型的张量值永远不是无穷大
  if (c10::isIntegralType(self.scalar_type(), /*includeBool=*/true)) {
    // 创建一个与输入张量相同形状的布尔类型张量，所有元素初始化为0
    return at::zeros_like(self, at::kBool, at::MemoryFormat::Preserve);
  }

  // 复数张量的判断：复数是无穷大的当且仅当实部或虚部有一部分是无穷大
  if (self.is_complex()) {
    // 利用位操作符 __ior__ 对实部和虚部的 isinf 结果进行按位或操作
    return at::isinf(at::real(self)).__ior__
          (at::isinf(at::imag(self)));
  }

  // 对于浮点类型，调用对应类型的分发宏，并返回判断结果
  return _AT_DISPATCH_INF_TYPES(self.scalar_type(), "isinf", [&]() {
    return self.abs() == std::numeric_limits<scalar_t>::infinity();
  });
}

Tensor isfinite(const Tensor& self) {
  // Note: Integral tensor values are always finite
  // 检查张量是否是整数类型，这些类型的张量值始终是有限的
  if (c10::isIntegralType(self.scalar_type(), /*includeBool=*/true)) {
    // 创建一个与输入张量相同形状的布尔类型张量，所有元素初始化为1
    return at::ones_like(self, at::kBool, at::MemoryFormat::Preserve);
  }

  // 复数张量的判断：复数是有限的当且仅当实部和虚部都是有限的
  if (self.is_complex()) {
    // 利用位操作符 __iand__ 对实部和虚部的 isfinite 结果进行按位与操作
    return at::isfinite(at::real(self)).__iand__(at::isfinite(at::imag(self)));
  }

  // 对于浮点类型，调用对应类型的分发宏，并返回判断结果
  return _AT_DISPATCH_INF_TYPES(self.scalar_type(), "isfinite", [&]() {
    // 判断张量的绝对值是否不等于浮点类型的无穷大值
    return (self == self) * (self.abs() != std::numeric_limits<scalar_t>::infinity());
  });
}

void _assert_async_cpu(const Tensor& self) {
  // 断言：张量中至少有一个非零值
  TORCH_CHECK(native::is_nonzero(self), "Expected Tensor with single nonzero value, but got zero");
}

void _assert_async_msg_cpu(const Tensor& self, c10::string_view assert_msg) {
  // 断言：张量中至少有一个非零值，同时可以附加断言信息
  TORCH_CHECK(native::is_nonzero(self), assert_msg != "" ? assert_msg : "Assertion is failed");
}

void _assert_scalar(const Scalar& scalar, c10::string_view assert_msg) {
  // 断言：标量应为布尔类型
  TORCH_SYM_CHECK(scalar.toSymBool(), assert_msg != "" ? assert_msg : "Assertion is failed");
}

Tensor _functional_assert_scalar(const Scalar& scalar, c10::string_view assert_msg, const Tensor& dep_token) {
  // 断言：验证标量，并返回依赖张量的克隆
  _assert_scalar(scalar, assert_msg);
  return dep_token.clone();
}

Tensor _functional_assert_async_msg_cpu(
  const Tensor& self,
  c10::string_view assert_msg,
  const Tensor& dep_token) {
  // 断言：验证张量，并返回依赖张量的克隆，同时可以附加断言信息
  _assert_async_msg_cpu(self, assert_msg);
  return dep_token.clone();
}

void _print(c10::string_view s) {
  // 打印字符串信息
  std::cout << s << "\n";
}

// Sorting-based algorithm for isin(); used when the number of test elements is large.
static void isin_sorting(
    const
    // 1. 如果 assume_unique 为真，则将 elements 和 test_elements 展平为一维数组，并且不调用 unique() 函数。
    //    否则，调用 at::_unique() 函数对 elements 进行唯一化处理，同时记录唯一化后的顺序。
    Tensor elements_flat, test_elements_flat, unique_order;
    if (assume_unique) {
      elements_flat = elements.ravel();
      test_elements_flat = test_elements.ravel();
    } else {
      std::tie(elements_flat, unique_order) = at::_unique(
          elements, /*sorted=*/ false, /*return_inverse=*/ true);
      std::tie(test_elements_flat, std::ignore) = at::_unique(test_elements, /*sorted=*/ false);
    }
    
    // 2. 对所有元素进行稳定排序，保持排序的顺序索引以便后续逆操作。
    //    稳定排序保证了在排序列表中，元素在 test_elements 之前的相对位置不变。
    Tensor all_elements = at::cat({std::move(elements_flat), std::move(test_elements_flat)});
    auto [sorted_elements, sorted_order] = all_elements.sort(
        /*stable=*/ true, /*dim=*/ 0, /*descending=*/ false);
    
    // 3. 创建一个掩码，标识排序列表中相邻重复值的位置。
    //    重复值指的是同时出现在 elements 和 test_elements 中的值。
    Tensor duplicate_mask = at::empty_like(sorted_elements, TensorOptions(ScalarType::Bool));
    Tensor sorted_except_first = sorted_elements.slice(0, 1, at::indexing::None);
    Tensor sorted_except_last = sorted_elements.slice(0, 0, -1);
    duplicate_mask.slice(0, 0, -1).copy_(
      invert ? sorted_except_first.ne(sorted_except_last) : sorted_except_first.eq(sorted_except_last));
    duplicate_mask.index_put_({-1}, invert);
    
    // 4. 将掩码按照预排序的元素顺序重新排序。
    Tensor mask = at::empty_like(duplicate_mask);
    mask.index_copy_(0, sorted_order, duplicate_mask);
    
    // 5. 使用索引将掩码匹配到预唯一元素顺序。
    //    如果 assume_unique 为真，则直接取掩码的前 N 个元素，其中 N 是原始元素的数量。
    if (assume_unique) {
      out.copy_(mask.slice(0, 0, elements.numel()).view_as(out));
    } else {
      out.copy_(at::index(mask, {std::optional<Tensor>(unique_order)}));
    }
}

template<typename... Args>
Device out_device(Args&... inps){
  // 确定输出张量的设备类型
  for (const auto& i : {inps...}) {
    if (i.device() != at::kCPU) {
      return i.device();
    }
  }
  // 如果所有输入张量在CPU上，则输出也在CPU上
  return at::kCPU;
}


Tensor& where_self_out(const Tensor& condition, const Tensor& self, const Tensor& other, Tensor& out) {
  // 确定结果张量的数据类型
  const auto result_type = at::native::result_type(self, other);
  // 检查输出张量的数据类型是否与结果类型匹配
  TORCH_CHECK(out.scalar_type() == result_type, "Expected out type to be ", result_type, " but got ", out.scalar_type());

  // 将输入张量转换为结果类型（如果它们不匹配）
  auto self_ = self.scalar_type() != result_type ? self.to(result_type): self;
  auto other_ = other.scalar_type() != result_type ? other.to(result_type): other;
  auto condition_ = condition;

  // 确定输出张量的设备类型
  auto device = out_device(condition, self_, other_);
  if (device != at::kCPU) { // 允许在非CPU设备上使用CPU标量
    // 如果条件张量在非CPU设备上且是标量，则将其转换为指定设备
    if (condition.device() != device && condition.ndimension() == 0) {
      condition_ = condition.to(device);
    }
    // 如果self_张量在非CPU设备上且是标量，则将其转换为指定设备
    if (self_.device() != device && self_.ndimension() == 0) {
        self_ = self_.to(device);
    }
    // 如果other_张量在非CPU设备上且是标量，则将其转换为指定设备
    if (other_.device() != device && other_.ndimension() == 0) {
        other_ = other_.to(device);
    }
  }
  
  // 如果条件张量的数据类型为Byte，则发出一次警告并转换为布尔类型
  if (condition_.scalar_type() == ScalarType::Byte) {
    TORCH_WARN_ONCE("where received a uint8 condition tensor. This behavior is deprecated and will be removed in a future version of PyTorch. Use a boolean condition instead.");
    condition_ = condition_.to(kBool);
  }
  
  // 检查条件张量的数据类型是否为布尔类型
  TORCH_CHECK(condition_.scalar_type() == kBool, "where expected condition to be a boolean tensor, but got a tensor with dtype ", condition_.scalar_type());

  // 配置TensorIterator，用于处理where操作
  auto iter = at::TensorIteratorConfig()
    .check_all_same_dtype(false)
    .add_output(out)
    .add_const_input(condition_)
    .add_const_input(self_)
    .add_const_input(other_)
    .build();
  
  // 调用where_kernel执行where操作
  where_kernel(iter.device_type(), iter);
  
  // 返回结果张量
  return out;
}


Tensor where(const Tensor& condition, const Tensor& self, const Tensor& other) {
  // 确定输出张量的设备类型
  auto device = out_device(condition, self, other);
  // 确定结果张量的数据类型
  auto result_type = at::native::result_type(self, other);
  // 创建一个空的结果张量，使用指定的数据类型和设备类型
  Tensor ret = at::empty({0}, self.options().dtype(result_type).device(device));
  // 调用where_self_out函数进行where操作
  at::native::where_self_out(condition, self, other, ret);
  // 返回结果张量
  return ret;
}

Tensor where(const Tensor& condition, const Scalar& self, const Tensor& other) {
  // 确定结果张量的数据类型
  auto result_type = at::native::result_type(other, self);
  // 将标量self转换为张量，并指定数据类型
  auto self_converted = at::scalar_tensor(self, other.options().dtype(result_type));
  // 将other张量转换为指定数据类型
  auto other_converted = other.to(result_type);
  // 调用标准库中的where函数进行操作
  return at::where(condition, self_converted, other_converted);
}

Tensor where(const Tensor& condition, const Tensor& self, const Scalar& other) {
  // 确定结果张量的数据类型
  auto result_type = at::native::result_type(self, other);
  // 将标量other转换为张量，并指定数据类型
  auto other_converted = at::scalar_tensor(other, self.options().dtype(result_type));
  // 将self张量转换为指定数据类型
  auto self_converted = self.to(result_type);
  // 调用标准库中的where函数进行操作
  return at::where(condition, self_converted, other_converted);
}
// 返回一个新的 Tensor，根据 condition 选择 self 或 other 中的标量值作为元素填充
Tensor where(const Tensor& condition, const Scalar& self, const Scalar& other) {
  // 确定结果 Tensor 的类型
  auto result_type = at::native::result_type(self, other);
  // 将 other 转换为与 condition 相同类型的 Tensor
  const Tensor& other_t = at::scalar_tensor(other, condition.options().dtype(result_type));
  // 将 self 转换为与 condition 相同类型的 Tensor
  const Tensor& self_t = at::scalar_tensor(self, condition.options().dtype(result_type));
  // 调用 where 函数，根据 condition 选择 self_t 或 other_t 的值
  return at::where(condition, self_t, other_t);
}

// 返回一个 Tensor 的 vector，其中包含满足条件的非零元素的索引
std::vector<Tensor> where(const Tensor& condition) {
  // 调用 nonzero_numpy 函数，返回满足条件的非零元素的索引
  return condition.nonzero_numpy();
}

// 返回一个包含众数和对应索引的 tuple
std::tuple<Tensor, Tensor> mode(const Tensor& self, int64_t dim, bool keepdim) {
  // 创建一个空的 Tensor values，使用 self 的选项（数据类型等）
  Tensor values = at::empty({0}, self.options());
  // 创建一个空的 Tensor indices，使用 self 的选项，并指定数据类型为 kLong
  Tensor indices = at::empty({0}, self.options().dtype(kLong));
  // 调用 mode_out 函数，计算 self 在指定维度上的众数，并返回结果
  return at::native::mode_out(self, dim, keepdim, values, indices);
}

// 计算 self 在指定维度上的众数，并填充给定的 values 和 indices Tensor
std::tuple<Tensor &,Tensor &> mode_out(const Tensor& self, int64_t dim, bool keepdim,
                                       Tensor& values, Tensor& indices) {
  // 检查 self 的设备类型是否为 CPU 或 CUDA
  TORCH_CHECK(self.device().is_cpu() || self.is_cuda(),
              "mode only supports CPU AND CUDA device type, got: ", self.device().type());
  // 检查 self 的布局是否为 Strided
  TORCH_CHECK(self.layout() == Layout::Strided,
              "mode only supports strided layout, got: ", self.layout());
  // 检查 values 和 indices 的设备是否与 self 的设备相同
  TORCH_CHECK(self.device() == values.device(),
              "expected device '", self.device(), "' but got '",
              values.device(), "' for values output");
  TORCH_CHECK(self.device() == indices.device(),
              "expected device '", self.device(), "' but got '",
              indices.device(), "' for indices output");
  // 检查 values 的标量类型是否与 self 的相同
  TORCH_CHECK(self.scalar_type() == values.scalar_type(),
              "expected scalar type '", self.scalar_type(), "' but got '",
              values.scalar_type(), "' for values output");
  // 检查 indices 的标量类型是否为 Long
  TORCH_CHECK(indices.scalar_type() == ScalarType::Long,
              "expected scalar type '", ScalarType::Long, "' but got '",
              indices.scalar_type(), "' for indices output");
  // 确定有效的维度值
  dim = maybe_wrap_dim(dim, self.dim());
  // 如果 self 的元素数为 0，则返回大小为 0 的 Tensor
  if (self.numel() == 0) {
    // 获取零元素 Tensor 的尺寸
    auto sizes = get_zero_numel_tensor_size(self, dim, keepdim, "mode()");
    // 调整 values 和 indices 的大小
    resize_output(values, sizes);
    resize_output(indices, sizes);
    // 返回 values 和 indices
    return std::tie(values, indices);
  }
  // 如果可以直接返回结果而不需要计算
  else if (_dimreduce_return_trivial_no_ident(values, self, dim, keepdim, "mode")) {
    // 确保 values 的维度为 0，并将 indices 设置为 0
    AT_ASSERT(values.dim() == 0);
    indices.resize_({}).fill_(0);
    // 返回 values 和 indices
    return std::forward_as_tuple(values, indices);
  } else {
    // 使用 mode_stub 计算众数，并返回结果
    auto result = [&]() {
      NoNamesGuard guard;
      mode_stub(self.device().type(), values, indices, self, dim, keepdim);
      return std::tuple<Tensor &,Tensor &>{values, indices};
    }();
    // 在减少操作的结果上传播命名
    namedinference::propagate_names_for_reduction(std::get<0>(result), self, dim, keepdim);
    namedinference::propagate_names_for_reduction(std::get<1>(result), self, dim, keepdim);
    // 返回计算结果
    return result;
  }
}

// 通过调用 stub 函数实现的 minmax_out 函数的模板化实现
template <class Stub>
void minmax_out_impl(
    const Tensor& self,
    int64_t dim,
    bool keepdim,
    const Tensor& values,
    const Tensor& indices,
    Stub& stub) {
  // 在这个作用域内不传播命名
  NoNamesGuard guard;
  // 如果 self 的元素数大于 0
  if (self.numel() > 0) {
    # 检查张量是否为标量（只有一个元素）且维度为0
    if (self.numel() == 1 && self.dim() == 0) {
      # 如果是标量，用自身的值填充values张量
      values.fill_(self);
      # 将indices张量填充为0
      indices.fill_(0);
    } else {
      # 如果不是标量，调用stub函数处理
      stub(self.device().type(), values, indices, self, dim, keepdim);
    }
  }
TORCH_IMPL_FUNC(max_out)
(const Tensor& self,                 // 输入参数：操作的张量
 int64_t dim,                        // 输入参数：操作的维度
 bool keepdim,                       // 输入参数：是否保持维度
 const Tensor& values,               // 输入参数：存储最大值的张量
 const Tensor& indices) {            // 输入参数：存储最大值索引的张量
  minmax_out_impl(self, dim, keepdim, values, indices, max_stub);  // 调用最大值操作的实现函数
}

TORCH_IMPL_FUNC(min_out)
(const Tensor& self,                 // 输入参数：操作的张量
 int64_t dim,                        // 输入参数：操作的维度
 bool keepdim,                       // 输入参数：是否保持维度
 const Tensor& values,               // 输入参数：存储最小值的张量
 const Tensor& indices) {            // 输入参数：存储最小值索引的张量
  minmax_out_impl(self, dim, keepdim, values, indices, min_stub);  // 调用最小值操作的实现函数
}

std::tuple<Tensor, Tensor> qmax(const Tensor& self, int64_t dim, bool keepdim) {
  TORCH_CHECK(self.qscheme() == at::kPerTensorAffine, "Max operator for quantized tensors only works for per tensor quantized tensors. "
  "Please open an issue on https://github.com/pytorch/pytorch/issues if you need per channel quantized tensor support.");
  Tensor max_indices = at::empty({0}, self.options().dtype(kLong));  // 创建一个空张量用于存储最大值索引
  Tensor max = at::empty({0}, self.options().dtype(toUnderlying(self.scalar_type())));  // 创建一个空张量用于存储最大值
  at::max_outf(self.int_repr(), dim, keepdim, max, max_indices);  // 调用最大值操作函数
  // TODO: qscheme  // TODO 注释，表示需要后续补充关于量化方案的处理
  return std::tuple<Tensor, Tensor>(
      at::_make_per_tensor_quantized_tensor(max, self.q_scale(), self.q_zero_point()), max_indices);  // 返回量化后的最大值张量和索引
}

std::tuple<Tensor, Tensor> qmin(const Tensor& self, int64_t dim, bool keepdim) {
  TORCH_CHECK(self.qscheme() == at::kPerTensorAffine, "Min operator for quantized tensors only works for per tensor quantized tensors. "
  "Please open an issue on https://github.com/pytorch/pytorch/issues if you need per channel quantized tensor support.");
  Tensor min_indices = at::empty({0}, self.options().dtype(kLong));  // 创建一个空张量用于存储最小值索引
  Tensor min = at::empty({0}, self.options().dtype(toUnderlying(self.scalar_type())));  // 创建一个空张量用于存储最小值
  at::min_outf(self.int_repr(), dim, keepdim, min, min_indices);  // 调用最小值操作函数
  return std::tuple<Tensor, Tensor>(
      at::_make_per_tensor_quantized_tensor(min, self.q_scale(), self.q_zero_point()), min_indices);  // 返回量化后的最小值张量和索引
}

// DEPRECATED: Use at::aminmax instead
std::tuple<Tensor, Tensor> _aminmax(const Tensor& self, int64_t dim, bool keepdim) {
  TORCH_WARN_ONCE("_aminmax is deprecated as of PyTorch 1.11 and will be removed in a future release. Use aminmax instead."
                  " This warning will only appear once per process.");
  return at::aminmax(self, dim, keepdim);  // 返回最大最小值的张量和索引
}

TORCH_IMPL_FUNC(clamp_out)
(
 const Tensor& /*self*/,             // 输入参数：操作的张量（但实际未使用）
 const OptionalScalarRef min,        // 输入参数：可选的最小值标量引用
 const OptionalScalarRef max,        // 输入参数：可选的最大值标量引用
 const Tensor& result) {             // 输入参数：存储结果的张量
  using at::native::detail::ClampLimits;  // 引入命名空间，用于限制范围的处理
  if (min && max) {                   // 如果同时指定了最小值和最大值
    if (min.get().toDouble() != min.get().toDouble() ||  // 如果最小值不是数字（NaN检查）
        max.get().toDouble() != max.get().toDouble()) {  // 如果最大值不是数字（NaN检查）
      at::fill_(const_cast<Tensor&>(result), std::numeric_limits<double>::quiet_NaN());  // 填充结果张量为NaN
    } else {
      clamp_scalar_stub(device_type(), *this, min.get(), max.get());  // 调用标量限制操作的函数
    }
  } else if (max) {                   // 如果只指定了最大值
    clamp_max_scalar_stub(device_type(), *this, max.get());  // 调用最大值限制操作的函数
  } else if (min) {                   // 如果只指定了最小值
    clamp_min_scalar_stub(device_type(), *this, min.get());  // 调用最小值限制操作的函数
  }
}

TORCH_IMPL_FUNC(clamp_Tensor_out)
(const Tensor& self,                 // 输入参数：操作的张量
 const OptionalTensorRef min,         // 输入参数：可选的最小值张量引用
 const OptionalTensorRef max,         // 输入参数：可选的最大值张量引用
 const Tensor&) {                    // 输入参数：存储结果的张量（但实际未使用）
  if (min && max) {                   // 如果同时指定了最小值和最大值
    # 如果 `clamp_stub` 是真值（即非空），调用 `clamp_stub` 函数
    clamp_stub(device_type(), *this);
  # 如果 `clamp_stub` 不是真值，且 `min` 是真值（非空），调用 `maximum_stub` 函数
  } else if (min) {
    maximum_stub(device_type(), *this);
  # 如果 `clamp_stub` 和 `min` 都不是真值，且 `max` 是真值（非空），调用 `minimum_stub` 函数
  } else if (max) {
    minimum_stub(device_type(), *this);
  }
}

// 实现clamp_max_out函数，用于将self张量中的元素限制在max标量指定的最大值以下，并将结果存储在result张量中
TORCH_IMPL_FUNC(clamp_max_out)
(const Tensor& self, const Scalar& max, const Tensor& result) {
  // 检查max是否为NaN，如果是，使用max填充result张量
  if (max.toDouble() != max.toDouble()) {
    // TODO 这种情况不理想，再次构建TI很昂贵，但无法使用fill_stub，因为fill未结构化
    // 这只是一个边缘情况
    at::fill_(const_cast<Tensor&>(result), wrapped_scalar_tensor(max));
  } else {
    // 调用clamp_max_scalar_stub函数，将self张量中的元素限制在max标量指定的最大值以下
    clamp_max_scalar_stub(device_type(), *this, max);
  }
}

// 实现clamp_max_Tensor_out函数，用于将self张量中的元素限制在max张量指定的最大值以下，并将结果存储在result张量中
TORCH_IMPL_FUNC(clamp_max_Tensor_out)
(const Tensor& self, const Tensor& max, const Tensor& result) {
  // 调用minimum_stub函数，计算self张量和max张量中对应元素的最小值，并将结果存储在result张量中
  minimum_stub(device_type(), *this);
}

// 实现clamp_min_out函数，用于将self张量中的元素限制在min标量指定的最小值以上，并将结果存储在result张量中
TORCH_IMPL_FUNC(clamp_min_out)
(const Tensor& self, const Scalar& min, const Tensor& result) {
  // 检查min是否为NaN，如果是，使用min填充result张量
  if (min.toDouble() != min.toDouble()) {
    at::fill_(const_cast<Tensor&>(result), min);
  } else {
    // 调用clamp_min_scalar_stub函数，将self张量中的元素限制在min标量指定的最小值以上
    clamp_min_scalar_stub(device_type(), *this, min);
  }
}

// 实现clamp_min_Tensor_out函数，用于将self张量中的元素限制在min张量指定的最小值以上，并将结果存储在result张量中
TORCH_IMPL_FUNC(clamp_min_Tensor_out)
(const Tensor& self, const Tensor& min, const Tensor& result) {
  // 调用maximum_stub函数，计算self张量和min张量中对应元素的最大值，并将结果存储在result张量中
  maximum_stub(device_type(), *this);
}

// 实现clip_out函数的标量版本，使用at::clamp_outf函数对self张量进行裁剪操作，将结果存储在result张量中
Tensor& clip_out(const Tensor& self, const std::optional<Scalar>& min, const std::optional<Scalar>& max, Tensor& result) {
  return at::clamp_outf(self, min, max, result);
}

// 实现clip_out函数的张量版本，使用at::clamp_outf函数对self张量进行裁剪操作，将结果存储在result张量中
Tensor& clip_out(const Tensor& self, const std::optional<Tensor>& min, const std::optional<Tensor>& max, Tensor& result) {
  return at::clamp_outf(self, min, max, result);
}

// 实现clip函数的标量版本，使用at::clamp函数对self张量进行裁剪操作，并返回结果张量
Tensor clip(const Tensor& self, const std::optional<Scalar>& min, const std::optional<Scalar>& max) {
  return at::clamp(self, min, max);
}

// 实现clip函数的张量版本，使用at::clamp函数对self张量进行裁剪操作，并返回结果张量
Tensor clip(const Tensor& self, const std::optional<Tensor>& min, const std::optional<Tensor>& max) {
  return at::clamp(self, min, max);
}

// 实现clip_函数的标量版本，使用at::clamp_函数对self张量进行原地裁剪操作，并返回修改后的self张量
Tensor& clip_(Tensor& self, const std::optional<Scalar>& min, const std::optional<Scalar>& max) {
  return at::clamp_(self, min, max);
}

// 实现clip_函数的张量版本，使用at::clamp_函数对self张量进行原地裁剪操作，并返回修改后的self张量
Tensor& clip_(Tensor& self, const std::optional<Tensor>& min, const std::optional<Tensor>& max) {
  return at::clamp_(self, min, max);
}

// 返回self张量指定维度dim上的最小值和对应的索引，支持命名维度
std::tuple<Tensor, Tensor> min(const Tensor& self, Dimname dim, bool keepdim) {
  return at::min(self, dimname_to_position(self, dim), keepdim);
}

// 在指定维度dim上计算self张量的最小值，并将结果存储在min和min_indices张量中，支持命名维度
std::tuple<Tensor &,Tensor &> min_out(const Tensor& self, Dimname dim, bool keepdim, Tensor& min, Tensor& min_indices) {
  return at::min_out(min, min_indices, self, dimname_to_position(self, dim), keepdim);
}

// 返回self张量指定维度dim上的最大值和对应的索引，支持命名维度
std::tuple<Tensor, Tensor> max(const Tensor& self, Dimname dim, bool keepdim) {
  return at::max(self, dimname_to_position(self, dim), keepdim);
}

// 在指定维度dim上计算self张量的最大值，并将结果存储在max和max_indices张量中，支持命名维度
std::tuple<Tensor&, Tensor&> max_out(const Tensor& self, Dimname dim, bool keepdim, Tensor& max, Tensor& max_indices) {
  return at::max_out(max, max_indices, self, dimname_to_position(self, dim), keepdim);
}

// 报告不支持命名维度的argsort函数，未实现该函数的具体功能
Tensor argsort(const Tensor& /*self*/, Dimname /*dim*/, bool /*keepdim*/) {
  reportNYIDimnameOverload("argsort");
}

// 返回self张量指定维度dim上的众数和对应的索引，支持命名维度
std::tuple<Tensor, Tensor> mode(const Tensor& self, Dimname dim, bool keepdim) {
  return at::mode(self, dimname_to_position(self, dim), keepdim);
}
std::tuple<Tensor &,Tensor &> mode_out(const Tensor& self, Dimname dim, bool keepdim, Tensor& values, Tensor& indices) {
  // 调用 ATen 的 mode_out 函数，返回值和索引的引用
  return at::mode_out(values, indices, self, dimname_to_position(self, dim), keepdim);
}

TORCH_IMPL_FUNC(isin_Tensor_Tensor_out) (
  const Tensor& elements, const Tensor& test_elements, bool assume_unique, bool invert, const Tensor& out
) {
  if (elements.numel() == 0) {
    // 如果 elements 张量为空，则直接返回
    return;
  }

  // 根据 numpy 实现的启发式方法进行判断
  // 参考 https://github.com/numpy/numpy/blob/fb215c76967739268de71aa4bda55dd1b062bc2e/numpy/lib/arraysetops.py#L575
  if (test_elements.numel() < static_cast<int64_t>(
        10.0f * std::pow(static_cast<double>(elements.numel()), 0.145))) {
    // 如果 test_elements 的元素数量小于启发式计算的阈值，使用默认的 isin 算法
    out.fill_(invert);
    isin_default_stub(elements.device().type(), elements, test_elements, invert, out);
  } else {
    // 否则，使用排序后的 isin 算法
    isin_sorting(elements, test_elements, assume_unique, invert, out);
  }
}

TORCH_IMPL_FUNC(isin_Tensor_Scalar_out) (
  const Tensor& elements, const c10::Scalar& test_elements, bool assume_unique, bool invert, const Tensor& out
) {
  // 根据标量 test_elements 的值调度到 eq_out 或者 ne_out 函数
  if (invert) {
    at::ne_out(const_cast<Tensor&>(out), elements, test_elements);
  } else {
    at::eq_out(const_cast<Tensor&>(out), elements, test_elements);
  }
}

TORCH_IMPL_FUNC(isin_Scalar_Tensor_out) (
  const c10::Scalar& elements, const Tensor& test_elements, bool assume_unique, bool invert, const Tensor& out
) {
  // 根据标量 elements 的值调度到 isin_out 函数
  at::isin_out(const_cast<Tensor&>(out), wrapped_scalar_tensor(elements, test_elements.device()),
    test_elements, assume_unique, invert);
}

TORCH_IMPL_FUNC(isposinf_out) (const Tensor& self, const Tensor& result) {
  // 如果 self 的标量类型是整数类型（包括布尔类型），则结果张量 result 填充为 false
  if (c10::isIntegralType(self.scalar_type(), /*includeBool=*/true)) {
    result.fill_(false);
  } else {
    // 否则调用 isposinf_stub 处理具体实现
    isposinf_stub(device_type(), *this);
  }
}

TORCH_IMPL_FUNC(isneginf_out) (const Tensor& self, const Tensor& result) {
  // 如果 self 的标量类型是整数类型（包括布尔类型），则结果张量 result 填充为 false
  if (c10::isIntegralType(self.scalar_type(), /*includeBool=*/true)) {
    result.fill_(false);
  } else {
    // 否则调用 isneginf_stub 处理具体实现
    isneginf_stub(device_type(), *this);
  }
}

} // namespace at::native


这些注释描述了每个函数的目的和每行代码的作用，帮助理解它们在实现中的具体功能和调用逻辑。
```