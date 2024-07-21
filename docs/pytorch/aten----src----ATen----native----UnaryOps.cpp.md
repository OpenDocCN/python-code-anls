# `.\pytorch\aten\src\ATen\native\UnaryOps.cpp`

```py
// 定义 TORCH_ASSERT_ONLY_METHOD_OPERATORS 宏，用于只包含方法操作符
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS

// 包含 ATen 库中的 Tensor 类定义
#include <ATen/core/Tensor.h>

// 包含 ATen 库中的扩展工具函数
#include <ATen/ExpandUtils.h>

// 包含 ATen 库中的内存重叠检查工具
#include <ATen/MemoryOverlap.h>

// 包含 ATen 库中的命名张量工具函数
#include <ATen/NamedTensorUtils.h>

// 包含 ATen 库中的并行计算工具
#include <ATen/Parallel.h>

// 包含 ATen 库中的标量操作函数
#include <ATen/ScalarOps.h>

// 包含 ATen 库中的张量迭代器
#include <ATen/TensorIterator.h>

// 包含 ATen 库中的张量操作函数
#include <ATen/TensorOperators.h>

// 包含 ATen 库中的维度包装工具函数
#include <ATen/WrapDimUtils.h>

// 包含 ATen 库中的 Resize 操作的本地实现
#include <ATen/native/Resize.h>

// 包含 ATen 库中的一元操作的本地实现
#include <ATen/native/UnaryOps.h>

// 包含 ATen 库中的复数操作辅助函数
#include <ATen/native/ComplexHelper.h>

// 包含 C10 库中的数学常量定义
#include <c10/util/MathConstants.h>

// 根据条件排除一组特定的 ATen 操作函数头文件
#ifndef AT_PER_OPERATOR_HEADERS
    // 包含 ATen 库中的一般操作函数和本地函数
    #include <ATen/Functions.h>
    #include <ATen/NativeFunctions.h>
#else
    // 包含 ATen 库中的具体操作的本地实现头文件
    #include <ATen/ops/_conj_native.h>
    #include <ATen/ops/_conj_physical.h>
    #include <ATen/ops/_conj_physical_native.h>
    #include <ATen/ops/_neg_view_native.h>
    #include <ATen/ops/abs.h>
    #include <ATen/ops/abs_native.h>
    #include <ATen/ops/absolute_native.h>
    #include <ATen/ops/acos.h>
    #include <ATen/ops/acos_native.h>
    #include <ATen/ops/acosh.h>
    #include <ATen/ops/acosh_native.h>
    #include <ATen/ops/angle.h>
    #include <ATen/ops/angle_native.h>
    #include <ATen/ops/arange_native.h>
    #include <ATen/ops/arccos_native.h>
    #include <ATen/ops/arccosh_native.h>
    #include <ATen/ops/arcsin_native.h>
    #include <ATen/ops/arcsinh_native.h>
    #include <ATen/ops/arctan_native.h>
    #include <ATen/ops/arctanh_native.h>
    #include <ATen/ops/asin.h>
    #include <ATen/ops/asin_native.h>
    #include <ATen/ops/asinh.h>
    #include <ATen/ops/asinh_native.h>
    #include <ATen/ops/atan.h>
    #include <ATen/ops/atan_native.h>
    #include <ATen/ops/atanh.h>
    #include <ATen/ops/atanh_native.h>
    #include <ATen/ops/bitwise_not_native.h>
    #include <ATen/ops/can_cast.h>
    #include <ATen/ops/ceil_native.h>
    #include <ATen/ops/conj_native.h>
    #include <ATen/ops/conj_physical.h>
    #include <ATen/ops/conj_physical_native.h>
    #include <ATen/ops/cos_native.h>
    #include <ATen/ops/cosh_native.h>
    #include <ATen/ops/deg2rad.h>
    #include <ATen/ops/deg2rad_native.h>
    #include <ATen/ops/digamma.h>
    #include <ATen/ops/digamma_native.h>
    #include <ATen/ops/empty.h>
    #include <ATen/ops/empty_like.h>
    #include <ATen/ops/erf.h>
    #include <ATen/ops/erf_native.h>
    #include <ATen/ops/erfc.h>
    #include <ATen/ops/erfc_native.h>
    #include <ATen/ops/erfinv.h>
    #include <ATen/ops/erfinv_native.h>
    #include <ATen/ops/exp2.h>
    #include <ATen/ops/exp2_native.h>
    #include <ATen/ops/exp_native.h>
    #include <ATen/ops/expm1.h>
    #include <ATen/ops/expm1_native.h>
    #include <ATen/ops/fix_native.h>
    #include <ATen/ops/floor_native.h>
    #include <ATen/ops/frac_native.h>
    #include <ATen/ops/frexp.h>
    #include <ATen/ops/frexp_native.h>
    #include <ATen/ops/i0.h>
    #include <ATen/ops/i0_native.h>
    #include <ATen/ops/imag_native.h>
    #include <ATen/ops/lgamma.h>
    #include <ATen/ops/lgamma_native.h>
    #include <ATen/ops/log10_native.h>
    #include <ATen/ops/log1p.h>
    #include <ATen/ops/log1p_native.h>
    #include <ATen/ops/log2_native.h>
    #include <ATen/ops/log_native.h>
    #include <ATen/ops/logical_not.h>
    #include <ATen/ops/logical_not_native.h>
    #include <ATen/ops/logit.h>
    #include <ATen/ops/logit_native.h>
    #include <ATen/ops/mul.h>
#endif
// 包含头文件，以便使用 ATen 库中的 mvlgamma 函数
#include <ATen/ops/mvlgamma.h>
// 包含头文件，以便使用 ATen 库中的 mvlgamma_native 函数
#include <ATen/ops/mvlgamma_native.h>
// 包含头文件，以便使用 ATen 库中的 nan_to_num 函数
#include <ATen/ops/nan_to_num.h>
// 包含头文件，以便使用 ATen 库中的 nan_to_num_native 函数
#include <ATen/ops/nan_to_num_native.h>
// 包含头文件，以便使用 ATen 库中的 neg 函数
#include <ATen/ops/neg.h>
// 包含头文件，以便使用 ATen 库中的 neg_native 函数
#include <ATen/ops/neg_native.h>
// 包含头文件，以便使用 ATen 库中的 negative_native 函数
#include <ATen/ops/negative_native.h>
// 包含头文件，以便使用 ATen 库中的 polygamma 函数
#include <ATen/ops/polygamma.h>
// 包含头文件，以便使用 ATen 库中的 polygamma_native 函数
#include <ATen/ops/polygamma_native.h>
// 包含头文件，以便使用 ATen 库中的 positive_native 函数
#include <ATen/ops/positive_native.h>
// 包含头文件，以便使用 ATen 库中的 pow 函数
#include <ATen/ops/pow.h>
// 包含头文件，以便使用 ATen 库中的 rad2deg 函数
#include <ATen/ops/rad2deg.h>
// 包含头文件，以便使用 ATen 库中的 rad2deg_native 函数
#include <ATen/ops/rad2deg_native.h>
// 包含头文件，以便使用 ATen 库中的 real 函数
#include <ATen/ops/real.h>
// 包含头文件，以便使用 ATen 库中的 real_native 函数
#include <ATen/ops/real_native.h>
// 包含头文件，以便使用 ATen 库中的 reciprocal_native 函数
#include <ATen/ops/reciprocal_native.h>
// 包含头文件，以便使用 ATen 库中的 resolve_conj_native 函数
#include <ATen/ops/resolve_conj_native.h>
// 包含头文件，以便使用 ATen 库中的 resolve_neg_native 函数
#include <ATen/ops/resolve_neg_native.h>
// 包含头文件，以便使用 ATen 库中的 round 函数
#include <ATen/ops/round.h>
// 包含头文件，以便使用 ATen 库中的 round_native 函数
#include <ATen/ops/round_native.h>
// 包含头文件，以便使用 ATen 库中的 rsqrt_native 函数
#include <ATen/ops/rsqrt_native.h>
// 包含头文件，以便使用 ATen 库中的 select 函数
#include <ATen/ops/select.h>
// 包含头文件，以便使用 ATen 库中的 sgn_native 函数
#include <ATen/ops/sgn_native.h>
// 包含头文件，以便使用 ATen 库中的 sigmoid 函数
#include <ATen/ops/sigmoid.h>
// 包含头文件，以便使用 ATen 库中的 sigmoid_native 函数
#include <ATen/ops/sigmoid_native.h>
// 包含头文件，以便使用 ATen 库中的 sign_native 函数
#include <ATen/ops/sign_native.h>
// 包含头文件，以便使用 ATen 库中的 signbit_native 函数
#include <ATen/ops/signbit_native.h>
// 包含头文件，以便使用 ATen 库中的 sin_native 函数
#include <ATen/ops/sin_native.h>
// 包含头文件，以便使用 ATen 库中的 sinc 函数
#include <ATen/ops/sinc.h>
// 包含头文件，以便使用 ATen 库中的 sinc_native 函数
#include <ATen/ops/sinc_native.h>
// 包含头文件，以便使用 ATen 库中的 sinh_native 函数
#include <ATen/ops/sinh_native.h>
// 包含头文件，以便使用 ATen 库中的 special_airy_ai_native 函数
#include <ATen/ops/special_airy_ai_native.h>
// 包含头文件，以便使用 ATen 库中的 special_bessel_j0_native 函数
#include <ATen/ops/special_bessel_j0_native.h>
// 包含头文件，以便使用 ATen 库中的 special_bessel_j1_native 函数
#include <ATen/ops/special_bessel_j1_native.h>
// 包含头文件，以便使用 ATen 库中的 special_bessel_y0_native 函数
#include <ATen/ops/special_bessel_y0_native.h>
// 包含头文件，以便使用 ATen 库中的 special_bessel_y1_native 函数
#include <ATen/ops/special_bessel_y1_native.h>
// 包含头文件，以便使用 ATen 库中的 special_digamma_native 函数
#include <ATen/ops/special_digamma_native.h>
// 包含头文件，以便使用 ATen 库中的 special_entr_native 函数
#include <ATen/ops/special_entr_native.h>
// 包含头文件，以便使用 ATen 库中的 special_erf_native 函数
#include <ATen/ops/special_erf_native.h>
// 包含头文件，以便使用 ATen 库中的 special_erfc_native 函数
#include <ATen/ops/special_erfc_native.h>
// 包含头文件，以便使用 ATen 库中的 special_erfcx_native 函数
#include <ATen/ops/special_erfcx_native.h>
// 包含头文件，以便使用 ATen 库中的 special_erfinv_native 函数
#include <ATen/ops/special_erfinv_native.h>
// 包含头文件，以便使用 ATen 库中的 special_exp2_native 函数
#include <ATen/ops/special_exp2_native.h>
// 包含头文件，以便使用 ATen 库中的 special_expit_native 函数
#include <ATen/ops/special_expit_native.h>
// 包含头文件，以便使用 ATen 库中的 special_expm1_native 函数
#include <ATen/ops/special_expm1_native.h>
// 包含头文件，以便使用 ATen 库中的 special_gammaln_native 函数
#include <ATen/ops/special_gammaln_native.h>
// 包含头文件，以便使用 ATen 库中的 special_i0_native 函数
#include <ATen/ops/special_i0_native.h>
// 包含头文件，以便使用 ATen 库中的 special_i0e_native 函数
#include <ATen/ops/special_i0e_native.h>
// 包含头文件，以便使用 ATen 库中的 special_i1_native 函数
#include <ATen/ops/special_i1_native.h>
// 包含头文件，以便使用 ATen 库中的 special_i1e_native 函数
#include <ATen/ops/special_i1e_native.h>
// 包含头文件，以便使用 ATen 库中的 special_log1p_native 函数
#include <ATen/ops/special_log1p_native.h>
// 包含头文件，以便使用 ATen 库中的 special_log_ndtr_native 函数
#include <ATen/ops/special_log_ndtr_native.h>
// 包含头文件，以便使用 ATen 库中的 special_logit_native 函数
#include <ATen/ops/special_logit_native.h>
// 包含头文件，以便使用 ATen 库中的 special_modified_bessel_i0_native 函数
#include <ATen/ops/special_modified_bessel_i0_native.h>
// 包含头文件，以便使用 ATen 库中的 special_modified_bessel_i1_native 函数
#include <ATen/ops/special_modified_bessel_i1_native.h>
// 包含头文件，以便使用 ATen 库中的 special_modified_bessel_k0_native 函数
#include <ATen/ops/special_modified_bessel_k0_native.h>
// 包含头文件，以便使用 ATen 库中的 special_modified_bessel_k1_native
// 定义宏 CREATE_UNARY_FLOAT_META_FUNC，用于生成一元浮点操作的元数据函数
#define CREATE_UNARY_FLOAT_META_FUNC(func)                  \
  // 使用 TORCH_META_FUNC 宏定义元数据函数 func，接受一个 Tensor 类型参数 self
  TORCH_META_FUNC(func) (const Tensor& self) {        \
    // 调用 build_borrowing_unary_float_op 函数，为可能的输出构建一元浮点操作，使用 self 引用
    build_borrowing_unary_float_op(maybe_get_output(), self);   \
  }

// 使用宏 CREATE_UNARY_FLOAT_META_FUNC 生成各种一元浮点操作的元数据函数
CREATE_UNARY_FLOAT_META_FUNC(acos)
CREATE_UNARY_FLOAT_META_FUNC(acosh)
CREATE_UNARY_FLOAT_META_FUNC(asin)
CREATE_UNARY_FLOAT_META_FUNC(asinh)
CREATE_UNARY_FLOAT_META_FUNC(atan)
CREATE_UNARY_FLOAT_META_FUNC(atanh)
CREATE_UNARY_FLOAT_META_FUNC(cos)
CREATE_UNARY_FLOAT_META_FUNC(cosh)
CREATE_UNARY_FLOAT_META_FUNC(digamma)
CREATE_UNARY_FLOAT_META_FUNC(erf)
CREATE_UNARY_FLOAT_META_FUNC(erfc)
CREATE_UNARY_FLOAT_META_FUNC(erfinv)
CREATE_UNARY_FLOAT_META_FUNC(exp)
CREATE_UNARY_FLOAT_META_FUNC(exp2)
CREATE_UNARY_FLOAT_META_FUNC(expm1)
CREATE_UNARY_FLOAT_META_FUNC(i0)
CREATE_UNARY_FLOAT_META_FUNC(lgamma)
CREATE_UNARY_FLOAT_META_FUNC(log)
CREATE_UNARY_FLOAT_META_FUNC(log10)
CREATE_UNARY_FLOAT_META_FUNC(log1p)
CREATE_UNARY_FLOAT_META_FUNC(log2)
CREATE_UNARY_FLOAT_META_FUNC(reciprocal)
CREATE_UNARY_FLOAT_META_FUNC(rsqrt)
CREATE_UNARY_FLOAT_META_FUNC(sigmoid)
CREATE_UNARY_FLOAT_META_FUNC(sin)
CREATE_UNARY_FLOAT_META_FUNC(sinc)
CREATE_UNARY_FLOAT_META_FUNC(sinh)
CREATE_UNARY_FLOAT_META_FUNC(special_entr)
CREATE_UNARY_FLOAT_META_FUNC(special_erfcx)
CREATE_UNARY_FLOAT_META_FUNC(special_i0e)
CREATE_UNARY_FLOAT_META_FUNC(special_i1)
CREATE_UNARY_FLOAT_META_FUNC(special_i1e)
CREATE_UNARY_FLOAT_META_FUNC(special_ndtri)
CREATE_UNARY_FLOAT_META_FUNC(special_log_ndtr)
CREATE_UNARY_FLOAT_META_FUNC(sqrt)
CREATE_UNARY_FLOAT_META_FUNC(tan)
CREATE_UNARY_FLOAT_META_FUNC(tanh)
CREATE_UNARY_FLOAT_META_FUNC(special_airy_ai)
CREATE_UNARY_FLOAT_META_FUNC(special_bessel_j0)
CREATE_UNARY_FLOAT_META_FUNC(special_bessel_j1)
CREATE_UNARY_FLOAT_META_FUNC(special_bessel_y0)
CREATE_UNARY_FLOAT_META_FUNC(special_bessel_y1)
CREATE_UNARY_FLOAT_META_FUNC(special_modified_bessel_i0)
CREATE_UNARY_FLOAT_META_FUNC(special_modified_bessel_i1)
CREATE_UNARY_FLOAT_META_FUNC(special_modified_bessel_k0)
CREATE_UNARY_FLOAT_META_FUNC(special_modified_bessel_k1)
CREATE_UNARY_FLOAT_META_FUNC(special_scaled_modified_bessel_k0)
CREATE_UNARY_FLOAT_META_FUNC(special_scaled_modified_bessel_k1)
CREATE_UNARY_FLOAT_META_FUNC(special_spherical_bessel_j0)

// 定义 TORCH_META_FUNC 宏生成 polygamma 函数的元数据函数
TORCH_META_FUNC(polygamma)(int64_t n, const Tensor& self) {
  // 使用 TORCH_CHECK 确保 n 大于等于 0，否则抛出异常
  TORCH_CHECK(n >= 0, "polygamma(n, x) does not support negative n.");
  // 调用 build_borrowing_unary_float_op 函数，为可能的输出构建一元浮点操作，使用 self 引用
  build_borrowing_unary_float_op(maybe_get_output(), self);
}

// 定义宏 CREATE_UNARY_META_FUNC，用于生成保持数据类型的一元操作的元数据函数
#define CREATE_UNARY_META_FUNC(func)                  \
  // 使用 TORCH_META_FUNC 宏定义元数据函数 func，接受一个 Tensor 类型参数 self
  TORCH_META_FUNC(func) (const Tensor& self) {        \
    // 调用 build_borrowing_unary_op 函数，为可能的输出构建一元操作，使用 self 引用
    build_borrowing_unary_op(maybe_get_output(), self);   \
  }

// 使用宏 CREATE_UNARY_META_FUNC 生成各种一元操作的元数据函数
CREATE_UNARY_META_FUNC(bitwise_not)
CREATE_UNARY_META_FUNC(frac)
CREATE_UNARY_META_FUNC(round)
CREATE_UNARY_META_FUNC(sgn)
TORCH_META_FUNC2(round, decimals)(const Tensor& self, int64_t decimals){
  build_unary_op(maybe_get_output(), self);
}


TORCH_META_FUNC(neg)(const Tensor& self) {
  TORCH_CHECK(self.scalar_type() != kBool,
              "Negation, the `-` operator, on a bool tensor is not supported. "
              "If you are trying to invert a mask, use the `~` or `logical_not()` operator instead.");
  // 构建一个借用操作的一元运算符，用于对输入张量进行否定操作
  build_borrowing_unary_op(maybe_get_output(), self);
}


TORCH_META_FUNC(trunc) (const Tensor& self) {
  // 注意：这与 NumPy 的行为保持一致
  TORCH_CHECK(!self.is_complex(),
    "trunc is not supported for complex inputs");
  // 构建一个借用操作的一元运算符，用于对输入张量进行截断操作
  build_borrowing_unary_op(maybe_get_output(), self);
}


TORCH_META_FUNC(floor) (const Tensor& self) {
  // 注意：这与 NumPy 的行为保持一致
  TORCH_CHECK(!self.is_complex(),
    "floor is not supported for complex inputs");
  // 构建一个借用操作的一元运算符，用于对输入张量进行向下取整操作
  build_borrowing_unary_op(maybe_get_output(), self);
}


TORCH_META_FUNC(sign) (const Tensor& self) {
  TORCH_CHECK(!self.is_complex(),
              "Unlike NumPy, torch.sign is not intended to support complex numbers. Please use torch.sgn instead.");
  // 构建一个借用操作的一元运算符，用于对输入张量进行符号函数操作
  build_borrowing_unary_op(maybe_get_output(), self);
}


TORCH_META_FUNC(signbit) (const Tensor& self) {
  TORCH_CHECK(!self.is_complex(), "signbit is not implemented for complex tensors.");
  TORCH_CHECK(maybe_get_output().defined() ? maybe_get_output().dtype() == at::kBool : true,
              "signbit does not support non-boolean outputs.");
  // 构建一个强制布尔类型输出的借用操作的一元运算符，用于计算输入张量每个元素的符号位
  build_borrowing_unary_force_boolean_op(maybe_get_output(), self);
}


TORCH_META_FUNC(ceil) (const Tensor& self) {
  // 注意：这与 NumPy 的行为保持一致
  TORCH_CHECK(!self.is_complex(),
    "ceil is not supported for complex inputs");
  // 构建一个借用操作的一元运算符，用于对输入张量进行向上取整操作
  build_borrowing_unary_op(maybe_get_output(), self);
}
    result.copy_(self);                                                 \

复制当前对象 `self` 的数据到 `result` 对象中。


  } else {                                                              \

如果上述条件不满足，则执行以下代码块。


    func_stub(device_type(), *this);                                    \

调用 `func_stub` 函数，传递当前设备类型和当前对象 `this` 的参数。


  }                                                                     \

结束代码块。
# 创建一元操作的 Torch 实现函数，无输出，整数参数，无操作函数
CREATE_UNARY_TORCH_IMPL_INTEGER_NO_OP_FUNC(ceil_out, ceil_stub)
CREATE_UNARY_TORCH_IMPL_INTEGER_NO_OP_FUNC(floor_out, floor_stub)
CREATE_UNARY_TORCH_IMPL_INTEGER_NO_OP_FUNC(round_out, round_stub)
CREATE_UNARY_TORCH_IMPL_INTEGER_NO_OP_FUNC(trunc_out, trunc_stub)

# 创建一元操作的 Torch 实现函数，输出为 acos 的结果，使用 acos_stub 函数
CREATE_UNARY_TORCH_IMPL_FUNC(acos_out, acos_stub)
CREATE_UNARY_TORCH_IMPL_FUNC(acosh_out, acosh_stub)
CREATE_UNARY_TORCH_IMPL_FUNC(asin_out, asin_stub)
CREATE_UNARY_TORCH_IMPL_FUNC(asinh_out, asinh_stub)
CREATE_UNARY_TORCH_IMPL_FUNC(atan_out, atan_stub)
CREATE_UNARY_TORCH_IMPL_FUNC(atanh_out, atanh_stub)
CREATE_UNARY_TORCH_IMPL_FUNC(bitwise_not_out, bitwise_not_stub)
CREATE_UNARY_TORCH_IMPL_FUNC(cos_out, cos_stub)
CREATE_UNARY_TORCH_IMPL_FUNC(cosh_out, cosh_stub)
CREATE_UNARY_TORCH_IMPL_FUNC(digamma_out, digamma_stub)
CREATE_UNARY_TORCH_IMPL_FUNC(erf_out, erf_stub)
CREATE_UNARY_TORCH_IMPL_FUNC(erfc_out, erfc_stub)
CREATE_UNARY_TORCH_IMPL_FUNC(erfinv_out, erfinv_stub)
CREATE_UNARY_TORCH_IMPL_FUNC(exp_out, exp_stub)
CREATE_UNARY_TORCH_IMPL_FUNC(exp2_out, exp2_stub)
CREATE_UNARY_TORCH_IMPL_FUNC(expm1_out, expm1_stub)
CREATE_UNARY_TORCH_IMPL_FUNC(frac_out, frac_stub)
CREATE_UNARY_TORCH_IMPL_FUNC(i0_out, i0_stub)
CREATE_UNARY_TORCH_IMPL_FUNC(lgamma_out, lgamma_stub)
CREATE_UNARY_TORCH_IMPL_FUNC(log_out, log_stub)
CREATE_UNARY_TORCH_IMPL_FUNC(log10_out, log10_stub)
CREATE_UNARY_TORCH_IMPL_FUNC(log1p_out, log1p_stub)
CREATE_UNARY_TORCH_IMPL_FUNC(log2_out, log2_stub)
CREATE_UNARY_TORCH_IMPL_FUNC(neg_out, neg_stub)
CREATE_UNARY_TORCH_IMPL_FUNC(reciprocal_out, reciprocal_stub)
CREATE_UNARY_TORCH_IMPL_FUNC(rsqrt_out, rsqrt_stub)
CREATE_UNARY_TORCH_IMPL_FUNC(sigmoid_out, sigmoid_stub)
CREATE_UNARY_TORCH_IMPL_FUNC(sign_out, sign_stub)
CREATE_UNARY_TORCH_IMPL_FUNC(sin_out, sin_stub)
CREATE_UNARY_TORCH_IMPL_FUNC(sinc_out, sinc_stub)
CREATE_UNARY_TORCH_IMPL_FUNC(sinh_out, sinh_stub)
CREATE_UNARY_TORCH_IMPL_FUNC(special_entr_out, special_entr_stub)
CREATE_UNARY_TORCH_IMPL_FUNC(special_erfcx_out, special_erfcx_stub)
CREATE_UNARY_TORCH_IMPL_FUNC(special_i0e_out, special_i0e_stub)
CREATE_UNARY_TORCH_IMPL_FUNC(special_i1e_out, special_i1e_stub)
CREATE_UNARY_TORCH_IMPL_FUNC(special_i1_out, special_i1_stub)
CREATE_UNARY_TORCH_IMPL_FUNC(special_ndtri_out, special_ndtri_stub)
CREATE_UNARY_TORCH_IMPL_FUNC(special_log_ndtr_out, special_log_ndtr_stub)
CREATE_UNARY_TORCH_IMPL_FUNC(sqrt_out, sqrt_stub)
CREATE_UNARY_TORCH_IMPL_FUNC(tan_out, tan_stub)
CREATE_UNARY_TORCH_IMPL_FUNC(tanh_out, tanh_stub)
CREATE_UNARY_TORCH_IMPL_FUNC(special_airy_ai_out, special_airy_ai_stub)
CREATE_UNARY_TORCH_IMPL_FUNC(special_bessel_j0_out, special_bessel_j0_stub)
CREATE_UNARY_TORCH_IMPL_FUNC(special_bessel_j1_out, special_bessel_j1_stub)
CREATE_UNARY_TORCH_IMPL_FUNC(special_bessel_y0_out, special_bessel_y0_stub)
CREATE_UNARY_TORCH_IMPL_FUNC(special_bessel_y1_out, special_bessel_y1_stub)
CREATE_UNARY_TORCH_IMPL_FUNC(special_modified_bessel_i0_out, special_modified_bessel_i0_stub)
// 创建一元操作的 Torch 实现函数，使用指定的函数和输出结果的存根
CREATE_UNARY_TORCH_IMPL_FUNC(special_modified_bessel_i1_out, special_modified_bessel_i1_stub)

// 创建一元操作的 Torch 实现函数，使用指定的函数和输出结果的存根
CREATE_UNARY_TORCH_IMPL_FUNC(special_modified_bessel_k0_out, special_modified_bessel_k0_stub)

// 创建一元操作的 Torch 实现函数，使用指定的函数和输出结果的存根
CREATE_UNARY_TORCH_IMPL_FUNC(special_modified_bessel_k1_out, special_modified_bessel_k1_stub)

// 创建一元操作的 Torch 实现函数，使用指定的函数和输出结果的存根
CREATE_UNARY_TORCH_IMPL_FUNC(special_scaled_modified_bessel_k0_out, special_scaled_modified_bessel_k0_stub)

// 创建一元操作的 Torch 实现函数，使用指定的函数和输出结果的存根
CREATE_UNARY_TORCH_IMPL_FUNC(special_scaled_modified_bessel_k1_out, special_scaled_modified_bessel_k1_stub)

// 创建一元操作的 Torch 实现函数，使用指定的函数和输出结果的存根
CREATE_UNARY_TORCH_IMPL_FUNC(special_spherical_bessel_j0_out, special_spherical_bessel_j0_stub)

// 定义 Torch 实现函数 round_decimals_out，用于对自身张量进行小数位数四舍五入的操作
// 根据 decimals 参数决定调用 round_decimals_stub 或 round_stub
TORCH_IMPL_FUNC(round_decimals_out)
(const Tensor& self, int64_t decimals, const Tensor& result) {
  if (decimals != 0) {
    round_decimals_stub(device_type(), *this, decimals);
  } else {
    round_stub(device_type(), *this);
  }
}

// 定义 Torch 实现函数 polygamma_out，用于计算自身张量的 n 阶多 Gamma 函数
TORCH_IMPL_FUNC(polygamma_out)
(int64_t n, const Tensor& self, const Tensor& result) {
  polygamma_stub(device_type(), *this, n);
}

// 定义 Torch 实现函数 signbit_out，用于检查自身张量的符号位
// 如果张量是布尔类型，则结果张量 result 填充为 false
// 否则调用 signbit_stub 进行计算
TORCH_IMPL_FUNC(signbit_out) (const Tensor& self, const Tensor& result) {
  if (self.dtype() == at::kBool) {
    result.fill_(false);
  } else {
    signbit_stub(device_type(), *this);
  }
}

// 由于 polygamma_ 的签名与其输出和功能变体不同，因此我们显式定义它，而不使用结构化内核。
// 在此函数中，调用 polygamma_out，使用自身张量和参数 n，然后返回修改后的自身张量
Tensor& polygamma_(Tensor& self, int64_t n) {
  return at::polygamma_out(self, n, self);
}

// 定义模板函数 unary_op_impl_out，用于执行一元操作并将结果写入结果张量
template <typename Stub>
static inline Tensor& unary_op_impl_out(Tensor& result, const Tensor& self, Stub& stub) {
  auto iter = TensorIterator::unary_op(result, self);
  stub(iter.device_type(), iter);
  return result;
}

// 定义模板函数 unary_op_impl_float_out，用于执行浮点数类型的一元操作并将结果写入结果张量
template <typename Stub, typename ...Args>
static inline Tensor& unary_op_impl_float_out(Tensor& result, const Tensor& self, Stub& stub, Args... args) {
  auto iter = TensorIterator::unary_float_op(result, self);
  stub(iter.device_type(), iter, args...);
  return result;
}

// 定义模板函数 unary_op_impl_float，用于执行浮点数类型的一元操作并返回结果张量
template <typename Stub, typename ...Args>
static inline Tensor unary_op_impl_float(const Tensor& self, Stub& stub, Args... args) {
  Tensor result;
  auto iter = TensorIterator::unary_float_op(result, self);
  stub(iter.device_type(), iter, args...);
  return iter.output();
}

// 定义模板函数 unary_op_impl_with_complex_to_float_out，根据 promotes_integer_to_float 参数执行一元操作
// 如果 promotes_integer_to_float 为 true，则复制操作结果到浮点数类型的结果张量
template <typename Stub>
static inline Tensor& unary_op_impl_with_complex_to_float_out(Tensor& result, const Tensor& self, Stub& stub, bool promotes_integer_to_float) {
    // 如果输入张量是复数并且输出张量不是复数
    if (self.is_complex() && !result.is_complex()) {
      // 检查相应的浮点类型是否可以转换为所需的数据类型
      const auto float_type = c10::toRealValueType(self.scalar_type());
      TORCH_CHECK(canCast(float_type, result.scalar_type()),
            "result type ", float_type, " can't be cast to the desired output type ",
            result.scalar_type());

      // 运行复数到复数的函数，因为TensorIterator期望的是这种类型
      Tensor complex_result = at::empty({0}, self.options());
      auto iter = TensorIterator::unary_op(complex_result, self);
      stub(iter.device_type(), iter);

      // 将复数结果复制到实际结果并返回
      at::native::resize_output(result, complex_result.sizes());
      result.copy_(at::real(complex_result));
      return result;
    }

    // 如果需要将整数提升为浮点数，则调用浮点数输出的实现函数
    if (promotes_integer_to_float) {
      return unary_op_impl_float_out(result, self, stub);
    }

    // 否则调用普通输出的实现函数
    return unary_op_impl_out(result, self, stub);
}

// out_impl passed into unary_op_impl and unary_op_impl_  must go through at:: device dispatch
// otherwise it won't dispatch to out-of-source devices like XLA.
// For example it must be at::bitwise_not_out instead of bitwise_not_out(which is at::native!).
// 将 out_impl 传递给 unary_op_impl 和 unary_op_impl_，必须通过 at:: device dispatch
// 否则它不会分派到像 XLA 这样的外部设备。
// 例如，必须是 at::bitwise_not_out 而不是 bitwise_not_out（这是 at::native!）。

template <typename OutImpl>
static inline Tensor unary_op_impl(const Tensor& self, OutImpl& out_impl) {
  // 创建一个空的张量 result，使用 self 的选项
  Tensor result = at::empty({0}, self.options());
  // 调用 out_impl 处理 result 和 self，返回结果张量
  return out_impl(result, self);
}

// An alternate version of unary_op_impl that follows the same pattern
// for non-complex inputs, but returns a floating point tensor
// for complex inputs by default.
// unary_op_impl 的另一版本，对于非复数输入遵循相同的模式，
// 但默认情况下为复数输入返回浮点数张量。
template <typename OutImpl>
static inline Tensor unary_op_impl_with_complex_to_float(const Tensor& self, OutImpl& out_impl) {
  if (self.is_complex()) {
    // 如果 self 是复数，使用其实数值类型创建与 self 类型相同的空张量 result
    const auto float_type = c10::toRealValueType(self.scalar_type());
    Tensor result = at::empty_like(self, self.options().dtype(float_type));
    // 调用 out_impl 处理 result 和 self，返回结果张量
    return out_impl(result, self);
  }

  // 否则创建一个空的张量 result，使用 self 的选项
  Tensor result = at::empty({0}, self.options());
  // 调用 out_impl 处理 result 和 self，返回结果张量
  return out_impl(result, self);
}

template <typename OutImpl>
static inline Tensor& unary_op_impl_(Tensor& self, OutImpl& out_impl) {
  // 调用 out_impl 处理 self 和 self 自身，并返回结果张量
  return out_impl(self, self);
}

// arccos, alias for acos
// arccos，acos 的别名
Tensor& arccos_out(const Tensor& self, Tensor& result) { return at::acos_out(result, self); }
Tensor arccos(const Tensor& self) { return self.acos(); }
Tensor& arccos_(Tensor& self) { return self.acos_(); }

Tensor& rad2deg_out(const Tensor& self, Tensor& result) {
  // 检查是否为复数张量，若是则抛出错误
  TORCH_CHECK(!self.is_complex(), "rad2deg is not supported for complex tensors.");
  // 定义常量 M_180_PI，表示弧度到角度的转换系数
  constexpr double M_180_PI = 57.295779513082320876798154814105170332405472466564;
  // 将 self 乘以 M_180_PI，结果存入 result
  return at::mul_out(result, self, wrapped_scalar_tensor(Scalar(M_180_PI)));
}
Tensor rad2deg(const Tensor& self) {
  // 注意：整数到浮点数的提升与其他一元操作不同，
  // 因为它不使用通常的 TensorIterator + Kernel Dispatch 模式。
  auto options = self.options();
  // 如果 self 的标量类型是整数（包括布尔值），使用默认浮点数类型的选项
  if (c10::isIntegralType(self.scalar_type(), /*includeBool=*/true)) {
    options = options.dtype(c10::get_default_dtype());
  }
  // 创建与 self 类型相同的空张量 result
  auto result = at::empty_like(self, options);
  // 调用 rad2deg_out 处理 self 和 result
  at::rad2deg_out(result, self);
  // 返回结果张量
  return result;
}
Tensor& rad2deg_(Tensor& self) { return unary_op_impl_(self, at::rad2deg_out); }

Tensor& deg2rad_out(const Tensor& self, Tensor& result) {
  // 检查是否为复数张量，若是则抛出错误
  TORCH_CHECK(!self.is_complex(), "deg2rad is not supported for complex tensors.");
  // 定义常量 M_PI_180，表示角度到弧度的转换系数
  constexpr double M_PI_180 = 0.017453292519943295769236907684886127134428718885417;
  // 将 self 乘以 M_PI_180，结果存入 result
  return at::mul_out(result, self, wrapped_scalar_tensor(Scalar(M_PI_180)));
}
Tensor deg2rad(const Tensor& self) {
  // 注意：整数到浮点数的提升与其他一元操作不同，
  // 因为它不使用通常的 TensorIterator + Kernel Dispatch 模式。
  auto options = self.options();
  // 如果 self 的标量类型是整数（包括布尔值），使用默认浮点数类型的选项
  if (c10::isIntegralType(self.scalar_type(), /*includeBool=*/true)) {
    options = options.dtype(c10::get_default_dtype());
  }
  auto result = at::empty_like(self, options);
  // 创建一个与输入张量 `self` 相同形状的空张量 `result`
  at::deg2rad_out(result, self);
  // 将输入张量 `self` 的值转换为弧度制，并将结果存储在 `result` 中
  return result;
  // 返回经过弧度制转换后的张量 `result`
}

// 将自身张量中的角度值转换为弧度值（in-place）
Tensor& deg2rad_(Tensor& self) { return unary_op_impl_(self, at::deg2rad_out); }

// arcsin，asin 的别名
Tensor& arcsin_out(const Tensor& self, Tensor& result) { return at::asin_out(result, self); }
Tensor arcsin(const Tensor& self) { return self.asin(); }
Tensor& arcsin_(Tensor& self) { return self.asin_(); }

// arctan，atan 的别名
Tensor& arctan_out(const Tensor& self, Tensor& result) { return at::atan_out(result, self); }
Tensor arctan(const Tensor& self) { return self.atan(); }
Tensor& arctan_(Tensor& self) { return self.atan_(); }

// 注释 [Complex abs and angle]
// 复数输入到 abs 和 angle 返回默认的浮点数结果。
// abs 和 angle 在 NumPy 和 C++ 中，当给定复数输入时返回浮点数结果。
// 这在数学上是合理的，因为复数的绝对值和角度没有虚部。
Tensor& abs_out(const Tensor& self, Tensor& result) {
  return unary_op_impl_with_complex_to_float_out(result, self, abs_stub, /*promotes_integer_to_float=*/false);
}
Tensor abs(const Tensor& self) {
  return unary_op_impl_with_complex_to_float(self, at::abs_out);
}
Tensor& abs_(Tensor& self) {
  TORCH_CHECK(!self.is_complex(), "In-place abs is not supported for complex tensors.");
  return unary_op_impl_(self, at::abs_out);
}

// Absolute，abs 的别名
Tensor& absolute_out(const Tensor& self, Tensor& result) {
  return at::abs_out(result, self);
}
Tensor absolute(const Tensor& self) {
  return self.abs();
}
Tensor& absolute_(Tensor& self) {
  return self.abs_();
}

// 计算张量中每个元素的角度值（out-place）
Tensor& angle_out(const Tensor& self, Tensor& result) {
  return unary_op_impl_with_complex_to_float_out(result, self, angle_stub, /*promotes_integer_to_float=*/true);
}
Tensor angle(const Tensor& self) {
  if (self.is_complex()) {
    const auto float_type = c10::toRealValueType(self.scalar_type());
    Tensor result = at::empty({0}, self.options().dtype(float_type));
    return at::angle_out(result, self);
  }

  return unary_op_impl_float(self, angle_stub);
}

// 获取张量的实部
Tensor real(const Tensor& self) {
  if (self.is_complex()) {
    Tensor real_tensor;
    if (self.is_conj()) {
      real_tensor = at::view_as_real(self._conj());
    } else {
      real_tensor = at::view_as_real(self);
    }
    return at::select(real_tensor, real_tensor.dim() - 1, 0);
  } else {
    return self;
  }
}

// 返回张量的负视图
Tensor _neg_view(const Tensor& self) {
  Tensor self_ = self.alias();
  self_._set_neg(!self.is_neg());
  namedinference::propagate_names(self_, self);
  return self_;
}

// 获取张量的虚部
Tensor imag(const Tensor& self) {
  if (self.is_complex()) {
    Tensor real_tensor;
    if (self.is_conj()) {
      real_tensor = at::view_as_real(self._conj());
      // 预先设置最终 imag 张量的负标志
      real_tensor = real_tensor._neg_view();
    } else {
      real_tensor = at::view_as_real(self);
    }
    return at::select(real_tensor, real_tensor.dim() - 1, 1);
  } else {
    // 如果不是复数张量，则直接返回自身
    return self;
  }
}
    # 使用 TORCH_CHECK 断言检查条件是否为 false，如果是，则抛出错误消息
    TORCH_CHECK(false, "imag is not implemented for tensors with non-complex dtypes.");
  }
}

# 在物理共轭操作中，将输入张量 `self` 进行共轭操作，并将结果存储到输出张量 `result` 中
Tensor& conj_physical_out(const Tensor& self, Tensor& result) {
  return unary_op_impl_out(result, self, conj_physical_stub);
}

# 对输入张量 `self` 执行物理共轭操作，并返回新的张量
Tensor _conj_physical(const Tensor& self) {
  # 如果输入张量已经是共轭的，则返回其克隆副本
  if (self.is_conj()) {
    return self.conj().clone();
  }
  # 否则调用通用的一元操作实现函数 `unary_op_impl`，传递物理共轭操作的函数指针 `conj_physical_stub`
  return unary_op_impl(self, at::conj_physical_out);
}

# 对输入张量 `self` 进行物理共轭操作
Tensor conj_physical(const Tensor& self) {
  # 如果输入张量不是复数类型，则直接返回该张量
  if (!self.is_complex()) return self;
  # 否则调用 `_conj_physical` 函数进行物理共轭操作
  return at::_conj_physical(self);
}

# 对输入张量 `self` 进行原地物理共轭操作
Tensor& conj_physical_(Tensor& self) {
  # 如果输入张量不是复数类型，则直接返回该张量
  if (!self.is_complex()) return self;
  # 否则调用通用的原地一元操作实现函数 `unary_op_impl_out`，传递物理共轭操作的函数指针 `conj_physical_stub`
  return unary_op_impl_out(self, self, conj_physical_stub);
}

# 如果输入张量 `self` 的负数标志位未设置，则返回原张量；否则返回一个新的克隆张量，其负数标志位被设置为未激活
Tensor resolve_neg(const Tensor& self) {
  # 如果输入张量的负数标志位未设置，则直接返回该张量
  if (!self.is_neg()) { return self; }
  # 否则返回输入张量的克隆副本，即将负数标志位重置为未激活
  return self.clone();
}

# 如果输入张量 `self` 的共轭标志位未设置，则返回原张量；否则返回一个新的克隆张量，其共轭标志位被设置为未激活
Tensor resolve_conj(const Tensor& self) {
  # 如果输入张量的共轭标志位未设置，则直接返回该张量
  if (!self.is_conj()) { return self; }
  # 否则返回输入张量的克隆副本，即将共轭标志位重置为未激活
  return self.clone();
}

# 对输入张量 `self` 的共轭标志位取反，并返回修改后的张量
Tensor _conj(const Tensor& self) {
  # 创建输入张量的别名 `self_`
  Tensor self_ = self.alias();
  # 对 `self_` 的共轭标志位进行取反操作
  self_._set_conj(!self.is_conj());
  # 将原张量 `self` 的命名推断属性传播到 `self_`
  namedinference::propagate_names(self_, self);
  # 返回更新后的张量 `self_`
  return self_;
}

# 对输入张量 `self` 执行共轭操作
Tensor conj(const Tensor& self) {
  # 此处看似会导致无限递归，实际上调用了张量类中定义的 `conj()` 函数
  return self.conj();
}

# `special_exp2` 是 `exp2` 的别名函数
Tensor& special_exp2_out(const Tensor& self, Tensor& result) { return at::exp2_out(result, self); }
Tensor special_exp2(const Tensor& self) { return self.exp2(); }

# `special_expm1` 是 `expm1` 的别名函数
Tensor& special_expm1_out(const Tensor& self, Tensor& result) { return at::expm1_out(result, self); }
Tensor special_expm1(const Tensor& self) { return self.expm1(); }

# `special_erf` 是 `erf` 的别名函数
Tensor& special_erf_out(const Tensor& self, Tensor& result) { return at::erf_out(result, self); }
Tensor special_erf(const Tensor& self) { return self.erf(); }

# `special_erfc` 是 `erfc` 的别名函数
Tensor& special_erfc_out(const Tensor& self, Tensor& result) { return at::erfc_out(result, self); }
Tensor special_erfc(const Tensor& self) { return self.erfc(); }

# `special_erfinv` 是 `erfinv` 的别名函数
Tensor& special_erfinv_out(const Tensor& self, Tensor& result) { return at::erfinv_out(result, self); }
Tensor special_erfinv(const Tensor& self) { return self.erfinv(); }

# `special_polygamma` 是 `polygamma` 的别名函数
Tensor& special_polygamma_out(int64_t n, const Tensor& self, Tensor& result) { return at::polygamma_out(result, n, self); }
Tensor special_polygamma(int64_t n, const Tensor& self) { return self.polygamma(n); }

# `special_psi` 是 `digamma` 的别名函数
Tensor& special_psi_out(const Tensor& self, Tensor& result) { return at::digamma_out(result, self); }
Tensor special_psi(const Tensor& self) { return self.digamma(); }
# `special_digamma` 是 `digamma` 的别名函数
// 返回经特殊 digamma 函数计算后的结果，将结果存储在给定的张量中
Tensor& special_digamma_out(const Tensor& self, Tensor& result) { return at::digamma_out(result, self); }

// 返回经特殊 digamma 函数计算后的结果
Tensor special_digamma(const Tensor& self) { return self.digamma(); }

// special_i0，别名函数，等同于调用 i0 函数
Tensor& special_i0_out(const Tensor& self, Tensor& result) { return at::i0_out(result, self); }

// 返回经 special_i0 函数计算后的结果
Tensor special_i0(const Tensor& self) { return self.i0(); }

// special_log1p，别名函数，等同于调用 log1p 函数
Tensor& special_log1p_out(const Tensor& self, Tensor& result) { return at::log1p_out(result, self); }

// 返回经 special_log1p 函数计算后的结果
Tensor special_log1p(const Tensor& self) { return self.log1p(); }

// special_round，别名函数，等同于调用 round 函数
Tensor& special_round_out(const Tensor& self, int64_t decimals, Tensor& result) { return at::round_out(result, self, decimals); }

// 返回经 special_round 函数计算后的结果
Tensor special_round(const Tensor& self, int64_t decimals) { return self.round(decimals); }

// special_sinc，别名函数，等同于调用 sinc 函数
Tensor& special_sinc_out(const Tensor& self, Tensor& result) { return at::sinc_out(result, self); }

// 返回经 special_sinc 函数计算后的结果
Tensor special_sinc(const Tensor& self) { return self.sinc(); }

namespace {

// 计算正态分布函数的累积分布函数 (CDF)，返回结果
inline Tensor calc_ndtr(const Tensor& self) {
  auto x_sqrt_2 = self * M_SQRT1_2;
  return (1 + at::erf(x_sqrt_2)) * 0.5;
}

} // namespace

// special_ndtr，别名函数，等同于调用 calc_ndtr 函数
Tensor& special_ndtr_out(const Tensor& self, Tensor& result) {
  // 检查输入张量和输出张量在同一设备上
  TORCH_CHECK(
      self.device() == result.device(),
      "Expected all tensors to be on the same device, but found at least two devices, ",
      self.device(),
      " and ",
      result.device(),
      "!");

  // 计算正态分布函数的累积分布函数 (CDF)
  auto ndtr = calc_ndtr(self);
  
  // 检查能否将 ndtr 的数据类型转换为结果张量的数据类型
  TORCH_CHECK(
      at::can_cast(ndtr.scalar_type(), result.scalar_type()),
      "result type ",
      ndtr.scalar_type(),
      " can't be cast to the desired output type ",
      result.scalar_type());

  // 调整输出张量的大小，使其与 ndtr 张量相同
  at::native::resize_output(result, ndtr.sizes());
  
  // 将 ndtr 张量的数据复制到结果张量中
  return result.copy_(ndtr);
}

// 返回经 special_ndtr 函数计算后的结果
Tensor special_ndtr(const Tensor& self) {
  return calc_ndtr(self);
}

// FIXME: 更新 unary_op_impl_out 后删除 const_cast
// 实现 sgn 函数的输出版本，将结果存储在给定的张量中
TORCH_IMPL_FUNC(sgn_out) (const Tensor& self, const Tensor& result) {
  // 如果输入张量是复数类型，则调用 sgn_stub
  if (self.is_complex()) {
    sgn_stub(device_type(), *this);
  } else {
    // 否则，调用 sign_stub
    sign_stub(device_type(), *this);
  }
}

// arccosh，别名函数，等同于调用 acosh 函数
Tensor& arccosh_out(const Tensor& self, Tensor& result) { return at::acosh_out(result, self); }

// 返回经 arccosh 函数计算后的结果
Tensor arccosh(const Tensor& self) { return at::acosh(self); }

// 在给定张量上执行 arcsinh 函数，并将结果存储在指定张量中
Tensor& arcsinh_out(const Tensor& self, Tensor& result) { return at::asinh_out(result, self); }

// 返回经 arcsinh 函数计算后的结果
Tensor arcsinh(const Tensor& self) { return self.asinh(); }

// 在给定张量上执行 arctanh 函数，并将结果存储在指定张量中
Tensor& arctanh_out(const Tensor& self, Tensor& result) { return at::atanh_out(result, self); }

// 返回经 arctanh 函数计算后的结果
Tensor arctanh(const Tensor& self) { return self.atanh(); }

// 返回给定张量的平方，存储在指定的结果张量中
Tensor& square_out(const Tensor& self, Tensor& result) { return at::pow_out(result, self, 2); }

// 返回给定张量的平方
Tensor square(const Tensor& self) { return at::pow(self, 2); }
// 将自身张量平方后返回（in-place 操作）
Tensor& square_(Tensor& self) { return self.pow_(2); }

// 计算输入张量的 logit 函数值，结果存储在预先分配的输出张量中
Tensor& logit_out(const Tensor& self,
    std::optional<double> eps,
    Tensor& result) {
  // 调用底层实现函数 unary_op_impl_float_out，计算 logit，并将结果存储在 result 中
  return unary_op_impl_float_out(
      result, self, logit_stub, Scalar(eps ? eps.value() : -1.0));
}

// 计算输入张量的 logit 函数值，返回新张量
Tensor logit(const Tensor& self, std::optional<double> eps) {
  // 调用底层实现函数 unary_op_impl_float，计算 logit，并返回结果张量
  return unary_op_impl_float(
      self, logit_stub, Scalar(eps ? eps.value() : -1.0));
}

// 在自身张量上执行 logit 操作（in-place 操作）
Tensor& logit_(Tensor& self, std::optional<double> eps) {
  // 调用 at::logit_out，将结果存储在自身张量中
  return at::logit_out(self, self, eps);
}

// 计算输入张量的特殊 logit 函数值，结果存储在预先分配的输出张量中
Tensor& special_logit_out(const Tensor& self, std::optional<double> eps, Tensor& result) {
  // 调用 at::logit_out，将结果存储在 result 张量中
  return at::logit_out(result, self, eps);
}

// 计算输入张量的特殊 logit 函数值，返回新张量
Tensor special_logit(const Tensor& self, std::optional<double> eps) {
  // 调用 self.logit，返回计算结果的新张量
  return self.logit(eps);
}

// special_expit 是 sigmoid 函数的别名
Tensor& special_expit_out(const Tensor& self, Tensor& result) {
  // 调用 at::sigmoid_out，将 sigmoid 函数的结果存储在 result 张量中
  return at::sigmoid_out(result, self);
}

// 计算输入张量的 sigmoid 函数值，返回新张量
Tensor special_expit(const Tensor& self) {
  // 调用 self.sigmoid，返回计算结果的新张量
  return self.sigmoid();
}

// 将输入张量中的 NaN 替换为指定值，正无穷替换为指定值，负无穷替换为指定值，结果存储在预先分配的输出张量中
Tensor& nan_to_num_out(const Tensor& self,
    std::optional<double> nan,
    std::optional<double> pos_inf,
    std::optional<double> neg_inf,
    Tensor& result) {
  // 检查输出张量的数据类型与输入张量是否一致
  TORCH_CHECK(
      self.scalar_type() == result.scalar_type(),
      "nan_to_num: dtype of out: ",
      result.scalar_type(),
      " should be same as input: ",
      self.scalar_type());

  // 若输入张量为整数类型，直接复制到输出张量中并返回
  if (c10::isIntegralType(self.scalar_type(), /*includeBool=*/true)) {
    at::native::resize_output(result, self.sizes());
    result.copy_(self);
    return result;
  }

  // 否则，使用 TensorIterator 执行 unary_op，将 NaN 替换为指定值，正负无穷替换为指定值
  auto iter = TensorIterator::unary_op(result, self);
  nan_to_num_stub(iter.device_type(), iter, nan, pos_inf, neg_inf);
  return result;
}

// 将输入张量中的 NaN 替换为指定值，正无穷替换为指定值，负无穷替换为指定值，返回新张量
Tensor nan_to_num(
    const Tensor& self,
    std::optional<double> nan,
    std::optional<double> pos_inf,
    std::optional<double> neg_inf) {
  // 创建与输入张量相同形状和数据类型的空张量
  auto result = at::empty_like(self);
  // 调用 at::nan_to_num_out，将结果存储在预先分配的 result 张量中
  return at::nan_to_num_out(result, self, nan, pos_inf, neg_inf);
}

// 在自身张量上执行 nan_to_num 操作（in-place 操作）
Tensor& nan_to_num_(
    Tensor& self,
    std::optional<double> nan,
    std::optional<double> pos_inf,
    std::optional<double> neg_inf) {
  // 调用 at::nan_to_num_out，将结果存储在自身张量中
  return at::nan_to_num_out(self, self, nan, pos_inf, neg_inf);
}

// fix 是 trunc 函数的别名
Tensor& fix_out(const Tensor& self, Tensor& result) { return at::trunc_out(result, self); }

// 计算输入张量的 trunc 函数值，返回新张量
Tensor fix(const Tensor& self) { return self.trunc(); }

// 在自身张量上执行 trunc 操作（in-place 操作）
Tensor& fix_(Tensor& self) { return self.trunc_(); }

// 返回输入张量本身，用于正数运算
Tensor positive(const Tensor& self) {
  // 若输入张量为布尔类型，抛出异常
  TORCH_CHECK(self.scalar_type() != kBool, "The `+` operator, on a bool tensor is not supported.");
  return self;
}

// 将输入张量中的元素取负，结果存储在预先分配的输出张量中
Tensor& negative_out(const Tensor& self, Tensor& result) { return at::neg_out(result, self); }

// 返回输入张量的负值，返回新张量
Tensor negative(const Tensor& self) { return self.neg(); }

// 在自身张量上执行取负操作（in-place 操作）
Tensor& negative_(Tensor& self) { return self.neg_(); }

// 计算输入张量的逻辑非运算，返回新张量
Tensor logical_not(const Tensor& self) {
  // 创建布尔类型的空张量作为输出
  Tensor result = at::empty({0}, self.options().dtype(kBool));
  // 调用 at::logical_not_out，计算逻辑非，并将结果存储在 result 中
  return at::logical_not_out(result, self);
}

// 在自身张量上执行逻辑非操作（in-place 操作）
Tensor& logical_not_(Tensor& self) {
  // 调用 at::logical_not_out，将结果存储在自身张量中
  return at::logical_not_out(self, self);
}
Tensor& logical_not_out(const Tensor& self, Tensor& result) {
  // 创建一个张量迭代器，配置为不检查所有输入张量的数据类型是否相同
  TensorIterator iter = TensorIteratorConfig()
    .check_all_same_dtype(false)
    // 将输出张量添加到迭代器
    .add_output(result)
    // 将输入张量作为常量添加到迭代器
    .add_const_input(self)
    // 构建迭代器
    .build();
  // 调用逻辑非的实现函数，传入设备类型和迭代器
  logical_not_stub(iter.device_type(), iter);
  // 返回输出张量
  return result;
}

namespace {
// 定义常量 HALF，其值为 0.5
constexpr double HALF = 0.5;
// 定义常量 QUARTER，其值为 0.25
constexpr double QUARTER = 0.25;
}

static inline void mvlgamma_check(const Tensor& self, int64_t p) {
  // 检查输入张量是否为布尔类型
  TORCH_CHECK(self.scalar_type() != kBool, "The input tensor may not be a boolean tensor.");
  // 检查 p 是否大于等于 1
  TORCH_CHECK(p >= 1, "p has to be greater than or equal to 1");
}

Tensor mvlgamma(const Tensor& self, int64_t p) {
  // 调用检查函数，确保输入张量和 p 的合法性
  mvlgamma_check(self, p);
  // 获取输入张量的数据类型
  auto dtype = c10::scalarTypeToTypeMeta(self.scalar_type());
  // 如果输入张量是整数类型（包括布尔类型），则将数据类型提升为默认的浮点类型
  if (at::isIntegralType(self.scalar_type(), /*include_bool=*/true)) {
    dtype = c10::get_default_dtype();
  }
  // 创建一个张量 args，其值为从 -p * HALF + HALF 开始，步长为 HALF，终点为 HALF
  Tensor args = native::arange(
      -p * HALF + HALF,
      HALF,
      HALF,
      optTypeMetaToScalarType(dtype),
      self.options().layout_opt(),
      self.options().device_opt(),
      self.options().pinned_memory_opt());
  // 在 args 上添加一个维度，并与原始张量相加
  args = args.add(self.unsqueeze(-1));
  // 计算 p * (p - 1) 的值
  const auto p2_sub_p = static_cast<double>(p * (p - 1));
  // 返回对 args 张量每个元素进行 lgamma 操作后按最后一个维度求和，并加上额外的数值
  return args.lgamma_().sum(-1).add_(p2_sub_p * std::log(c10::pi<double>) * QUARTER);
}

Tensor& mvlgamma_(Tensor& self, int64_t p) {
  // 调用检查函数，确保输入张量和 p 的合法性
  mvlgamma_check(self, p);
  // 创建一个张量 args，其值为从 -p * HALF + HALF 开始，步长为 HALF，终点为 HALF
  Tensor args = native::arange(
      -p * HALF + HALF,
      HALF,
      HALF,
      optTypeMetaToScalarType(self.options().dtype_opt()),
      self.options().layout_opt(),
      self.options().device_opt(),
      self.options().pinned_memory_opt());
  // 在 args 上添加一个维度，并与原始张量相加
  args = args.add(self.unsqueeze(-1));
  // 计算 p * (p - 1) 的值
  const auto p2_sub_p = static_cast<double>(p * (p - 1));
  // 在原始张量上执行 args 张量每个元素进行 lgamma 操作后按最后一个维度求和，并加上额外的数值，并返回结果的引用
  return self.copy_(args.lgamma_().sum(-1).add_(p2_sub_p * std::log(c10::pi<double>) * QUARTER));
}

Tensor& mvlgamma_out(const Tensor& self, int64_t p, Tensor& result) {
  // 调用 mvlgamma 函数计算输出张量 out
  auto out = self.mvlgamma(p);
  // 检查是否可以将 out 的数据类型转换为结果张量 result 的数据类型
  TORCH_CHECK(
      at::can_cast(out.scalar_type(), result.scalar_type()),
      "mvlgamma: result type ",
      self.scalar_type(),
      " can't be cast to the desired output type ",
      out.scalar_type());
  // 调整结果张量 result 的大小以匹配 out 的大小
  at::native::resize_output(result, out.sizes());
  // 将 out 的内容复制到结果张量 result，并返回结果的引用
  return result.copy_(out);
}

Tensor special_multigammaln(const Tensor& self, int64_t p) {
  // 调用 mvlgamma 函数计算特殊多元 gamma 函数的自然对数
  return self.mvlgamma(p);
};

Tensor& special_multigammaln_out(const Tensor& self, int64_t p, Tensor& result) {
  // 调用 mvlgamma_out 函数计算特殊多元 gamma 函数的自然对数，并返回结果的引用
  return at::mvlgamma_out(result, self, p);
};

std::tuple<Tensor, Tensor> frexp(const Tensor& self) {
  // 创建一个与输入张量相同大小的空张量 mantissa
  Tensor mantissa = at::empty_like(self);
  // 创建一个与输入张量相同大小的整数类型张量 exponent
  Tensor exponent = at::empty_like(self, self.options().dtype(at::kInt));

  // 调用 frexp_out 函数计算输入张量的 mantissa 和 exponent，并返回二元组
  at::frexp_out(mantissa, exponent, self);
  return std::tuple<Tensor, Tensor>(mantissa, exponent);
}
// 返回一个 tuple，包含两个引用 Tensor，分别表示 frexp 操作的结果 mantissa 和 exponent
std::tuple<Tensor&, Tensor&> frexp_out(const Tensor& self,
                                       Tensor& mantissa, Tensor& exponent) {
  // 检查 self 是否为浮点数类型，因为 torch.frexp 目前仅支持浮点数类型
  TORCH_CHECK(at::isFloatingType(self.scalar_type()),
              "torch.frexp() only supports floating-point dtypes");

  // 检查 mantissa 的 dtype 是否与 self 相同
  TORCH_CHECK(mantissa.dtype() == self.dtype(),
              "torch.frexp() expects mantissa to have dtype ", self.dtype(),
              " but got ", mantissa.dtype());
  
  // 检查 exponent 的 dtype 是否为整型 (int)
  TORCH_CHECK(exponent.dtype() == at::kInt,
              "torch.frexp() expects exponent to have int dtype "
              "but got ", exponent.dtype());

  // 创建一个 Tensor 迭代器配置
  auto iter = TensorIteratorConfig()
    .add_output(mantissa)  // 添加 mantissa 为输出
    .add_output(exponent)  // 添加 exponent 为输出
    .add_const_input(self) // 添加 self 为常量输入
    .check_all_same_dtype(false)  // 不检查所有 Tensor 的 dtype 是否相同
    .set_check_mem_overlap(true)  // 设置检查内存重叠
    .build();
  
  // 调用 frexp_stub 来执行实际的 frexp 操作，根据设备类型来选择具体的实现
  frexp_stub(iter.device_type(), iter);

  // 返回包含 mantissa 和 exponent 引用的 tuple
  return std::tuple<Tensor&, Tensor&>(mantissa, exponent);
}

// 别名函数，等同于 lgamma，实现 special.gammaln 相当于 scipy.special.gammaln
Tensor special_gammaln(const Tensor& self) { return self.lgamma(); }

// special_gammaln 的 out 版本，将 lgamma 的结果存储到 result 中
Tensor& special_gammaln_out(const Tensor& self, Tensor& result) { return at::lgamma_out(result, self); }

// 下面的代码定义了一系列的分发调度函数，用于不同的数学操作和位操作，例如 abs、acos、cos 等
// 这些宏会生成特定操作的分发函数，以便在不同的后端实现中选择正确的处理逻辑
DEFINE_DISPATCH(abs_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(angle_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(conj_physical_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(acos_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(acosh_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(asinh_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(atanh_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(asin_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(atan_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(bitwise_not_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(ceil_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(cos_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(cosh_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(digamma_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(special_entr_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(special_erfcx_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(erf_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(erfc_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
# 定义并声明一系列函数调度的宏，每个宏表示一个特定数学函数的存根
DEFINE_DISPATCH(erfinv_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(exp_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(exp2_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(expm1_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(floor_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(frac_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(frexp_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(i0_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(special_i0e_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(special_i1_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(special_i1e_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(log_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(log10_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(log1p_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(log2_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(logical_not_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(special_ndtri_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(special_log_ndtr_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(neg_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(nan_to_num_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(polygamma_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(reciprocal_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(round_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(round_decimals_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(rsqrt_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(sigmoid_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(logit_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(sign_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(signbit_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(sgn_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(sin_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(sinc_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
// 定义和声明 sinh_stub 函数的分发器
DEFINE_DISPATCH(sinh_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
// 定义和声明 sqrt_stub 函数的分发器
DEFINE_DISPATCH(sqrt_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
// 定义和声明 tan_stub 函数的分发器
DEFINE_DISPATCH(tan_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
// 定义和声明 tanh_stub 函数的分发器
DEFINE_DISPATCH(tanh_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
// 定义和声明 trigamma_stub 函数的分发器
DEFINE_DISPATCH(trigamma_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
// 定义和声明 trunc_stub 函数的分发器
DEFINE_DISPATCH(trunc_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
// 定义和声明 lgamma_stub 函数的分发器
DEFINE_DISPATCH(lgamma_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
// 定义和声明 special_airy_ai_stub 函数的分发器
DEFINE_DISPATCH(special_airy_ai_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
// 定义和声明 special_bessel_j0_stub 函数的分发器
DEFINE_DISPATCH(special_bessel_j0_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
// 定义和声明 special_bessel_j1_stub 函数的分发器
DEFINE_DISPATCH(special_bessel_j1_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
// 定义和声明 special_bessel_y0_stub 函数的分发器
DEFINE_DISPATCH(special_bessel_y0_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
// 定义和声明 special_bessel_y1_stub 函数的分发器
DEFINE_DISPATCH(special_bessel_y1_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
// 定义和声明 special_modified_bessel_i0_stub 函数的分发器
DEFINE_DISPATCH(special_modified_bessel_i0_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
// 定义和声明 special_modified_bessel_i1_stub 函数的分发器
DEFINE_DISPATCH(special_modified_bessel_i1_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
// 定义和声明 special_modified_bessel_k0_stub 函数的分发器
DEFINE_DISPATCH(special_modified_bessel_k0_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
// 定义和声明 special_modified_bessel_k1_stub 函数的分发器
DEFINE_DISPATCH(special_modified_bessel_k1_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
// 定义和声明 special_scaled_modified_bessel_k0_stub 函数的分发器
DEFINE_DISPATCH(special_scaled_modified_bessel_k0_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
// 定义和声明 special_scaled_modified_bessel_k1_stub 函数的分发器
DEFINE_DISPATCH(special_scaled_modified_bessel_k1_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
// 定义和声明 special_spherical_bessel_j0_stub 函数的分发器
DEFINE_DISPATCH(special_spherical_bessel_j0_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)

} // namespace at::native
```