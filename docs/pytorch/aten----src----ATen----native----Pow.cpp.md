# `.\pytorch\aten\src\ATen\native\Pow.cpp`

```py
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/Pow.h>

#include <ATen/core/Tensor.h>
#include <ATen/ScalarOps.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/float_power_native.h>
#include <ATen/ops/pow.h>
#include <ATen/ops/pow_native.h>
#include <ATen/ops/result_type.h>
#endif

namespace at::meta {

// 定义 pow 函数的元函数重载，处理两个 Tensor 类型的输入
TORCH_META_FUNC2(pow, Tensor_Tensor) (const Tensor& base, const Tensor& exp) {
  // 调用二进制操作的构建函数，为输出结果准备空间，并使用 base 和 exp 作为参数
  build_borrowing_binary_op(maybe_get_output(), base, exp);
}

// 定义 pow 函数的元函数重载，处理一个 Tensor 和一个 Scalar 类型的输入
TORCH_META_FUNC2(pow, Tensor_Scalar) (const Tensor& base, const Scalar& exp) {
  // Numpy 兼容性检查：当 base 是整数类型且 exp 是负整数且 exp 转换为长整型小于零时抛出异常
  TORCH_CHECK(!(isIntegralType(base.scalar_type(), true) &&
              exp.isIntegral(true) && exp.toLong() < 0),
              "Integers to negative integer powers are not allowed.");

  // 确定输出的数据类型为 base 和 exp 的结果类型
  auto common_dtype = at::result_type(base, exp);
  // 使用 base 转换为 common_dtype 来构建输出的借用参数拥有的一元操作
  build_output_borrowing_argument_owning_unary_op(maybe_get_output(), base.to(common_dtype));
}

// 定义 pow 函数的元函数重载，处理一个 Scalar 和一个 Tensor 类型的输入
TORCH_META_FUNC2(pow, Scalar) (const Scalar& base, const Tensor& exp) {
    // 这个重载不直接使用 TensorIterator，试图进行快速调度，否则重新调度到 Tensor_Tensor 的重载
    auto dtype = maybe_get_output().defined() ? maybe_get_output().scalar_type() : at::result_type(base, exp);
    // 设置输出的原始步进值为 0，维度为 exp 的大小，选项为 dtype，如果有命名，则使用 exp 的命名
    set_output_raw_strided(0, exp.sizes(), {}, exp.options().dtype(dtype), exp.has_names() ? exp.names() : ArrayRef<Dimname>());
}

} // namespace at::meta

namespace at::native {

// 定义 pow_Tensor_Tensor_out 函数，处理 Tensor 和 Tensor 类型的输出
DEFINE_DISPATCH(pow_tensor_tensor_stub);
TORCH_IMPL_FUNC(pow_Tensor_Tensor_out) (const Tensor& base, const Tensor& exp, const Tensor& out) {
  // 调度到 pow_tensor_tensor_stub 处理
  pow_tensor_tensor_stub(device_type(), *this);
}

// 定义 pow_Tensor_Scalar_out 函数，处理 Tensor 和 Scalar 类型的输出
DEFINE_DISPATCH(pow_tensor_scalar_stub);
TORCH_IMPL_FUNC(pow_Tensor_Scalar_out) (const Tensor& base, const Scalar& exp, const Tensor& out) {
  if (exp.equal(0.0) || exp.equal(false)) {
    // 如果 exp 是 0 或者 false，将输出 out 填充为 1
    out.fill_(1);
  } else if (exp.equal(1.0) || exp.equal(true) ) {
    // 如果 exp 是 1 或者 true，将输出 out 复制为 base
    out.copy_(base);
  } else {
    // 否则调度到 pow_tensor_scalar_stub 处理
    pow_tensor_scalar_stub(device_type(), *this, exp);
  }
}

// 定义 pow_Scalar_out 函数，处理 Scalar 和 Tensor 类型的输出
TORCH_IMPL_FUNC(pow_Scalar_out) (const Scalar& base, const Tensor& exp, const Tensor& out) {
  if (base.equal(1.0)) {
    // 如果 base 是 1.0，将输出 out 填充为 1
    out.fill_(1);
  } else {
    // 否则调用 at::pow_out，将 base 包装成 Tensor 并使用 exp 进行幂运算，重新调度
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
    at::pow_out(const_cast<Tensor&>(out), wrapped_scalar_tensor(base, exp.device()), exp); // redispatch!
  }
}

// 定义 float_power_out 函数，处理复数或双精度浮点数类型的幂运算输出
Tensor& float_power_out(const Tensor& base, const Tensor& exp, Tensor& result) {
  auto dtype = (at::isComplexType(base.scalar_type()) || at::isComplexType(exp.scalar_type())) ?
                at::kComplexDouble : at::kDouble;
  // 检查给定的输出 result 的数据类型是否与操作的结果数据类型匹配
  TORCH_CHECK(result.scalar_type() == dtype,
              "the output given to float_power has dtype ", result.scalar_type(),
              " but the operation's result requires dtype ", dtype);

  // 使用 at::pow_out 进行幂运算，将 base 和 exp 转换成相应的数据类型
  return at::pow_out(result, base.to(dtype), exp.to(dtype));
}

} // namespace at::native
#`
# 定义一个函数，用于计算浮点数的幂运算，并将结果存储到指定的输出张量中
Tensor& float_power_out(const Tensor& base, const Scalar& exp, Tensor& result) {
  // 确定结果张量的数据类型，基于输入张量和指数的复杂性
  auto dtype = (at::isComplexType(base.scalar_type()) || exp.isComplex()) ? at::kComplexDouble : at::kDouble;
  // 检查结果张量是否与所需的数据类型匹配
  TORCH_CHECK(result.scalar_type() == dtype,
              "the output given to float_power has dtype ", result.scalar_type(),
              " but the operation's result requires dtype ", dtype);

  // 在三元操作符中需要类型转换，因为转换函数返回例如 c10::complex，
  // 这会导致始终返回复数标量。
  auto casted_exp = (dtype == at::kComplexDouble) ? Scalar(exp.toComplexDouble()) : Scalar(exp.toDouble());
  // 调用 ATen 库的 pow_out 函数执行幂运算，并将结果存储在输出张量中
  return at::pow_out(result, base.to(dtype), casted_exp);
}

# 定义一个函数，用于计算浮点数的幂运算，并将结果存储到指定的输出张量中
Tensor& float_power_out(const Scalar& base, const Tensor& exp, Tensor& result) {
  // 确定结果张量的数据类型，基于输入张量和基数的复杂性
  auto dtype = (at::isComplexType(exp.scalar_type()) || base.isComplex()) ? at::kComplexDouble : at::kDouble;
  // 检查结果张量是否与所需的数据类型匹配
  TORCH_CHECK(result.scalar_type() == dtype,
              "the output given to float_power has dtype ", result.scalar_type(),
              " but the operation's result requires dtype ", dtype);

  // 根据数据类型进行基数的类型转换
  auto casted_base = (dtype == at::kComplexDouble) ? Scalar(base.toComplexDouble()) : Scalar(base.toDouble());
  // 调用 ATen 库的 pow_out 函数执行幂运算，并将结果存储在输出张量中
  return at::pow_out(result, casted_base, exp.to(dtype));
}

# 定义一个函数，用于计算浮点数的幂运算，并返回结果张量
Tensor float_power(const Tensor& base, const Scalar& exp) {
  // 确定结果张量的数据类型，基于输入张量和指数的复杂性
  auto dtype = (at::isComplexType(base.scalar_type()) || exp.isComplex()) ? at::kComplexDouble : at::kDouble;
  // 根据数据类型进行指数的类型转换
  auto casted_exp = (dtype == at::kComplexDouble) ? Scalar(exp.toComplexDouble()) : Scalar(exp.toDouble());
  // 调用 ATen 库的 pow 函数执行幂运算，并返回结果张量
  return at::pow(base.to(dtype), casted_exp);
}

# 定义一个函数，用于计算浮点数的幂运算，并返回结果张量
Tensor float_power(const Scalar& base, const Tensor& exp) {
  // 确定结果张量的数据类型，基于输入张量和基数的复杂性
  auto dtype = (at::isComplexType(exp.scalar_type()) || base.isComplex()) ? at::kComplexDouble : at::kDouble;
  // 根据数据类型进行基数的类型转换
  auto casted_base = (dtype == at::kComplexDouble) ? Scalar(base.toComplexDouble()) : Scalar(base.toDouble());
  // 调用 ATen 库的 pow 函数执行幂运算，并返回结果张量
  return at::pow(casted_base, exp.to(dtype));
}

# 定义一个函数，用于计算浮点数的幂运算，并返回结果张量
Tensor float_power(const Tensor& base, const Tensor& exp) {
  // 确定结果张量的数据类型，基于输入张量和指数的复杂性
  auto dtype = (at::isComplexType(base.scalar_type()) || at::isComplexType(exp.scalar_type())) ? at::kComplexDouble : at::kDouble;
  // 调用 ATen 库的 pow 函数执行幂运算，并返回结果张量
  return at::pow(base.to(dtype), exp.to(dtype));
}

# 定义一个函数，用于计算浮点数的幂运算，并将结果存储在输入张量中
Tensor& float_power_(Tensor& base, const Tensor& exp) {
  // 确定输入张量的数据类型，基于输入张量和指数的复杂性
  auto dtype = (at::isComplexType(base.scalar_type()) || at::isComplexType(exp.scalar_type())) ? at::kComplexDouble : at::kDouble;
  // 检查输入张量是否与所需的数据类型匹配
  TORCH_CHECK(base.scalar_type() == dtype,
              "the base given to float_power_ has dtype ", base.scalar_type(),
              " but the operation's result requires dtype ", dtype);

  // 调用 ATen 库的 pow_ 函数执行原位幂运算，并将结果存储在输入张量中
  return base.pow_(exp.to(dtype));
}
// 根据给定的指数将张量的每个元素进行幂运算，并更新原始张量
Tensor& float_power_(Tensor& base, const Scalar& exp) {
  // 确定结果张量的数据类型，如果基础张量或指数是复数，则结果为复数类型；否则为双精度类型
  auto dtype = (at::isComplexType(base.scalar_type()) || exp.isComplex()) ? at::kComplexDouble : at::kDouble;
  // 检查基础张量的数据类型是否与预期的结果数据类型匹配
  TORCH_CHECK(base.scalar_type() == dtype,
              "the base given to float_power_ has dtype ", base.scalar_type(),
              " but the operation's result requires dtype ", dtype);

  // 将指数类型转换为结果所需的数据类型
  auto casted_exp = (dtype == at::kComplexDouble) ? Scalar(exp.toComplexDouble()) : Scalar(exp.toDouble());
  // 对基础张量执行指数运算，并将结果更新到原始张量中
  return base.pow_(casted_exp);
}

} // namespace at::native
```