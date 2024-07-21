# `.\pytorch\aten\src\ATen\native\Correlation.cpp`

```py
// 定义宏以限制在编译期间仅包含方法操作符
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
// 包含张量的核心头文件
#include <ATen/core/Tensor.h>
// 包含张量操作的头文件
#include <ATen/TensorOperators.h>
// 包含张量子类化实用程序的头文件
#include <ATen/TensorSubclassLikeUtils.h>

// 如果未定义每个运算符的头文件，则包含以下功能的头文件
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
// 如果定义了每个运算符的头文件，则包含以下具体操作的头文件
#else
#include <ATen/ops/complex.h>
#include <ATen/ops/corrcoef_native.h>
#include <ATen/ops/cov.h>
#include <ATen/ops/cov_native.h>
#include <ATen/ops/imag.h>
#include <ATen/ops/mm.h>
#include <ATen/ops/real.h>
#include <ATen/ops/scalar_tensor.h>
#include <ATen/ops/sqrt.h>
#include <ATen/ops/true_divide.h>
#endif

// 命名空间定义为 at::native
namespace at::native {

// 计算给定张量的协方差矩阵
Tensor cov(
    const Tensor& self,  // 输入张量
    int64_t correction,  // 校正项
    const std::optional<Tensor>& fweights,  // 频率权重张量
    const std::optional<Tensor>& aweights) {  // 平均权重张量

  constexpr int64_t OBSERVATIONS_DIM = 1;  // 观察维度的常量定义为1

  // 检查输入张量的维度是否不超过2
  TORCH_CHECK(
      self.ndimension() <= 2,
      "cov(): expected input to have two or fewer dimensions but got an input with ",
      self.ndimension(),
      " dimensions");

  // 检查输入张量的数据类型是否不是布尔类型
  TORCH_CHECK(
      self.scalar_type() != kBool,
      "cov(): bool dtype is not supported for input");

  // 将输入张量视图为2D张量（变量，观测值）
  auto in = self.ndimension() < 2 ? self.view({1, -1}) : self;
  const auto num_observations = in.size(OBSERVATIONS_DIM);  // 获取观测值的数量

  // 频率权重与平均权重的乘积
  Tensor w;

  // 如果存在频率权重张量，则进行以下检查和操作
  if (fweights.has_value()) {
    w = fweights.value();
    // 检查频率权重张量的维度是否不超过1
    TORCH_CHECK(
        w.ndimension() <= 1,
        "cov(): expected fweights to have one or fewer dimensions but got fweights with ",
        w.ndimension(),
        " dimensions");
    // 检查频率权重张量的数据类型是否为整数类型
    TORCH_CHECK(
        at::isIntegralType(w.scalar_type(), false),
        "cov(): expected fweights to have integral dtype but got fweights with ",
        w.scalar_type(),
        " dtype");
    // 检查频率权重张量的元素数量是否与观测值数量相同
    TORCH_CHECK(
        w.numel() == num_observations,
        "cov(): expected fweights to have the same numel as there are observations in the input but got ",
        w.numel(),
        " != ",
        num_observations);
    // 检查频率权重张量的最小值是否非负
    TORCH_CHECK(
        num_observations == 0 || at::is_scalar_tensor_true(w.min().ge(0)),
        "cov(): fweights cannot be negative");
  }

  // 如果存在平均权重张量，则进行以下检查和操作
  if (aweights.has_value()) {
    const auto& aw = aweights.value();
    // 检查平均权重张量的维度是否不超过1
    TORCH_CHECK(
        aw.ndimension() <= 1,
        "cov(): expected aweights to have one or fewer dimensions but got aweights with ",
        aw.ndimension(),
        " dimensions");
    // 检查平均权重张量的数据类型是否为浮点数类型
    TORCH_CHECK(
        at::isFloatingType(aw.scalar_type()),
        "cov(): expected aweights to have floating point dtype but got aweights with ",
        aw.scalar_type(),
        " dtype");
    // 检查平均权重张量的元素数量是否与观测值数量相同
    TORCH_CHECK(
        aw.numel() == num_observations,
        "cov(): expected aweights to have the same numel as there are observations in the input but got ",
        aw.numel(),
        " != ",
        num_observations);
    // 检查平均权重张量的最小值是否非负
    TORCH_CHECK(
        num_observations == 0 || at::is_scalar_tensor_true(aw.min().ge(0)),
        "cov(): aweights cannot be negative");


这段代码定义了一个 `cov` 函数，用于计算输入张量的协方差矩阵，并包含了对输入参数的各种检查和处理。
  w = w.defined() ? w * aw : aw;


// 根据权重是否已定义，更新权重值为 w * aw 或者 aw。

  // Compute a weighted average of the observations
  const auto w_sum = w.defined()
      ? w.sum()
      : at::scalar_tensor(num_observations, in.options().dtype(kLong));

// 计算观测值的加权平均值 w_sum。如果权重 w 已定义，则对权重进行求和；否则创建一个标量张量，其值为 num_observations，数据类型与输入张量 in 相同。

  TORCH_CHECK(
      !w.defined() || at::is_scalar_tensor_true(w_sum.ne(0)),
      "cov(): weights sum to zero, can't be normalized");

// 使用 TORCH_CHECK 进行断言检查：确保权重 w 未定义，或者确保权重 w_sum 不为零。否则输出错误信息 "cov(): weights sum to zero, can't be normalized"。

  const auto avg = (w.defined() ? in * w : in).sum(OBSERVATIONS_DIM) / w_sum;

// 计算平均值 avg。如果权重 w 已定义，则将输入张量 in 与权重 w 逐元素相乘；否则直接使用输入张量 in。然后在观测维度 OBSERVATIONS_DIM 上对结果进行求和，再除以 w_sum。

  // Compute the normalization factor
  Tensor norm_factor;

// 定义标准化因子张量 norm_factor。

  if (w.defined() && aweights.has_value() && correction != 0) {
    norm_factor = w_sum - correction * (w * aweights.value()).sum() / w_sum;
  } else {
    norm_factor = w_sum - correction;
  }

// 根据条件计算标准化因子 norm_factor。如果权重 w 已定义、aweight 值已有且校正值 correction 不为零，则使用给定公式计算；否则使用简化公式计算。

  if (at::is_scalar_tensor_true(norm_factor.le(0))) {
    TORCH_WARN("cov(): degrees of freedom is <= 0. Correction should be strictly less than the number of observations.");
    norm_factor.zero_();
  }

// 如果 norm_factor 小于等于零，则发出警告 "cov(): degrees of freedom is <= 0. Correction should be strictly less than the number of observations." 并将 norm_factor 设为零。

  // Compute covariance matrix
  in = in - avg.unsqueeze(1);
  const auto c = at::mm(in, (w.defined() ? in * w : in).t().conj());
  return at::true_divide(c, norm_factor).squeeze();

// 计算协方差矩阵 c。首先更新输入张量 in 为 in 减去 avg 在第一维度上的 unsqueeze(1) 结果。然后计算矩阵乘积 in 与 (如果权重 w 已定义则为 in * w，否则为 in) 的转置共轭。最后返回 c 除以 norm_factor 的真除结果，并对结果进行压缩（squeeze）。
}

// 计算输入张量的相关系数矩阵
Tensor corrcoef(const Tensor& self) {
  // 检查输入张量的维度是否不超过2，否则抛出错误
  TORCH_CHECK(
      self.ndimension() <= 2,
      "corrcoef(): expected input to have two or fewer dimensions but got an input with ",
      self.ndimension(),
      " dimensions");

  // 计算输入张量的协方差矩阵
  auto c = at::cov(self);

  if (c.ndimension() == 0) {
    // 如果协方差是标量，返回 NaN，如果 c 是 {nan, inf, 0}，否则返回 1
    return c / c;
  }

  // 标准化协方差矩阵
  const auto d = c.diagonal();
  const auto stddev = at::sqrt(d.is_complex() ? at::real(d) : d);
  c = c / stddev.view({-1, 1});
  c = c / stddev.view({1, -1});

  // 由于浮点数舍入可能导致值不在 [-1, 1] 范围内，因此通过裁剪值来改善结果，类似于 NumPy 的做法。
  return c.is_complex()
      ? at::complex(at::real(c).clip(-1, 1), at::imag(c).clip(-1, 1))
      : c.clip(-1, 1);
}

// 命名空间结束标记，at::native 命名空间结束
} // namespace at::native
```