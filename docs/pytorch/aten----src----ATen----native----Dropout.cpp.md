# `.\pytorch\aten\src\ATen\native\Dropout.cpp`

```py
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/TensorOperators.h>
#include <c10/util/irange.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/alpha_dropout_native.h>
#include <ATen/ops/dropout_native.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/feature_alpha_dropout_native.h>
#include <ATen/ops/feature_dropout_native.h>
#include <ATen/ops/native_dropout.h>
#include <ATen/ops/native_dropout_backward_native.h>
#include <ATen/ops/native_dropout_native.h>
#include <ATen/ops/ones_like.h>
#include <ATen/ops/zeros.h>
#endif

namespace at::native {

namespace {

// 创建特征噪声张量
Tensor make_feature_noise(const Tensor& input) {
  auto input_sizes = input.sym_sizes();
  TORCH_CHECK(input.dim() >= 2, "Feature dropout requires at least 2 dimensions in the input");
  c10::SymDimVector sizes;
  sizes.reserve(input.dim());
  sizes.push_back(input_sizes[0]);
  sizes.push_back(input_sizes[1]);
  for (C10_UNUSED const auto i : c10::irange(2, input.dim())) {
    sizes.push_back(1);
  }
  return input.new_empty_symint(sizes);
}

// 检查融合内核是否可接受
bool is_fused_kernel_acceptable(const Tensor& input, double p) {
  return (input.is_cuda() || input.is_xpu() || input.is_lazy() || input.is_privateuseone()) && p > 0 && p < 1 && input.sym_numel() > 0;
}

// 基于是否原地操作选择相乘的函数重载
template<bool inplace>
Tensor& multiply(Tensor& input, const Tensor& noise) {
  static_assert(inplace, "Wrong multiply overload triggered in Dropout.cpp");
  return input.mul_(noise);
}

// 非原地操作的相乘函数重载
template<bool inplace>
Tensor multiply(const Tensor& input, const Tensor& noise) {
  static_assert(!inplace, "Wrong multiply overload triggered in Dropout.cpp");
  return input.mul(noise);
}

// 实现丢弃操作的模板函数，根据输入参数进行不同的处理
template<bool feature_dropout, bool alpha_dropout, bool inplace, typename T>
Ctype<inplace> _dropout_impl(T& input, double p, bool train) {
  TORCH_CHECK(p >= 0 && p <= 1, "dropout probability has to be between 0 and 1, but got ", p);
  if (p == 0 || !train || input.sym_numel() == 0) {
    return input;  // 如果概率为0，或者不训练，或者输入张量元素数为0，直接返回输入
  }

  if (p == 1) {
    return multiply<inplace>(input, at::zeros({}, input.options()));  // 如果概率为1，将输入与零张量相乘（用于全丢弃）
  }

  at::Tensor b;  // 仅用于 alpha_dropout
  auto noise = feature_dropout ? make_feature_noise(input) : at::empty_like(input);  // 生成噪声张量或空张量
  noise.bernoulli_(1 - p);  // 对噪声张量进行伯努利采样

  if (alpha_dropout) {
    constexpr double alpha = 1.7580993408473766;
    double a = 1. / std::sqrt((alpha * alpha * p + 1) * (1 - p));
    b = noise.add(-1).mul_(alpha * a).add_(alpha * a * p);  // 对噪声张量进行 alpha_dropout 处理
    noise.mul_(a);
  } else {
    noise.div_(1 - p);  // 对噪声张量进行标准 dropout 处理
  }

  if (!alpha_dropout) {
    return multiply<inplace>(input, noise);  // 根据 inplace 参数选择原地或非原地相乘
  } else {
    return multiply<inplace>(input, noise).add_(b);  // 根据 inplace 参数选择原地或非原地相乘，并添加 b
  }
}

} // namespace

} // namespace at::native
#define ALIAS_SPECIALIZATION(ALIAS_NAME, IS_FEATURE, IS_ALPHA)                      \
template <bool inplace, typename... Args>                                           \
Ctype<inplace> ALIAS_NAME(Args&&... args) {                                         \
  return _dropout_impl<IS_FEATURE, IS_ALPHA, inplace>(std::forward<Args>(args)...); \
}
# 定义宏 ALIAS_SPECIALIZATION，用于生成模板别名函数
# ALIAS_NAME: 别名函数名称
# IS_FEATURE: 是否为特征丢弃
# IS_ALPHA: 是否为Alpha丢弃

ALIAS_SPECIALIZATION(_dropout,               false, false)
# 生成 _dropout 函数的别名，不支持特征和Alpha丢弃

ALIAS_SPECIALIZATION(_feature_dropout,       true,  false)
# 生成 _feature_dropout 函数的别名，支持特征丢弃但不支持Alpha丢弃

ALIAS_SPECIALIZATION(_alpha_dropout,         false, true )
# 生成 _alpha_dropout 函数的别名，支持Alpha丢弃但不支持特征丢弃

ALIAS_SPECIALIZATION(_feature_alpha_dropout, true,  true )
# 生成 _feature_alpha_dropout 函数的别名，同时支持特征和Alpha丢弃

} // anonymous namespace
# 结束匿名命名空间

std::tuple<Tensor,Tensor>
native_dropout_cpu(const Tensor& input, double p, std::optional<bool> train) {
# 定义函数 native_dropout_cpu，实现CPU上的dropout操作
  if (input.numel() == 0) {
  # 如果输入张量为空，则直接返回空的输出和空的掩码张量
    return std::make_tuple(input, at::empty_like(input, input.options()));
  }

  Tensor mask;
  Tensor output;

  if (!train.has_value() || *train) {
  # 如果未指定训练标志或者训练标志为真
    double p1m = 1. - p;
    # 计算1-p
    // Check for probability of zero to avoid divide by zero and NaN results
    # 检查概率是否为零，以避免除以零和NaN结果
    double scale = p1m == 0 ? 0. : 1. / p1m;
    # 计算比例因子，避免p1m为零时除法错误
    mask = at::empty_like(input, input.options().dtype(c10::CppTypeToScalarType<bool>::value));
    # 创建与输入张量相同类型的空掩码张量
    mask.bernoulli_(p1m);
    # 生成伯努利分布的掩码张量，用于丢弃操作
    output = input.mul(mask).mul_(scale);
    # 应用掩码并乘以比例因子得到输出张量
  } else {
  # 如果训练标志为假
    mask = at::ones_like(input, input.options().dtype(c10::CppTypeToScalarType<bool>::value));
    # 创建与输入张量相同类型的全1张量作为掩码
    output = input.clone();
    # 复制输入张量作为输出
  }
  return std::make_tuple(output, mask);
  # 返回输出张量和掩码张量的元组
}

Tensor native_dropout_backward(const Tensor& grad, const Tensor& mask, double scale) {
# 定义函数 native_dropout_backward，计算dropout操作的反向传播梯度
  Tensor result = grad * mask * scale;
  # 计算梯度结果，考虑掩码和比例因子
  return result;
  # 返回计算得到的梯度张量
}

Tensor dropout(const Tensor& input, double p, bool train) {
# 定义函数 dropout，实现dropout操作
  auto result = [&]() {
    NoNamesGuard guard;
    # 使用 NoNamesGuard 保护不命名的操作
    // TODO: we can remove this is_nested() code smell in the future
    # 可以在将来去除此处的 is_nested() 代码以提升代码质量
    # 如果支持嵌套张量或者训练为真且支持融合内核，则使用 native_dropout 函数
    if (input.is_nested() || (train && is_fused_kernel_acceptable(input, p))) {
      return std::get<0>(at::native_dropout(input, p, train));
    }
    # 否则调用 _dropout 函数
    return _dropout<false>(input, p, train);
  }();
  namedinference::propagate_names(result, input);
  # 传播结果张量的名称信息
  return result;
  # 返回dropout操作的结果张量
}

Tensor& dropout_(Tensor& input, double p, bool train) {
# 定义函数 dropout_，原地版本的dropout操作
  return _dropout<true>(input, p, train);
  # 调用 _dropout 函数进行原地dropout操作并返回结果张量
}

Tensor feature_dropout(const Tensor& input, double p, bool train) {
# 定义函数 feature_dropout，应用特征dropout操作
  return _feature_dropout<false>(input, p, train);
  # 调用 _feature_dropout 函数进行特征dropout操作并返回结果张量
}

Tensor& feature_dropout_(Tensor& input, double p, bool train) {
# 定义函数 feature_dropout_，原地版本的特征dropout操作
  return _feature_dropout<true>(input, p, train);
  # 调用 _feature_dropout 函数进行原地特征dropout操作并返回结果张量
}

Tensor alpha_dropout(const Tensor& input, double p, bool train) {
# 定义函数 alpha_dropout，应用Alpha dropout操作
  return _alpha_dropout<false>(input, p, train);
  # 调用 _alpha_dropout 函数进行Alpha dropout操作并返回结果张量
}

Tensor& alpha_dropout_(Tensor& input, double p, bool train) {
# 定义函数 alpha_dropout_，原地版本的Alpha dropout操作
  return _alpha_dropout<true>(input, p, train);
  # 调用 _alpha_dropout 函数进行原地Alpha dropout操作并返回结果张量
}

Tensor feature_alpha_dropout(const Tensor& input, double p, bool train) {
# 定义函数 feature_alpha_dropout，应用特征和Alpha dropout操作
  return _feature_alpha_dropout<false>(input, p, train);
  # 调用 _feature_alpha_dropout 函数进行特征和Alpha dropout操作并返回结果张量
}

Tensor& feature_alpha_dropout_(Tensor& input, double p, bool train) {
# 定义函数 feature_alpha_dropout_，原地版本的特征和Alpha dropout操作
  return _feature_alpha_dropout<true>(input, p, train);
  # 调用 _feature_alpha_dropout 函数进行原地特征和Alpha dropout操作并返回结果张量
}

} // namespace at::native
# 结束命名空间 at::native
```