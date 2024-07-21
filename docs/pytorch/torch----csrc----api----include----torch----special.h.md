# `.\pytorch\torch\csrc\api\include\torch\special.h`

```py
#pragma once

#include <ATen/ATen.h>  // 引入 ATen 库，用于张量操作
#include <torch/types.h>  // 引入 Torch 的类型定义

namespace torch {
namespace special {

/// 计算伽马函数的绝对值的自然对数
/// 参见 https://pytorch.org/docs/main/special.html#torch.special.gammaln.
///
/// Example:
/// ```
/// auto t = torch::randn(128, dtype=kDouble);
/// torch::special::gammaln(t);
/// ```py
inline Tensor gammaln(const Tensor& self) {
  // 调用 Torch 库函数计算伽马函数的对数
  return torch::special_gammaln(self);
}

/// 计算伽马函数的绝对值的自然对数，并将结果写入指定的输出张量
inline Tensor& gammaln_out(Tensor& result, const Tensor& self) {
  // 调用 Torch 库函数计算伽马函数的对数，并将结果写入指定的输出张量
  return torch::special_gammaln_out(result, self);
}

/// 计算正则化的下不完全伽马函数
/// 参见 https://pytorch.org/docs/main/special.html#torch.special.gammainc.
///
/// Example:
/// ```
/// auto t = torch::randn(128, dtype=kDouble);
/// auto s = torch::randn(128, dtype=kDouble);
/// torch::special::gammainc(s, t);
/// ```py
inline Tensor gammainc(const Tensor& self, const Tensor& other) {
  // 调用 Torch 库函数计算正则化的下不完全伽马函数
  return torch::special_gammainc(self, other);
}

/// 计算正则化的下不完全伽马函数，并将结果写入指定的输出张量
inline Tensor& gammainc_out(
    Tensor& result,
    const Tensor& self,
    const Tensor& other) {
  // 调用 Torch 库函数计算正则化的下不完全伽马函数，并将结果写入指定的输出张量
  return torch::special_gammainc_out(result, self, other);
}

/// 计算正则化的上不完全伽马函数
/// 参见 https://pytorch.org/docs/main/special.html#torch.special.gammainc.
///
/// Example:
/// ```
/// auto t = torch::randn(128, dtype=kDouble);
/// auto s = torch::randn(128, dtype=kDouble);
/// torch::special::gammaincc(s, t);
/// ```py
inline Tensor gammaincc(const Tensor& self, const Tensor& other) {
  // 调用 Torch 库函数计算正则化的上不完全伽马函数
  return torch::special_gammaincc(self, other);
}

/// 计算正则化的上不完全伽马函数，并将结果写入指定的输出张量
inline Tensor& gammaincc_out(
    Tensor& result,
    const Tensor& self,
    const Tensor& other) {
  // 调用 Torch 库函数计算正则化的上不完全伽马函数，并将结果写入指定的输出张量
  return torch::special_gammaincc_out(result, self, other);
}

/// 计算维度为 `p` 的多元对数伽马函数，逐元素计算
/// 参见 https://pytorch.org/docs/main/special.html#torch.special.multigammaln.
///
/// Example:
/// ```
/// auto t = torch::randn(128, dtype=kDouble);
/// torch::special::multigammaln(t, 1);
/// ```py
inline Tensor multigammaln(const Tensor& self, int64_t p) {
  // 调用 Torch 库函数计算维度为 `p` 的多元对数伽马函数
  return torch::special_multigammaln(self, p);
}

/// 计算维度为 `p` 的多元对数伽马函数，逐元素计算，并将结果写入指定的输出张量
inline Tensor& multigammaln_out(Tensor& result, const Tensor& self, int64_t p) {
  // 调用 Torch 库函数计算维度为 `p` 的多元对数伽马函数，并将结果写入指定的输出张量
  return torch::special_multigammaln_out(result, self, p);
}

/// 计算输入上的第 `n` 阶数的 digamma 函数的对数导数
/// 参见 https://pytorch.org/docs/main/special.html#torch.special.polygamma.
///
/// Example:
/// ```
/// auto t = torch::randn(128, dtype=kDouble);
/// torch::special::polygamma(2, t);
/// ```py
inline Tensor polygamma(int64_t n, const Tensor& self) {
  // 调用 Torch 库函数计算输入上的第 `n` 阶数的 digamma 函数的对数导数
  return torch::special_polygamma(n, self);
}

/// 计算输入上的第 `n` 阶数的 digamma 函数的对数导数，并将结果写入指定的输出张量
inline Tensor& polygamma_out(Tensor& result, int64_t n, const Tensor& self) {
  // 调用 Torch 库函数计算输入上的第 `n` 阶数的 digamma 函数的对数导数，并将结果写入指定的输出张量
  return torch::special_polygamma_out(result, n, self);
}

/// 计算输入上的伽马函数的对数导数
/// 参见 https://pytorch.org/docs/main/special.html#torch.special.psi
///
/// Example:
/// ```
/// auto t = torch::randn(128, dtype=kDouble);
/// ```py
/// Computes the digamma function, which is the logarithmic derivative of the gamma function.
/// See https://pytorch.org/docs/main/special.html#torch.special.digamma
///
/// Example:
/// ```
/// auto t = torch::randn(128, dtype=kDouble);
/// torch::special::digamma(t);
/// ```py
inline Tensor digamma(const Tensor& self) {
  return torch::special_digamma(self);
}

/// Computes the digamma function out-of-place, storing the result in the provided result tensor.
inline Tensor& digamma_out(Tensor& result, const Tensor& self) {
  return torch::special_digamma_out(result, self);
}

/// Computes the elementwise entropy of the input tensor.
/// See https://pytorch.org/docs/main/special.html#torch.special.entr.
///
/// Example:
/// ```
/// auto t = torch::randn(128, dtype=kDouble);
/// torch::special::entr(t);
/// ```py
inline Tensor entr(const Tensor& self) {
  return torch::special_entr(self);
}

/// Computes the elementwise entropy out-of-place, storing the result in the provided result tensor.
inline Tensor& entr_out(Tensor& result, const Tensor& self) {
  return torch::special_entr_out(result, self);
}

/// Computes the error function (or Gauss error function) elementwise.
/// See https://pytorch.org/docs/main/special.html#torch.special.erf.
///
/// Example:
/// ```
/// auto t = torch::randn(128, dtype=kDouble);
/// torch::special::erf(t);
/// ```py
inline Tensor erf(const Tensor& self) {
  return torch::special_erf(self);
}

/// Computes the error function out-of-place, storing the result in the provided result tensor.
inline Tensor& erf_out(Tensor& result, const Tensor& self) {
  return torch::special_erf_out(result, self);
}

/// Computes the complementary error function elementwise.
/// See https://pytorch.org/docs/main/special.html#torch.special.erfc.
///
/// Example:
/// ```
/// auto t = torch::randn(128, dtype=kDouble);
/// torch::special::erfc(t);
/// ```py
inline Tensor erfc(const Tensor& self) {
  return torch::special_erfc(self);
}

/// Computes the complementary error function out-of-place, storing the result in the provided result tensor.
inline Tensor& erfc_out(Tensor& result, const Tensor& self) {
  return torch::special_erfc_out(result, self);
}

/// Computes the scaled complementary error function elementwise.
/// See https://pytorch.org/docs/main/special.html#torch.special.erfcx.
///
/// Example:
/// ```
/// auto t = torch::randn(128, dtype=kDouble);
/// torch::special::erfcx(t);
/// ```py
inline Tensor erfcx(const Tensor& self) {
  return torch::special_erfcx(self);
}

/// Computes the scaled complementary error function out-of-place, storing the result in the provided result tensor.
inline Tensor& erfcx_out(Tensor& result, const Tensor& self) {
  return torch::special_erfcx_out(result, self);
}

/// Computes the inverse error function (also known as the quantile function of the normal distribution) elementwise.
/// See https://pytorch.org/docs/main/special.html#torch.special.erfinv.
///
/// Example:
/// ```
/// auto t = torch::randn(128, dtype=kDouble);
/// torch::special::erfinv(t);
/// ```py
inline Tensor erfinv(const Tensor& self) {
  return torch::special_erfinv(self);
}

/// Computes the inverse error function out-of-place, storing the result in the provided result tensor.
inline Tensor& erfinv_out(Tensor& result, const Tensor& self) {
  return torch::special_erfinv_out(result, self);
}

/// Computes the log of summed exponentials of each row of input tensor in the specified dimension.
/// See https://pytorch.org/docs/main/special.html#torch.special.logsumexp.
///
/// Example:
/// ```
/// auto t = torch::randn(128, dtype=kDouble);
/// torch::special::logsumexp(t, /*dim=*/1);
/// ```py
/// Generates a 3x3 tensor of random numbers from a standard normal distribution.
/// Example:
/// ```
/// auto t = torch::randn(3, 3);
/// ```py
auto t = torch::randn(3, 3);

/// Computes the logsumexp operation along specified dimensions of the input tensor.
/// Example:
/// ```
/// torch::special::logsumexp(t, 1);
/// ```py
torch::special::logsumexp(t, 1);

/// Computes the inverse cumulative distribution function (quantile) of the standard
/// normal distribution elementwise for the input tensor.
/// Example:
/// ```
/// auto t = torch::rand(128, dtype=kDouble);
/// torch::special::ndtri(t);
/// ```py
auto t = torch::rand(128, dtype=kDouble);
torch::special::ndtri(t);

/// Computes the logarithm of the complementary cumulative distribution function
/// of the standard normal distribution elementwise for the input tensor.
/// Example:
/// ```
/// auto t = torch::randn(128, dtype=kDouble);
/// torch::special::log_ndtr(t);
/// ```py
auto t = torch::randn(128, dtype=kDouble);
torch::special::log_ndtr(t);

/// Computes the logistic sigmoid (expit) function elementwise for the input tensor.
/// Example:
/// ```
/// auto t = torch::randn(128, dtype=kDouble);
/// torch::special::logit(t);
/// ```py
auto t = torch::randn(128, dtype=kDouble);
torch::special::logit(t);

/// Computes the exponential of the elements minus 1, elementwise for the input tensor.
/// See https://pytorch.org/docs/main/special.html#torch.special.expm1.
/// (Note: The function definition is not fully provided in the snippet.)
/// Computes the exponential of each element minus one (exp(x) - 1) for the input tensor.
///
/// Example:
/// ```
/// auto t = torch::randn(128, dtype=kDouble);
/// torch::special::expm1(t);
/// ```py
inline Tensor expm1(const Tensor& self) {
  return torch::special_expm1(self);
}

/// Computes the exponential of each element minus one (exp(x) - 1) for the input tensor and writes
/// the result into the output tensor.
///
/// Example:
/// ```
/// auto t = torch::empty_like(input_tensor);
/// torch::special::expm1_out(t, input_tensor);
/// ```py
inline Tensor& expm1_out(Tensor& result, const Tensor& self) {
  return torch::special_expm1_out(result, self);
}

/// Computes x * log(y) for inputs, elementwise.
/// See https://pytorch.org/docs/main/special.html#torch.special.xlogy.
///
/// Example:
/// ```
/// auto x = torch::randn(128, dtype=kDouble);
/// auto y = torch::randn(128, dtype=kDouble);
/// torch::special::xlogy(x, y);
/// ```py
inline Tensor xlogy(const Tensor& self, const Tensor& other) {
  return torch::special_xlogy(self, other);
}

/// Computes x * log(y) for a scalar x and a tensor y, elementwise.
inline Tensor xlogy(const Scalar& self, const Tensor& other) {
  return torch::special_xlogy(self, other);
}

/// Computes x * log(y) for a tensor x and a scalar y, elementwise.
inline Tensor xlogy(const Tensor& self, const Scalar& other) {
  return torch::special_xlogy(self, other);
}

/// Computes x * log(y) for inputs, elementwise, and writes the result into the output tensor.
inline Tensor& xlogy_out(
    Tensor& result,
    const Tensor& self,
    const Tensor& other) {
  return torch::special_xlogy_out(result, self, other);
}

/// Computes x * log(y) for a scalar x and a tensor y, elementwise, and writes the result
/// into the output tensor.
inline Tensor& xlogy_out(
    Tensor& result,
    const Scalar& self,
    const Tensor& other) {
  return torch::special_xlogy_out(result, self, other);
}

/// Computes x * log(y) for a tensor x and a scalar y, elementwise, and writes the result
/// into the output tensor.
inline Tensor& xlogy_out(
    Tensor& result,
    const Tensor& self,
    const Scalar& other) {
  return torch::special_xlogy_out(result, self, other);
}

/// Computes x * log1p(y) for inputs, elementwise.
/// See https://pytorch.org/docs/main/special.html#torch.special.xlog1py.
///
/// Example:
/// ```
/// auto x = torch::randn(128, dtype=kDouble);
/// auto y = torch::randn(128, dtype=kDouble);
/// torch::special::xlog1py(x, y);
/// ```py
inline Tensor xlog1py(const Tensor& self, const Tensor& other) {
  return torch::special_xlog1py(self, other);
}

/// Computes x * log1p(y) for a scalar x and a tensor y, elementwise.
inline Tensor xlog1py(const Scalar& self, const Tensor& other) {
  return torch::special_xlog1py(self, other);
}

/// Computes x * log1p(y) for a tensor x and a scalar y, elementwise.
inline Tensor xlog1py(const Tensor& self, const Scalar& other) {
  return torch::special_xlog1py(self, other);
}

/// Computes x * log1p(y) for inputs, elementwise, and writes the result into the output tensor.
inline Tensor& xlog1py_out(
    Tensor& result,
    const Tensor& self,
    const Tensor& other) {
  return torch::special_xlog1py_out(result, self, other);
}

/// Computes x * log1p(y) for a scalar x and a tensor y, elementwise, and writes the result
/// into the output tensor.
inline Tensor& xlog1py_out(
    Tensor& result,
    const Scalar& self,
    const Tensor& other) {
  return torch::special_xlog1py_out(result, self, other);
}

/// Computes x * log1p(y) for a tensor x and a scalar y, elementwise, and writes the result
/// into the output tensor.
inline Tensor& xlog1py_out(
    Tensor& result,
    const Tensor& self,
    const Scalar& other) {
  return torch::special_xlog1py_out(result, self, other);
}

/// Computes the Hurwitz Zeta function for inputs, elementwise.
/// See https://pytorch.org/docs/main/special.html#torch.special.zeta.
///
/// Example:
/// ```
/// auto x = torch::randn(128, dtype=kDouble);
/// auto y = torch::randn(128, dtype=kDouble);
/// torch::special::zeta(x, y);
/// ```py
inline Tensor zeta(const Tensor& self, const Tensor& other) {
  return torch::special_zeta(self, other);
}

/// Computes the Hurwitz Zeta function for a scalar x and a tensor y, elementwise.
inline Tensor zeta(const Scalar& self, const Tensor& other) {
  return torch::special_zeta(self, other);
}
/// 计算输入张量 self 的 Riemann zeta 函数值
inline Tensor zeta(const Tensor& self, const Scalar& other) {
  return torch::special_zeta(self, other);
}

/// 计算输入张量 self 和 other 的 Riemann zeta 函数值，并将结果存入 result 张量
inline Tensor& zeta_out(
    Tensor& result,
    const Tensor& self,
    const Tensor& other) {
  return torch::special_zeta_out(result, self, other);
}

/// 计算标量 self 和张量 other 的 Riemann zeta 函数值，并将结果存入 result 张量
inline Tensor& zeta_out(
    Tensor& result,
    const Scalar& self,
    const Tensor& other) {
  return torch::special_zeta_out(result, self, other);
}

/// 计算张量 self 和标量 other 的 Riemann zeta 函数值，并将结果存入 result 张量
inline Tensor& zeta_out(
    Tensor& result,
    const Tensor& self,
    const Scalar& other) {
  return torch::special_zeta_out(result, self, other);
}

/// 计算输入张量 self 的零阶修正第一类 Bessel 函数，逐元素操作
///
/// 示例:
/// ```
/// auto t = torch::randn(128, dtype=kDouble);
/// torch::special::i0(t);
/// ```py
inline Tensor i0(const Tensor& self) {
  return torch::special_i0(self);
}

/// 计算输入张量 self 的零阶修正第一类 Bessel 函数，并将结果存入 result 张量
inline Tensor& i0_out(Tensor& result, const Tensor& self) {
  return torch::special_i0_out(result, self);
}

/// 计算标准高斯概率密度函数从负无穷到输入 self 的累积面积，逐元素操作
///
/// 示例:
/// ```
/// auto t = torch::randn(128, dtype=kDouble);
/// torch::special::ndtr(t);
/// ```py
inline Tensor ndtr(const Tensor& self) {
  return torch::special_ndtr(self);
}

/// 计算输入张量 self 的标准高斯累积概率密度函数值，并将结果存入 result 张量
inline Tensor& ndtr_out(Tensor& result, const Tensor& self) {
  return torch::special_ndtr_out(result, self);
}

/// 计算输入张量 self 的指数缩放零阶修正第一类 Bessel 函数，逐元素操作
///
/// 示例:
/// ```
/// auto t = torch::randn(128, dtype=kDouble);
/// torch::special::i0e(t);
/// ```py
inline Tensor i0e(const Tensor& self) {
  return torch::special_i0e(self);
}

/// 计算输入张量 self 的指数缩放零阶修正第一类 Bessel 函数，并将结果存入 result 张量
inline Tensor& i0e_out(Tensor& result, const Tensor& self) {
  return torch::special_i0e_out(result, self);
}

/// 计算输入张量 self 的一阶修正第一类 Bessel 函数，逐元素操作
///
/// 示例:
/// ```
/// auto t = torch::randn(128, dtype=kDouble);
/// torch::special::i1(t);
/// ```py
inline Tensor i1(const Tensor& self) {
  return torch::special_i1(self);
}

/// 计算输入张量 self 的一阶修正第一类 Bessel 函数，并将结果存入 result 张量
inline Tensor& i1_out(Tensor& result, const Tensor& self) {
  return torch::special_i1_out(result, self);
}

/// 计算输入张量 self 的指数缩放一阶修正第一类 Bessel 函数，逐元素操作
///
/// 示例:
/// ```
/// auto t = torch::randn(128, dtype=kDouble);
/// torch::special::i1e(t);
/// ```py
inline Tensor i1e(const Tensor& self) {
  return torch::special_i1e(self);
}

/// 计算输入张量 self 的指数缩放一阶修正第一类 Bessel 函数，并将结果存入 result 张量
inline Tensor& i1e_out(Tensor& result, const Tensor& self) {
  return torch::special_i1e_out(result, self);
}

/// 计算输入张量 self 的 sinc 函数值，逐元素操作
/// Computes the sinc function elementwise.
///
/// See https://pytorch.org/docs/main/special.html#torch.special.sinc.
///
/// Example:
/// ```
/// auto t = torch::randn(128, dtype=kDouble);
/// torch::special::sinc(t);
/// ```py
inline Tensor sinc(const Tensor& self) {
  // 使用 torch 库计算输入张量的 sinc 函数
  return torch::special_sinc(self);
}

/// Computes the sinc function elementwise and writes the result into the output tensor.
///
/// Example:
/// ```
/// auto t = torch::randn(128, dtype=kDouble);
/// torch::special::sinc_out(out_tensor, t);
/// ```py
inline Tensor& sinc_out(Tensor& result, const Tensor& self) {
  // 使用 torch 库计算输入张量的 sinc 函数，并将结果写入给定的输出张量
  return torch::special_sinc_out(result, self);
}

/// Rounds the elements of the input tensor to the nearest integer.
///
/// See https://pytorch.org/docs/main/special.html#torch.special.round.
///
/// Example:
/// ```
/// auto t = torch::randn(128, dtype=kDouble);
/// torch::special::round(t);
/// ```py
inline Tensor round(const Tensor& self) {
  // 使用 torch 库将输入张量的元素四舍五入到最接近的整数
  return torch::special_round(self);
}

/// Rounds the elements of the input tensor to the nearest integer and writes the result into the output tensor.
///
/// Example:
/// ```
/// auto t = torch::randn(128, dtype=kDouble);
/// torch::special::round_out(out_tensor, t);
/// ```py
inline Tensor& round_out(Tensor& result, const Tensor& self) {
  // 使用 torch 库将输入张量的元素四舍五入到最接近的整数，并将结果写入给定的输出张量
  return torch::special_round_out(result, self);
}

/// Computes log(1 + x) elementwise for the input tensor.
///
/// See https://pytorch.org/docs/main/special.html#torch.special.log1p.
///
/// Example:
/// ```
/// auto t = torch::randn(128, dtype=kDouble);
/// torch::special::log1p(t);
/// ```py
inline Tensor log1p(const Tensor& self) {
  // 使用 torch 库计算输入张量每个元素的 log(1 + x)
  return torch::special_log1p(self);
}

/// Computes log(1 + x) elementwise for the input tensor and writes the result into the output tensor.
///
/// Example:
/// ```
/// auto t = torch::randn(128, dtype=kDouble);
/// torch::special::log1p_out(out_tensor, t);
/// ```py
inline Tensor& log1p_out(Tensor& result, const Tensor& self) {
  // 使用 torch 库计算输入张量每个元素的 log(1 + x)，并将结果写入给定的输出张量
  return torch::special_log1p_out(result, self);
}

/// Computes log followed by softmax of the input tensor along a specified dimension.
///
/// See https://pytorch.org/docs/main/special.html#torch.special.log_softmax.
///
/// Example:
/// ```
/// auto t = torch::randn(128, 128, dtype=kDouble);
/// torch::special::log_softmax(t, 0);
/// ```py
inline Tensor log_softmax(
    const Tensor& self,
    int64_t dim,
    std::optional<ScalarType> dtype) {
  // 使用 torch 库计算输入张量沿指定维度的 log softmax
  return torch::special_log_softmax(self, dim, dtype);
}

/// Computes softmax of the input tensor along a specified dimension.
///
/// See https://pytorch.org/docs/main/special.html#torch.special.softmax.
///
/// Example:
/// ```
/// auto t = torch::randn(128, 128, dtype=kDouble);
/// torch::special::softmax(t, 0);
/// ```py
inline Tensor softmax(
    const Tensor& self,
    int64_t dim,
    std::optional<ScalarType> dtype) {
  // 使用 torch 库计算输入张量沿指定维度的 softmax
  return torch::special_softmax(self, dim, dtype);
}

/// Computes the Airy function Ai of the input tensor.
///
/// See https://pytorch.org/docs/main/special.html#torch.special.airy_ai.
///
/// Example:
/// ```
/// auto x = torch::randn(128, dtype=kDouble);
/// torch::special::airy_ai(x);
/// ```py
inline Tensor airy_ai(const Tensor& x) {
  // 使用 torch 库计算输入张量的 Airy 函数 Ai
  return torch::special_airy_ai(x);
}

/// Computes the Airy function Ai of the input tensor and writes the result into the output tensor.
///
/// Example:
/// ```
/// auto x = torch::randn(128, dtype=kDouble);
/// torch::special::airy_ai_out(out_tensor, x);
/// ```py
inline Tensor& airy_ai_out(Tensor& y, const Tensor& x) {
  // 使用 torch 库计算输入张量的 Airy 函数 Ai，并将结果写入给定的输出张量
  return torch::special_airy_ai_out(y, x);
}

/// Computes the Bessel function of the first kind of order 0 for the input tensor.
///
/// See https://pytorch.org/docs/main/special.html#torch.special.bessel_j0.
///
/// Example:
/// ```
/// auto x = torch::randn(128, dtype=kDouble);
/// torch::special::bessel_j0(x);
/// ```py
inline Tensor bessel_j0(const Tensor& self) {
  // 使用 torch 库计算输入张量的第一类零阶贝塞尔函数
  return torch::special_bessel_j0(self);
}

/// Computes the Bessel function of the first kind of order 0 for the input tensor and writes the result into the output tensor.
///
/// Example:
/// ```
/// auto x = torch::randn(128, dtype=kDouble);
/// torch::special::bessel_j0_out(out_tensor, x);
/// ```py
inline Tensor& bessel_j0_out(Tensor& result, const Tensor& self) {
  // 使用 torch 库计算输入张量的第一类零阶贝塞尔函数，并将结果写入给定的输出张量
  return torch::special_bessel_j0_out(result, self);
}
/// Compute the Bessel function of the first kind of order 1.
///
/// See https://pytorch.org/docs/main/special.html#torch.special.bessel_j1.
///
/// Example:
///
/// ```
/// auto x = torch::randn(128, dtype=kDouble);
///
/// torch::special::bessel_j1(x);
/// ```py
inline Tensor bessel_j1(const Tensor& self) {
  // Delegate computation to PyTorch's special_bessel_j1 function
  return torch::special_bessel_j1(self);
}

/// Compute the Bessel function of the first kind of order 1 with output tensor.
///
/// Example usage:
///
/// ```
/// auto x = torch::randn(128, dtype=kDouble);
/// torch::Tensor result;
///
/// torch::special::bessel_j1_out(result, x);
/// ```py
inline Tensor& bessel_j1_out(Tensor& result, const Tensor& self) {
  // Delegate computation to PyTorch's special_bessel_j1_out function
  return torch::special_bessel_j1_out(result, self);
}

/// Compute the Bessel function of the second kind of order 0.
///
/// See https://pytorch.org/docs/main/special.html#torch.special.bessel_y0.
///
/// Example:
///
/// ```
/// auto x = torch::randn(128, dtype=kDouble);
///
/// torch::special::bessel_y0(x);
/// ```py
inline Tensor bessel_y0(const Tensor& self) {
  // Delegate computation to PyTorch's special_bessel_y0 function
  return torch::special_bessel_y0(self);
}

/// Compute the Bessel function of the second kind of order 0 with output tensor.
///
/// Example usage:
///
/// ```
/// auto x = torch::randn(128, dtype=kDouble);
/// torch::Tensor result;
///
/// torch::special::bessel_y0_out(result, x);
/// ```py
inline Tensor& bessel_y0_out(Tensor& result, const Tensor& self) {
  // Delegate computation to PyTorch's special_bessel_y0_out function
  return torch::special_bessel_y0_out(result, self);
}

/// Compute the Bessel function of the second kind of order 1.
///
/// See https://pytorch.org/docs/main/special.html#torch.special.bessel_y1.
///
/// Example:
///
/// ```
/// auto x = torch::randn(128, dtype=kDouble);
///
/// torch::special::bessel_y1(x);
/// ```py
inline Tensor bessel_y1(const Tensor& self) {
  // Delegate computation to PyTorch's special_bessel_y1 function
  return torch::special_bessel_y1(self);
}

/// Compute the Bessel function of the second kind of order 1 with output tensor.
///
/// Example usage:
///
/// ```
/// auto x = torch::randn(128, dtype=kDouble);
/// torch::Tensor result;
///
/// torch::special::bessel_y1_out(result, x);
/// ```py
inline Tensor& bessel_y1_out(Tensor& result, const Tensor& self) {
  // Delegate computation to PyTorch's special_bessel_y1_out function
  return torch::special_bessel_y1_out(result, self);
}

/// Compute the Chebyshev polynomial of the first kind.
///
/// See https://pytorch.org/docs/main/special.html#torch.special.chebyshev_polynomial_t.
///
/// Example:
///
/// ```
/// auto x = torch::randn(128, dtype=kDouble);
/// auto n = torch::randn(128, dtype=kDouble);
///
/// torch::special::chebyshev_polynomial_t(x, n);
/// ```py
inline Tensor chebyshev_polynomial_t(const Tensor& x, const Tensor& n) {
  // Delegate computation to PyTorch's special_chebyshev_polynomial_t function
  return torch::special_chebyshev_polynomial_t(x, n);
}

/// Compute the Chebyshev polynomial of the first kind with scalar argument.
///
/// Example usage:
///
/// ```
/// auto x = torch::randn(128, dtype=kDouble);
/// auto n = 5.0;
/// 
/// torch::special::chebyshev_polynomial_t(x, n);
/// ```py
inline Tensor chebyshev_polynomial_t(const Scalar& x, const Tensor& n) {
  // Delegate computation to PyTorch's special_chebyshev_polynomial_t function
  return torch::special_chebyshev_polynomial_t(x, n);
}

/// Compute the Chebyshev polynomial of the first kind with scalar argument.
///
/// Example usage:
///
/// ```
/// auto x = torch::randn(128, dtype=kDouble);
/// auto n = 5.0;
/// 
/// torch::special::chebyshev_polynomial_t(x, n);
/// ```py
inline Tensor chebyshev_polynomial_t(const Tensor& x, const Scalar& n) {
  // Delegate computation to PyTorch's special_chebyshev_polynomial_t function
  return torch::special_chebyshev_polynomial_t(x, n);
}

/// Compute the Chebyshev polynomial of the first kind with output tensor.
///
/// Example usage:
///
/// ```
/// auto x = torch::randn(128, dtype=kDouble);
/// auto n = torch::randn(128, dtype=kDouble);
/// torch::Tensor output;
///
/// torch::special::chebyshev_polynomial_t_out(output, x, n);
/// ```py
inline Tensor& chebyshev_polynomial_t_out(
    Tensor& output,
    const Tensor& x,
    const Tensor& n) {
  // Delegate computation to PyTorch's special_chebyshev_polynomial_t_out function
  return torch::special_chebyshev_polynomial_t_out(output, x, n);
}

/// Compute the Chebyshev polynomial of the first kind with scalar argument and output tensor.
///
/// Example usage:
///
/// ```
/// auto x = torch::randn(128, dtype=kDouble);
/// auto n = 5.0;
/// torch::Tensor output;
///
/// torch::special::chebyshev_polynomial_t_out(output, x, n);
/// ```py
inline Tensor& chebyshev_polynomial_t_out(
    Tensor& output,
    const Scalar& x,
    const Tensor& n) {
  // Delegate computation to PyTorch's special_chebyshev_polynomial_t_out function
  return torch::special_chebyshev_polynomial_t_out(output, x, n);
}

/// Compute the Chebyshev polynomial of the first kind with tensor argument and scalar output tensor.
///
/// Example usage:
///
/// ```
/// auto x = torch::randn(128, dtype=kDouble);
/// auto n = 5.0;
/// torch::Tensor output;
///
/// torch::special::chebyshev_polynomial_t_out(output, x, n);
/// ```py
inline Tensor& chebyshev_polynomial_t_out(
    Tensor& output,
    const Tensor& x,
    const Scalar& n) {
  // Delegate computation to PyTorch's special_chebyshev_polynomial_t_out function
  return torch::special_chebyshev_polynomial_t_out(output, x, n);
}

/// Compute the Chebyshev polynomial of the second kind.
///
/// See https://pytorch.org/docs/main/special.html#torch.special.chebyshev_polynomial_u.
///
/// Example:
///
/// ```
/// auto x = torch::randn(128, dtype=kDouble);
/// auto n = torch::randn(128, dtype=kDouble);
///
/// torch::special::chebyshev_polynomial_u(x, n);
/// ```py
/// 使用 Torch 提供的函数计算第一类 Chebyshev 多项式 U(x, n)，返回计算结果
inline Tensor chebyshev_polynomial_u(const Tensor& x, const Tensor& n) {
  return torch::special_chebyshev_polynomial_u(x, n);
}

/// 使用 Torch 提供的函数计算第一类 Chebyshev 多项式 U(x, n)，返回计算结果
inline Tensor chebyshev_polynomial_u(const Scalar& x, const Tensor& n) {
  return torch::special_chebyshev_polynomial_u(x, n);
}

/// 使用 Torch 提供的函数计算第一类 Chebyshev 多项式 U(x, n)，返回计算结果
inline Tensor chebyshev_polynomial_u(const Tensor& x, const Scalar& n) {
  return torch::special_chebyshev_polynomial_u(x, n);
}

/// 使用 Torch 提供的函数计算第一类 Chebyshev 多项式 U(x, n)，将结果存入指定的输出张量
inline Tensor& chebyshev_polynomial_u_out(
    Tensor& output,
    const Tensor& x,
    const Tensor& n) {
  return torch::special_chebyshev_polynomial_u_out(output, x, n);
}

/// 使用 Torch 提供的函数计算第一类 Chebyshev 多项式 U(x, n)，将结果存入指定的输出张量
inline Tensor& chebyshev_polynomial_u_out(
    Tensor& output,
    const Scalar& x,
    const Tensor& n) {
  return torch::special_chebyshev_polynomial_u_out(output, x, n);
}

/// 使用 Torch 提供的函数计算第一类 Chebyshev 多项式 U(x, n)，将结果存入指定的输出张量
inline Tensor& chebyshev_polynomial_u_out(
    Tensor& output,
    const Tensor& x,
    const Scalar& n) {
  return torch::special_chebyshev_polynomial_u_out(output, x, n);
}

/// Chebyshev 多项式的第二类，计算 V(x, n)
///
/// 查看详情：https://pytorch.org/docs/main/special.html#torch.special.chebyshev_polynomial_v.
///
/// 示例：
///
/// ```
/// auto x = torch::randn(128, dtype=kDouble);
/// auto n = torch::randn(128, dtype=kDouble);
///
/// torch::special::chebyshev_polynomial_v(x, n);
/// ```py
inline Tensor chebyshev_polynomial_v(const Tensor& x, const Tensor& n) {
  return torch::special_chebyshev_polynomial_v(x, n);
}

/// Chebyshev 多项式的第二类，计算 V(x, n)
///
/// 查看详情：https://pytorch.org/docs/main/special.html#torch.special.chebyshev_polynomial_v.
///
/// 示例：
///
/// ```
/// auto x = torch::randn(128, dtype=kDouble);
/// auto n = torch::randn(128, dtype=kDouble);
///
/// torch::special::chebyshev_polynomial_v(x, n);
/// ```py
inline Tensor chebyshev_polynomial_v(const Scalar& x, const Tensor& n) {
  return torch::special_chebyshev_polynomial_v(x, n);
}

/// Chebyshev 多项式的第二类，计算 V(x, n)
///
/// 查看详情：https://pytorch.org/docs/main/special.html#torch.special.chebyshev_polynomial_v.
///
/// 示例：
///
/// ```
/// auto x = torch::randn(128, dtype=kDouble);
/// auto n = torch::randn(128, dtype=kDouble);
///
/// torch::special::chebyshev_polynomial_v(x, n);
/// ```py
inline Tensor chebyshev_polynomial_v(const Tensor& x, const Scalar& n) {
  return torch::special_chebyshev_polynomial_v(x, n);
}

/// 使用 Torch 提供的函数计算第二类 Chebyshev 多项式 V(x, n)，将结果存入指定的输出张量
inline Tensor& chebyshev_polynomial_v_out(
    Tensor& output,
    const Tensor& x,
    const Tensor& n) {
  return torch::special_chebyshev_polynomial_v_out(output, x, n);
}

/// 使用 Torch 提供的函数计算第二类 Chebyshev 多项式 V(x, n)，将结果存入指定的输出张量
inline Tensor& chebyshev_polynomial_v_out(
    Tensor& output,
    const Scalar& x,
    const Tensor& n) {
  return torch::special_chebyshev_polynomial_v_out(output, x, n);
}

/// 使用 Torch 提供的函数计算第二类 Chebyshev 多项式 V(x, n)，将结果存入指定的输出张量
inline Tensor& chebyshev_polynomial_v_out(
    Tensor& output,
    const Tensor& x,
    const Scalar& n) {
  return torch::special_chebyshev_polynomial_v_out(output, x, n);
}

/// Chebyshev 多项式的第三类，计算 W(x, n)
///
/// 查看详情：https://pytorch.org/docs/main/special.html#torch.special.chebyshev_polynomial_w.
///
/// 示例：
///
/// ```
/// auto x = torch::randn(128, dtype=kDouble);
/// auto n = torch::randn(128, dtype=kDouble);
///
/// torch::special::chebyshev_polynomial_w(x, n);
/// ```py
inline Tensor chebyshev_polynomial_w(const Tensor& x, const Tensor& n) {
  return torch::special_chebyshev_polynomial_w(x, n);
}

/// Chebyshev 多项式的第三类，计算 W(x, n)
///
/// 查看详情：https://pytorch.org/docs/main/special.html#torch.special.chebyshev_polynomial_w.
///
/// 示例：
///
/// ```
/// auto x = torch::randn(128, dtype=kDouble);
/// auto n = torch::randn(128, dtype=kDouble);
///
/// torch::special::chebyshev_polynomial_w(x, n);
/// ```py
inline Tensor chebyshev_polynomial_w(const Scalar& x, const Tensor& n) {
  return torch::special_chebyshev_polynomial_w(x, n);
}

/// Chebyshev 多项式的第三类，计算 W(x, n)
///
/// 查看详情：https://pytorch.org/docs/main/special.html#torch.special.chebyshev_polynomial_w.
///
/// 示例：
///
/// ```
/// auto x = torch::randn(128, dtype=kDouble);
/// auto n = torch::randn(128, dtype=kDouble);
///
/// torch::special::chebyshev_polynomial_w(x, n);
/// ```py
inline Tensor chebyshev_polynomial_w(const Tensor& x, const Scalar& n) {
  return torch::special_chebyshev_polynomial_w(x, n);
}

/// 使用 Torch 提供的函数计算第三类 Chebyshev 多项式 W(x, n)，将结果存入指定的输出张量
inline Tensor& chebyshev_polynomial_w_out(
    Tensor& output,
    const Tensor& x,
    const Tensor& n) {
  return torch::special_chebyshev_polynomial_w_out(output, x, n);
}

/// 使用 Torch 提供的函数计算第三类 Chebyshev 多项式 W(x, n)，将结果存入指定的输出张量
inline Tensor& chebyshev_polynomial_w_out(
    Tensor& output,
    const Scalar& x,
    const Tensor& n) {
  return torch::special_chebyshev_polynomial_w_out(output, x, n);
}

/// 使用 Torch 提供的函数计算第三类 Chebyshev 多项式 W(x, n)，将结果存入指定的输出张量
inline Tensor& chebyshev_polynomial_w_out(
    Tensor& output,
    const Tensor& x,
    const Scalar& n) {
  return torch::special_chebyshev_polynomial_w_out(output, x, n);
}
    // 调用 torch 库中的特殊切比雪夫多项式函数，计算结果存储在 output 张量中
    // 使用 x 张量作为输入，n 参数用于指定切比雪夫多项式的阶数
    return torch::special_chebyshev_polynomial_w_out(output, x, n);
/// Laguerre polynomial.
///
/// See
/// https://pytorch.org/docs/main/special.html#torch.special.laguerre_polynomial_l.
///
/// Example:
///
/// ```
/// Generate a tensor `x` filled with random numbers from a standard normal distribution.
/// ```py
/// auto x = torch::randn(128, dtype=kDouble);
/// Generate a random tensor of size 128x128 filled with double precision values.
auto n = torch::randn(128, dtype=kDouble);

/// Compute the Laguerre polynomial L(x, n).
torch::special::laguerre_polynomial_l(x, n);



/// Compute the Laguerre polynomial L(x, n) using the first tensor version.
inline Tensor laguerre_polynomial_l(const Tensor& x, const Tensor& n) {
  return torch::special_laguerre_polynomial_l(x, n);
}

/// Compute the Laguerre polynomial L(x, n) using the scalar and tensor versions.
inline Tensor laguerre_polynomial_l(const Scalar& x, const Tensor& n) {
  return torch::special_laguerre_polynomial_l(x, n);
}

/// Compute the Laguerre polynomial L(x, n) using the tensor and scalar versions.
inline Tensor laguerre_polynomial_l(const Tensor& x, const Scalar& n) {
  return torch::special_laguerre_polynomial_l(x, n);
}

/// Compute the Laguerre polynomial L(x, n) and store the result in the output tensor.
inline Tensor& laguerre_polynomial_l_out(
    Tensor& output,
    const Tensor& x,
    const Tensor& n) {
  return torch::special_laguerre_polynomial_l_out(output, x, n);
}

/// Compute the Laguerre polynomial L(x, n) using the scalar and tensor versions
/// and store the result in the output tensor.
inline Tensor& laguerre_polynomial_l_out(
    Tensor& output,
    const Scalar& x,
    const Tensor& n) {
  return torch::special_laguerre_polynomial_l_out(output, x, n);
}

/// Compute the Laguerre polynomial L(x, n) using the tensor and scalar versions
/// and store the result in the output tensor.
inline Tensor& laguerre_polynomial_l_out(
    Tensor& output,
    const Tensor& x,
    const Scalar& n) {
  return torch::special_laguerre_polynomial_l_out(output, x, n);
}



/// Legendre polynomial P(x, n).
///
/// See https://pytorch.org/docs/main/special.html#torch.special.legendre_polynomial_p.
///
/// Example:
///
/// ```
/// auto x = torch::randn(128, dtype=kDouble);
/// auto n = torch::randn(128, dtype=kDouble);
///
/// torch::special::legendre_polynomial_p(x, n);
/// ```py
inline Tensor legendre_polynomial_p(const Tensor& x, const Tensor& n) {
  return torch::special_legendre_polynomial_p(x, n);
}

/// Compute the Legendre polynomial P(x, n) using the scalar and tensor versions.
inline Tensor legendre_polynomial_p(const Scalar& x, const Tensor& n) {
  return torch::special_legendre_polynomial_p(x, n);
}

/// Compute the Legendre polynomial P(x, n) using the tensor and scalar versions.
inline Tensor legendre_polynomial_p(const Tensor& x, const Scalar& n) {
  return torch::special_legendre_polynomial_p(x, n);
}

/// Compute the Legendre polynomial P(x, n) and store the result in the output tensor.
inline Tensor& legendre_polynomial_p_out(
    Tensor& output,
    const Tensor& x,
    const Tensor& n) {
  return torch::special_legendre_polynomial_p_out(output, x, n);
}

/// Compute the Legendre polynomial P(x, n) using the scalar and tensor versions
/// and store the result in the output tensor.
inline Tensor& legendre_polynomial_p_out(
    Tensor& output,
    const Scalar& x,
    const Tensor& n) {
  return torch::special_legendre_polynomial_p_out(output, x, n);
}

/// Compute the Legendre polynomial P(x, n) using the tensor and scalar versions
/// and store the result in the output tensor.
inline Tensor& legendre_polynomial_p_out(
    Tensor& output,
    const Tensor& x,
    const Scalar& n) {
  return torch::special_legendre_polynomial_p_out(output, x, n);
}



/// Compute the modified Bessel function of the first kind of order 0.
///
/// See https://pytorch.org/docs/main/special.html#torch.special.modified_bessel_i0.
///
/// Example:
///
/// ```
/// auto x = torch::randn(128, dtype=kDouble);
///
/// torch::special::modified_bessel_i0(x);
/// ```py
inline Tensor modified_bessel_i0(const Tensor& self) {
  return torch::special_modified_bessel_i0(self);
}

/// Compute the modified Bessel function of the first kind of order 0 and store the result in the output tensor.
inline Tensor& modified_bessel_i0_out(Tensor& result, const Tensor& self) {
  return torch::special_modified_bessel_i0_out(result, self);
}

/// Compute the modified Bessel function of the first kind of order 1.
///
/// See https://pytorch.org/docs/main/special.html#torch.special.modified_bessel_i1.
///
/// Example:
///
/// ```
/// auto x = torch::randn(128, dtype=kDouble);
/// ```py
/// Calculate the modified Bessel function of the first kind of order 1 for a given tensor `self`.
inline Tensor modified_bessel_i1(const Tensor& self) {
  return torch::special_modified_bessel_i1(self);
}

/// Calculate the modified Bessel function of the first kind of order 1 and store the result in `result`.
inline Tensor& modified_bessel_i1_out(Tensor& result, const Tensor& self) {
  return torch::special_modified_bessel_i1_out(result, self);
}

/// Calculate the modified Bessel function of the second kind of order 0 for a given tensor `self`.
///
/// Example:
///
/// ```
/// auto x = torch::randn(128, dtype=kDouble);
///
/// torch::special::modified_bessel_k0(x);
/// ```py
inline Tensor modified_bessel_k0(const Tensor& self) {
  return torch::special_modified_bessel_k0(self);
}

/// Calculate the modified Bessel function of the second kind of order 0 and store the result in `result`.
inline Tensor& modified_bessel_k0_out(Tensor& result, const Tensor& self) {
  return torch::special_modified_bessel_k0_out(result, self);
}

/// Calculate the modified Bessel function of the second kind of order 1 for a given tensor `self`.
///
/// Example:
///
/// ```
/// auto x = torch::randn(128, dtype=kDouble);
///
/// torch::special::modified_bessel_k1(x);
/// ```py
inline Tensor modified_bessel_k1(const Tensor& self) {
  return torch::special_modified_bessel_k1(self);
}

/// Calculate the modified Bessel function of the second kind of order 1 and store the result in `result`.
inline Tensor& modified_bessel_k1_out(Tensor& result, const Tensor& self) {
  return torch::special_modified_bessel_k1_out(result, self);
}

/// Calculate the scaled modified Bessel function of the second kind of order 0 for a given tensor `x`.
///
/// Example:
///
/// ```
/// auto x = torch::randn(128, dtype=kDouble);
///
/// torch::special::scaled_modified_bessel_k0(x);
/// ```py
inline Tensor scaled_modified_bessel_k0(const Tensor& x) {
  return torch::special_scaled_modified_bessel_k0(x);
}

/// Calculate the scaled modified Bessel function of the second kind of order 0 and store the result in `y`.
inline Tensor& scaled_modified_bessel_k0_out(Tensor& y, const Tensor& x) {
  return torch::special_scaled_modified_bessel_k0_out(y, x);
}

/// Calculate the scaled modified Bessel function of the second kind of order 1 for a given tensor `x`.
///
/// Example:
///
/// ```
/// auto x = torch::randn(128, dtype=kDouble);
///
/// torch::special::scaled_modified_bessel_k1(x);
/// ```py
inline Tensor scaled_modified_bessel_k1(const Tensor& x) {
  return torch::special_scaled_modified_bessel_k1(x);
}

/// Calculate the scaled modified Bessel function of the second kind of order 1 and store the result in `y`.
inline Tensor& scaled_modified_bessel_k1_out(Tensor& y, const Tensor& x) {
  return torch::special_scaled_modified_bessel_k1_out(y, x);
}

/// Calculate the shifted Chebyshev polynomial of the first kind for tensors `x` and `n`.
///
/// Example:
///
/// ```
/// auto x = torch::randn(128, dtype=kDouble);
/// auto n = torch::randn(128, dtype=kDouble);
///
/// torch::special::shifted_chebyshev_polynomial_t(x, n);
/// ```py
inline Tensor shifted_chebyshev_polynomial_t(const Tensor& x, const Tensor& n) {
  return torch::special_shifted_chebyshev_polynomial_t(x, n);
}
/// 返回由特殊库函数计算的带偏移第一类切比雪夫多项式。
///
/// 使用给定的标量 x 和张量 n 计算切比雪夫多项式。
inline Tensor shifted_chebyshev_polynomial_t(const Scalar& x, const Tensor& n) {
  return torch::special_shifted_chebyshev_polynomial_t(x, n);
}

/// 返回由特殊库函数计算的带偏移第一类切比雪夫多项式。
///
/// 使用给定的张量 x 和标量 n 计算切比雪夫多项式。
inline Tensor shifted_chebyshev_polynomial_t(const Tensor& x, const Scalar& n) {
  return torch::special_shifted_chebyshev_polynomial_t(x, n);
}

/// 将带偏移第一类切比雪夫多项式计算结果写入输出张量。
///
/// 使用给定的张量 x 和 n 计算切比雪夫多项式，并将结果写入输出张量 output。
inline Tensor& shifted_chebyshev_polynomial_t_out(
    Tensor& output,
    const Tensor& x,
    const Tensor& n) {
  return torch::special_shifted_chebyshev_polynomial_t_out(output, x, n);
}

/// 将带偏移第一类切比雪夫多项式计算结果写入输出张量。
///
/// 使用给定的标量 x 和张量 n 计算切比雪夫多项式，并将结果写入输出张量 output。
inline Tensor& shifted_chebyshev_polynomial_t_out(
    Tensor& output,
    const Scalar& x,
    const Tensor& n) {
  return torch::special_shifted_chebyshev_polynomial_t_out(output, x, n);
}

/// 将带偏移第一类切比雪夫多项式计算结果写入输出张量。
///
/// 使用给定的张量 x 和标量 n 计算切比雪夫多项式，并将结果写入输出张量 output。
inline Tensor& shifted_chebyshev_polynomial_t_out(
    Tensor& output,
    const Tensor& x,
    const Scalar& n) {
  return torch::special_shifted_chebyshev_polynomial_t_out(output, x, n);
}

/// 带偏移第二类切比雪夫多项式。
///
/// 查看文档了解更多信息：https://pytorch.org/docs/main/special.html#torch.special.shifted_chebyshev_polynomial_u.
///
/// 示例：
///
/// ```
/// auto x = torch::randn(128, dtype=kDouble);
/// auto n = torch::randn(128, dtype=kDouble);
///
/// torch::special::shifted_chebyshev_polynomial_u(x, n);
/// ```py
inline Tensor shifted_chebyshev_polynomial_u(const Tensor& x, const Tensor& n) {
  return torch::special_shifted_chebyshev_polynomial_u(x, n);
}

/// 带偏移第二类切比雪夫多项式。
///
/// 使用给定的标量 x 和张量 n 计算切比雪夫多项式。
inline Tensor shifted_chebyshev_polynomial_u(const Scalar& x, const Tensor& n) {
  return torch::special_shifted_chebyshev_polynomial_u(x, n);
}

/// 带偏移第二类切比雪夫多项式。
///
/// 使用给定的张量 x 和标量 n 计算切比雪夫多项式。
inline Tensor shifted_chebyshev_polynomial_u(const Tensor& x, const Scalar& n) {
  return torch::special_shifted_chebyshev_polynomial_u(x, n);
}

/// 将带偏移第二类切比雪夫多项式计算结果写入输出张量。
///
/// 使用给定的张量 x 和 n 计算切比雪夫多项式，并将结果写入输出张量 output。
inline Tensor& shifted_chebyshev_polynomial_u_out(
    Tensor& output,
    const Tensor& x,
    const Tensor& n) {
  return torch::special_shifted_chebyshev_polynomial_u_out(output, x, n);
}

/// 将带偏移第二类切比雪夫多项式计算结果写入输出张量。
///
/// 使用给定的标量 x 和张量 n 计算切比雪夫多项式，并将结果写入输出张量 output。
inline Tensor& shifted_chebyshev_polynomial_u_out(
    Tensor& output,
    const Scalar& x,
    const Tensor& n) {
  return torch::special_shifted_chebyshev_polynomial_u_out(output, x, n);
}

/// 将带偏移第二类切比雪夫多项式计算结果写入输出张量。
///
/// 使用给定的张量 x 和标量 n 计算切比雪夫多项式，并将结果写入输出张量 output。
inline Tensor& shifted_chebyshev_polynomial_u_out(
    Tensor& output,
    const Tensor& x,
    const Scalar& n) {
  return torch::special_shifted_chebyshev_polynomial_u_out(output, x, n);
}

/// 带偏移第三类切比雪夫多项式。
///
/// 查看文档了解更多信息：https://pytorch.org/docs/main/special.html#torch.special.shifted_chebyshev_polynomial_v.
///
/// 示例：
///
/// ```
/// auto x = torch::randn(128, dtype=kDouble);
/// auto n = torch::randn(128, dtype=kDouble);
///
/// torch::special::shifted_chebyshev_polynomial_v(x, n);
/// ```py
inline Tensor shifted_chebyshev_polynomial_v(const Tensor& x, const Tensor& n) {
  return torch::special_shifted_chebyshev_polynomial_v(x, n);
}

/// 带偏移第三类切比雪夫多项式。
///
/// 使用给定的标量 x 和张量 n 计算切比雪夫多项式。
inline Tensor shifted_chebyshev_polynomial_v(const Scalar& x, const Tensor& n) {
  return torch::special_shifted_chebyshev_polynomial_v(x, n);
}
/// 使用 Torch 库计算特殊函数，返回 x 的特殊移位切比雪夫多项式 V 类型
inline Tensor shifted_chebyshev_polynomial_v(const Tensor& x, const Scalar& n) {
  return torch::special_shifted_chebyshev_polynomial_v(x, n);
}

/// 使用 Torch 库计算特殊函数，将特殊移位切比雪夫多项式 V 类型的结果输出到给定的 output 张量
inline Tensor& shifted_chebyshev_polynomial_v_out(
    Tensor& output,
    const Tensor& x,
    const Tensor& n) {
  return torch::special_shifted_chebyshev_polynomial_v_out(output, x, n);
}

/// 使用 Torch 库计算特殊函数，将特殊移位切比雪夫多项式 V 类型的结果输出到给定的 output 张量
inline Tensor& shifted_chebyshev_polynomial_v_out(
    Tensor& output,
    const Scalar& x,
    const Tensor& n) {
  return torch::special_shifted_chebyshev_polynomial_v_out(output, x, n);
}

/// 使用 Torch 库计算特殊函数，将特殊移位切比雪夫多项式 V 类型的结果输出到给定的 output 张量
inline Tensor& shifted_chebyshev_polynomial_v_out(
    Tensor& output,
    const Tensor& x,
    const Scalar& n) {
  return torch::special_shifted_chebyshev_polynomial_v_out(output, x, n);
}

/// 特殊移位切比雪夫多项式 W 类型，用于 n 和 x 的计算。
///
/// 参见 https://pytorch.org/docs/main/special.html#torch.special.shifted_chebyshev_polynomial_w.
///
/// 示例：
///
/// ```
/// auto x = torch::randn(128, dtype=kDouble);
/// auto n = torch::randn(128, dtype=kDouble);
///
/// torch::special::shifted_chebyshev_polynomial_w(x, n);
/// ```py
inline Tensor shifted_chebyshev_polynomial_w(const Tensor& x, const Tensor& n) {
  return torch::special_shifted_chebyshev_polynomial_w(x, n);
}

/// 使用 Torch 库计算特殊函数，用于 n 和 x 的计算。
inline Tensor shifted_chebyshev_polynomial_w(const Scalar& x, const Tensor& n) {
  return torch::special_shifted_chebyshev_polynomial_w(x, n);
}

/// 使用 Torch 库计算特殊函数，用于 n 和 x 的计算。
inline Tensor shifted_chebyshev_polynomial_w(const Tensor& x, const Scalar& n) {
  return torch::special_shifted_chebyshev_polynomial_w(x, n);
}

/// 球形贝塞尔函数 J0 的计算，用于 x 的计算。
///
/// 参见 https://pytorch.org/docs/main/special.html#torch.special.spherical_bessel_j0.
///
/// 示例：
///
/// ```
/// auto x = torch::randn(128, dtype=kDouble);
///
/// torch::special::spherical_bessel_j0(x);
/// ```py
inline Tensor spherical_bessel_j0(const Tensor& x) {
  return torch::special_spherical_bessel_j0(x);
}

/// 使用 Torch 库计算特殊函数，将球形贝塞尔函数 J0 的计算结果输出到给定的 y 张量。
inline Tensor& spherical_bessel_j0_out(Tensor& y, const Tensor& x) {
  return torch::special_spherical_bessel_j0_out(y, x);
}
```