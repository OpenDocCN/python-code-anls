# `.\pytorch\torch\csrc\api\include\torch\fft.h`

```
#pragma once

#include <ATen/ATen.h>

namespace torch {
namespace fft {

/// 计算给定维度上的一维快速傅里叶变换。
/// 参见 https://pytorch.org/docs/main/fft.html#torch.fft.fft。
///
/// 示例：
/// ```
/// auto t = torch::randn(128, dtype=kComplexDouble);
/// torch::fft::fft(t);
/// ```
inline Tensor fft(
    const Tensor& self,
    std::optional<SymInt> n = c10::nullopt,
    int64_t dim = -1,
    std::optional<c10::string_view> norm = c10::nullopt) {
  return torch::fft_fft_symint(self, n, dim, norm);
}

/// 计算给定维度上的一维逆傅里叶变换。
/// 参见 https://pytorch.org/docs/main/fft.html#torch.fft.ifft。
///
/// 示例：
/// ```
/// auto t = torch::randn(128, dtype=kComplexDouble);
/// torch::fft::ifft(t);
/// ```
inline Tensor ifft(
    const Tensor& self,
    std::optional<SymInt> n = c10::nullopt,
    int64_t dim = -1,
    std::optional<c10::string_view> norm = c10::nullopt) {
  return torch::fft_ifft_symint(self, n, dim, norm);
}

/// 计算给定维度上的二维快速傅里叶变换。
/// 参见 https://pytorch.org/docs/main/fft.html#torch.fft.fft2。
///
/// 示例：
/// ```
/// auto t = torch::randn({128, 128}, dtype=kComplexDouble);
/// torch::fft::fft2(t);
/// ```
inline Tensor fft2(
    const Tensor& self,
    OptionalIntArrayRef s = c10::nullopt,
    IntArrayRef dim = {-2, -1},
    std::optional<c10::string_view> norm = c10::nullopt) {
  return torch::fft_fft2(self, s, dim, norm);
}

/// 计算给定维度上的二维逆傅里叶变换。
/// 参见 https://pytorch.org/docs/main/fft.html#torch.fft.ifft2。
///
/// 示例：
/// ```
/// auto t = torch::randn({128, 128}, dtype=kComplexDouble);
/// torch::fft::ifft2(t);
/// ```
inline Tensor ifft2(
    const Tensor& self,
    at::OptionalIntArrayRef s = c10::nullopt,
    IntArrayRef dim = {-2, -1},
    std::optional<c10::string_view> norm = c10::nullopt) {
  return torch::fft_ifft2(self, s, dim, norm);
}

/// 计算给定维度上的N维快速傅里叶变换。
/// 参见 https://pytorch.org/docs/main/fft.html#torch.fft.fftn。
///
/// 示例：
/// ```
/// auto t = torch::randn({128, 128}, dtype=kComplexDouble);
/// torch::fft::fftn(t);
/// ```
inline Tensor fftn(
    const Tensor& self,
    at::OptionalIntArrayRef s = c10::nullopt,
    at::OptionalIntArrayRef dim = c10::nullopt,
    std::optional<c10::string_view> norm = c10::nullopt) {
  return torch::fft_fftn(self, s, dim, norm);
}

/// 计算给定维度上的N维逆傅里叶变换。
/// 参见 https://pytorch.org/docs/main/fft.html#torch.fft.ifftn。
///
/// 示例：
/// ```
/// auto t = torch::randn({128, 128}, dtype=kComplexDouble);
/// torch::fft::ifftn(t);
/// ```
inline Tensor ifftn(
    const Tensor& self,
    at::OptionalIntArrayRef s = c10::nullopt,
    at::OptionalIntArrayRef dim = c10::nullopt,
    std::optional<c10::string_view> norm = c10::nullopt) {
  return torch::fft_ifftn(self, s, dim, norm);
}

} // namespace fft
} // namespace torch
/// Computes the 1 dimensional FFT of real input with onesided Hermitian output.
/// See https://pytorch.org/docs/main/fft.html#torch.fft.rfft.
///
/// Example:
/// ```
/// auto t = torch::randn(128);
/// auto T = torch::fft::rfft(t);
/// assert(T.is_complex() && T.numel() == 128 / 2 + 1);
/// ```
inline Tensor rfft(
    const Tensor& self,
    std::optional<SymInt> n = c10::nullopt,   // Optional parameter for FFT size
    int64_t dim = -1,                         // Dimension along which to compute FFT
    std::optional<c10::string_view> norm = c10::nullopt) {  // Optional normalization mode
  return torch::fft_rfft_symint(self, n, dim, norm);   // Call the FFT function with specified parameters
}

/// Computes the inverse of torch.fft.rfft
///
/// The input is a onesided Hermitian Fourier domain signal, with real-valued
/// output. See https://pytorch.org/docs/main/fft.html#torch.fft.irfft
///
/// Example:
/// ```
/// auto T = torch::randn(128 / 2 + 1, torch::kComplexDouble);
/// auto t = torch::fft::irfft(t, /*n=*/128);
/// assert(t.is_floating_point() && T.numel() == 128);
/// ```
inline Tensor irfft(
    const Tensor& self,
    std::optional<SymInt> n = c10::nullopt,   // Optional parameter for output size
    int64_t dim = -1,                         // Dimension along which to compute inverse FFT
    std::optional<c10::string_view> norm = c10::nullopt) {  // Optional normalization mode
  return torch::fft_irfft_symint(self, n, dim, norm);   // Call the inverse FFT function with specified parameters
}

/// Computes the 2-dimensional FFT of real input. Returns a onesided Hermitian
/// output. See https://pytorch.org/docs/main/fft.html#torch.fft.rfft2
///
/// Example:
/// ```
/// auto t = torch::randn({128, 128}, dtype=kDouble);
/// torch::fft::rfft2(t);
/// ```
inline Tensor rfft2(
    const Tensor& self,
    at::OptionalIntArrayRef s = c10::nullopt,   // Optional parameter for FFT size
    IntArrayRef dim = {-2, -1},                 // Dimensions along which to compute FFT
    std::optional<c10::string_view> norm = c10::nullopt) {  // Optional normalization mode
  return torch::fft_rfft2(self, s, dim, norm);   // Call the 2D FFT function with specified parameters
}

/// Computes the inverse of torch.fft.rfft2.
/// See https://pytorch.org/docs/main/fft.html#torch.fft.irfft2.
///
/// Example:
/// ```
/// auto t = torch::randn({128, 128}, dtype=kComplexDouble);
/// torch::fft::irfft2(t);
/// ```
inline Tensor irfft2(
    const Tensor& self,
    at::OptionalIntArrayRef s = c10::nullopt,   // Optional parameter for output size
    IntArrayRef dim = {-2, -1},                 // Dimensions along which to compute inverse FFT
    std::optional<c10::string_view> norm = c10::nullopt) {  // Optional normalization mode
  return torch::fft_irfft2(self, s, dim, norm);   // Call the inverse 2D FFT function with specified parameters
}

/// Computes the N dimensional FFT of real input with onesided Hermitian output.
/// See https://pytorch.org/docs/main/fft.html#torch.fft.rfftn
///
/// Example:
/// ```
/// auto t = torch::randn({128, 128}, dtype=kDouble);
/// torch::fft::rfftn(t);
/// ```
inline Tensor rfftn(
    const Tensor& self,
    at::OptionalIntArrayRef s = c10::nullopt,   // Optional parameter for FFT size
    at::OptionalIntArrayRef dim = c10::nullopt, // Optional dimensions along which to compute FFT
    std::optional<c10::string_view> norm = c10::nullopt) {  // Optional normalization mode
  return torch::fft_rfftn(self, s, dim, norm);   // Call the N-dimensional FFT function with specified parameters
}

/// Computes the inverse of torch.fft.rfftn.
/// See https://pytorch.org/docs/main/fft.html#torch.fft.irfftn.
///
/// Example:
/// ```
/// auto t = torch::randn({128, 128}, dtype=kComplexDouble);
/// torch::fft::irfftn(t);
/// ```
inline Tensor irfftn(
    const Tensor& self,
    at::OptionalIntArrayRef s = c10::nullopt,   // Optional parameter for output size
    at::OptionalIntArrayRef dim = c10::nullopt, // Optional dimensions along which to compute inverse FFT
    std::optional<c10::string_view> norm = c10::nullopt) {  // Optional normalization mode
  return torch::fft_irfftn(self, s, dim, norm);   // Call the inverse N-dimensional FFT function with specified parameters
}
    // 使用 torch 库中的 fft_irfftn 函数进行反 Fourier 变换
    std::optional<c10::string_view> norm = c10::nullopt) {
    // 调用 fft_irfftn 函数，并返回其结果
    return torch::fft_irfftn(self, s, dim, norm);
/// Computes the N-dimensional FFT of a Hermitian symmetric input signal.
///
/// The input tensor `self` represents a Hermitian symmetric time domain signal
/// in the Fourier domain. The returned tensor is its N-dimensional Fourier
/// transform. See https://pytorch.org/docs/main/fft.html#torch.fft.hfftn.
///
/// Example:
/// ```
/// auto t = torch::randn({128, 65}, torch::kComplexDouble);
/// auto T = torch::fft::hfftn(t, /*s=*/{128, 128});
/// assert(T.is_floating_point() && T.numel() == 128 * 128);
/// ```
inline Tensor hfftn(
    const Tensor& self,
    at::OptionalIntArrayRef s = c10::nullopt,
    std::optional<c10::string_view> norm = c10::nullopt) {
  return torch::fft_hfftn(self, s, norm);
}
/// Computes the N-dimensional FFT of a real input signal with half-complex output.
/// This function performs the FFT along specified dimensions.
/// 
/// Example:
/// ```
/// auto T = torch::randn({128, 128}, torch::kDouble);
/// auto t = torch::fft::hfftn(T);
/// assert(t.is_complex() && t.size(1) == 65);
/// ```
inline Tensor hfftn(
    const Tensor& self,
    at::OptionalIntArrayRef s = c10::nullopt,   // Optional size of the FFT
    IntArrayRef dim = {-2, -1},                 // Dimensions along which to compute FFT
    std::optional<c10::string_view> norm = c10::nullopt) {  // Optional normalization
  return torch::fft_hfftn(self, s, dim, norm);
}

/// Computes the inverse N-dimensional FFT of a half-complex input signal.
/// This function computes the inverse FFT along specified dimensions.
///
/// Example:
/// ```
/// auto T = torch::randn({128, 128}, torch::kDouble);
/// auto t = torch::fft::ihfftn(T);
/// assert(t.is_real() && t.size(1) == 128);
/// ```
inline Tensor ihfftn(
    const Tensor& self,
    at::OptionalIntArrayRef s = c10::nullopt,   // Optional size of the FFT
    IntArrayRef dim = {-2, -1},                 // Dimensions along which to compute inverse FFT
    std::optional<c10::string_view> norm = c10::nullopt) {  // Optional normalization
  return torch::fft_ihfftn(self, s, dim, norm);
}

/// Computes the discrete Fourier Transform sample frequencies for a signal of size n.
///
/// Example:
/// ```
/// auto frequencies = torch::fft::fftfreq(128, torch::kDouble);
/// ```
inline Tensor fftfreq(int64_t n, double d, const TensorOptions& options = {}) {
  return torch::fft_fftfreq(n, d, options);
}

/// Computes the discrete Fourier Transform sample frequencies for a signal of size n
/// with default spacing of 1.0.
///
/// Example:
/// ```
/// auto frequencies = torch::fft::fftfreq(128);
/// ```
inline Tensor fftfreq(int64_t n, const TensorOptions& options = {}) {
  return torch::fft_fftfreq(n, /*d=*/1.0, options);
}

/// Computes the sample frequencies for torch.fft.rfft with a signal of size n.
///
/// Example:
/// ```
/// auto frequencies = torch::fft::rfftfreq(128, torch::kDouble);
/// ```
inline Tensor rfftfreq(int64_t n, double d, const TensorOptions& options) {
  return torch::fft_rfftfreq(n, d, options);
}

/// Computes the sample frequencies for torch.fft.rfft with a signal of size n
/// with default spacing of 1.0.
///
/// Example:
/// ```
/// auto frequencies = torch::fft::rfftfreq(128);
/// ```
inline Tensor rfftfreq(int64_t n, const TensorOptions& options) {
  return torch::fft_rfftfreq(n, /*d=*/1.0, options);
}

/// Reorders n-dimensional FFT output to have negative frequency terms first, by
/// performing a torch.roll operation.
///
/// Example:
/// ```
/// auto x = torch::randn({127, 4});
/// auto centred_fft = torch::fft::fftshift(torch::fft::fftn(x));
/// ```
inline Tensor fftshift(
    const Tensor& x,
    at::OptionalIntArrayRef dim = c10::nullopt) {   // Dimensions along which to perform fftshift
  return torch::fft_fftshift(x, dim);
}

/// Inverse operation of torch.fft.fftshift, restores the original order of shifted data.
///
/// Example:
/// ```
/// auto x = torch::randn({127, 4});
/// auto shift = torch::fft::fftshift(x);
/// auto unshift = torch::fft::ifftshift(shift);
/// assert(torch::allclose(x, unshift));
/// ```
inline Tensor ifftshift(
    const Tensor& x,
    at::OptionalIntArrayRef dim = c10::nullopt) {   // Dimensions along which to perform ifftshift
  return torch::fft_ifftshift(x, dim);
}
```