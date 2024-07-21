# `.\pytorch\aten\src\ATen\native\cpu\LogAddExp.h`

```py
#pragma once

#include <c10/util/complex.h>  // 包含复数类型的头文件
#include <ATen/NumericUtils.h>  // 包含数值计算的实用函数头文件

namespace at { namespace native {
inline namespace CPU_CAPABILITY {

// 用于复数参数的logcumsumexp中的自定义最小值和最大值计算
template <typename scalar_t>
std::pair<c10::complex<scalar_t>, c10::complex<scalar_t>> _logcumsumexp_minmax(c10::complex<scalar_t> x, c10::complex<scalar_t> y) {
  if (at::_isnan(y)) {  // 如果y中有NaN，则返回一对(y, y)
    return std::make_pair(y, y);
  } else if (at::_isnan(x)) {  // 如果x中有NaN，则返回一对(x, x)
    return std::make_pair(x, x);
  } else {
    return (x.real() < y.real()) ? std::make_pair(x, y) : std::make_pair(y, x);  // 返回最小和最大值对
  }
}

// 辅助函数，用于计算log(exp(x) + exp(y))，处理标量参数
template <typename scalar_t>
scalar_t _log_add_exp_helper(scalar_t x, scalar_t y) {
  // 参考资料：https://www.tensorflow.org/api_docs/python/tf/math/cumulative_logsumexp
  scalar_t min = at::_isnan(y) ? y : std::min(x, y);  // 计算最小值，处理NaN情况
  scalar_t max = at::_isnan(y) ? y : std::max(x, y);  // 计算最大值，处理NaN情况
  if (min != max || std::isfinite(min)) {
    // 处理正常情况和NaN传播
    return std::log1p(std::exp(min - max)) + max;
  } else {
    // 处理特殊情况以正确处理无穷大的情况
    return x;
  }
}

// 辅助函数，用于计算log(exp(x) + exp(y))，处理复数参数
template <typename scalar_t>
c10::complex<scalar_t> _log_add_exp_helper(const c10::complex<scalar_t>& x, const c10::complex<scalar_t>& y) {
  auto [min, max] = _logcumsumexp_minmax<scalar_t>(x, y);  // 计算复数的最小和最大值
  auto min_real = std::real(min);
  auto max_real = std::real(max);

  if (at::_isnan(min)) {  // 处理"传染性"NaN
    return {std::numeric_limits<scalar_t>::quiet_NaN(), std::numeric_limits<scalar_t>::quiet_NaN()};
  } else if (!std::isfinite(min_real) && (min_real == max_real)) {
    if (min_real < 0) {
      // 处理-inf情况，虚部无关紧要，因为exp(值)将接近0.0
      return min;
    } else {
      // 处理+inf情况，避免在real(max) == real(min) == +inf时产生NaN
      return std::log(std::exp(min) + std::exp(max));
    }
  } else {
    return std::log1p(std::exp(min - max)) + max;
  }
}

} // end namespace
}} //end at::native
```