# `.\pytorch\aten\src\ATen\native\Math.h`

```
#pragma once

#include <ATen/AccumulateType.h>  // 引入 ATen 库中的 AccumulateType 类
#include <ATen/NumericUtils.h>    // 引入 ATen 库中的 NumericUtils 类
#include <ATen/jiterator_macros.h>    // 引入 ATen 库中的 jiterator 宏定义
#include <c10/util/BFloat16.h>    // 引入 c10 库中的 BFloat16 类
#include <c10/util/Half.h>        // 引入 c10 库中的 Half 类
#include <c10/util/MathConstants.h>   // 引入 c10 库中的 MathConstants 常量定义
#include <cfloat>                 // 引入 C 标准库中的浮点数常量定义
#include <cmath>                  // 引入 C 标准库中的数学函数
#include <cstdint>                // 引入 C 标准库中的整数类型定义
#include <cstdlib>                // 引入 C 标准库中的通用工具函数
#include <limits>                 // 引入 C++ 标准库中的数值极限定义
#include <type_traits>            // 引入 C++ 标准库中的类型特性定义

C10_CLANG_DIAGNOSTIC_PUSH()       // 压栈保存当前编译器诊断设置
#if C10_CLANG_HAS_WARNING("-Wimplicit-float-conversion")
C10_CLANG_DIAGNOSTIC_IGNORE("-Wimplicit-float-conversion")  // 忽略隐式浮点数转换警告
#endif

/* The next function is taken from  https://github.com/antelopeusersgroup/antelope_contrib/blob/master/lib/location/libgenloc/erfinv.c.
Below is the copyright.
Output was modified to be inf or -inf when input is 1 or -1. */

/*
    Copyright (c) 2014 Indiana University
    All rights reserved.

    Written by Prof. Gary L. Pavlis, Dept. of Geol. Sci.,
            Indiana University, Bloomington, IN

    This software is licensed under the New BSD license:

    Redistribution and use in source and binary forms,
    with or without modification, are permitted provided
    that the following conditions are met:

    Redistributions of source code must retain the above
    copyright notice, this list of conditions and the
    following disclaimer.

    Redistributions in binary form must reproduce the
    above copyright notice, this list of conditions and
    the following disclaimer in the documentation and/or
    other materials provided with the distribution.

    Neither the name of Indiana University nor
    the names of its contributors may be used to endorse
    or promote products derived from this software without
    specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
    CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED
    WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
    PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL
    THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY
    DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
    CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
    PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
    USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
    HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
    IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
    NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
    USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
    POSSIBILITY OF SUCH DAMAGE.
*/

namespace {
/*
 * This function is derived from the implementation of the i0e function in the
 * Cephes Math Library. See note [3-Clause BSD License for the Cephes Math
 * Library].
 *
 * Computes an approximation of the exponentially scaled zeroth order modified
 * Bessel function of the first kind. The approximation is actually two
 * (sub)approximations, both using a Chebyshev polynomial expansion. One
 * approximates the function over [0, 8], and the other over (8, infinity). This
 * function takes the absolute value of all inputs to convert them into the
 * domain of the approximation.
 */
jiterator_also_stringify_as(jiterator_code(
  template <typename T>
  JITERATOR_HOST_DEVICE T chbevl(T x, const T array[], const int len) {
    // 定义三个变量 b0, b1, b2 用于存储多项式展开的系数
    T b0, b1, b2;

    // 初始化 b0 为系数数组的第一个元素
    b0 = array[0];
    // 初始化 b1 为0
    b1 = 0;

    // 循环计算多项式展开
    for (int i = 1; i < len; ++i) {
      // 保存上一次的 b1 到 b2
      b2 = b1;
      // 计算当前的 b0，更新为 x*b1 - b2 + array[i]
      b0 = x * b1 - b2 + array[i];
      // 更新 b1 为当前的 b0
      b1 = b0;
    }

    // 返回多项式的计算结果
    return T{0.5} * (b0 - b2);
  }

  template <typename T>
  JITERATOR_HOST_DEVICE T calc_i0e(T _x) {
    // 对输入取绝对值
    T x = std::fabs(_x);

    // 如果 x 小于等于 8.0
    if (x <= T{8.0}) {
      // 定义静态的系数数组，用于 [0, 8] 区间的多项式展开
      static const T coefficients[] = {
          -4.41534164647933937950E-18, 3.33079451882223809783E-17,
          -2.43127984654795469359E-16, 1.71539128555513303061E-15,
          -1.16853328779934516808E-14, 7.67618549860493561688E-14,
          -4.85644678311192946090E-13, 2.95505266312963983461E-12,
          -1.72682629144155570723E-11, 9.67580903537323691224E-11,
          -5.18979560163526290666E-10, 2.65982372468238665035E-9,
          -1.30002500998624804212E-8,  6.04699502254191894932E-8,
          -2.67079385394061173391E-7,  1.11738753912010371815E-6,
          -4.41673835845875056359E-6,  1.64484480707288970893E-5,
          -5.75419501008210370398E-5,  1.88502885095841655729E-4,
          -5.76375574538582365885E-4,  1.63947561694133579842E-3,
          -4.32430999505057594430E-3,  1.05464603945949983183E-2,
          -2.37374148058994688156E-2,  4.93052842396707084878E-2,
          -9.49010970480476444210E-2,  1.71620901522208775349E-1,
          -3.04682672343198398683E-1,  6.76795274409476084995E-1};

      // 计算 y = (x / 2.0) - 2.0
      T y = (x / T{2.0}) - T{2.0};
      // 使用 chbevl 函数计算多项式展开的结果，系数为 coefficients，长度为 30
      return chbevl(y, coefficients, int{30});
    }

    // 如果 x 大于 8，这部分未完整展示

/*
 * This function is derived from the implementation of the i0e function in the
 * Cephes Math Library. See note [3-Clause BSD License for the Cephes Math
 * Library].
 *
 * Computes an approximation of the exponentially scaled zeroth order modified
 * Bessel function of the first kind. The approximation is actually two
 * (sub)approximations, both using a Chebyshev polynomial expansion. One
 * approximates the function over [0, 8], and the other over (8, infinity). This
 * function takes the absolute value of all inputs to convert them into the
 * domain of the approximation.
 */
jiterator_also_stringify_as(jiterator_code(
  template <typename T>
  JITERATOR_HOST_DEVICE T chbevl(T x, const T array[], const int len) {
    // 定义三个变量 b0, b1, b2 用于存储多项式展开的系数
    T b0, b1, b2;

    // 初始化 b0 为系数数组的第一个元素
    b0 = array[0];
    // 初始化 b1 为0
    b1 = 0;

    // 循环计算多项式展开
    for (int i = 1; i < len; ++i) {
      // 保存上一次的 b1 到 b2
      b2 = b1;
      // 计算当前的 b0，更新为 x*b1 - b2 + array[i]
      b0 = x * b1 - b2 + array[i];
      // 更新 b1 为当前的 b0
      b1 = b0;
    }

    // 返回多项式的计算结果
    return T{0.5} * (b0 - b2);
  }

  template <typename T>
  JITERATOR_HOST_DEVICE T calc_i0e(T _x) {
    // 对输入取绝对值
    T x = std::fabs(_x);

    // 如果 x 小于等于 8.0
    if (x <= T{8.0}) {
      // 定义静态的系数数组，用于 [0, 8] 区间的多项式展开
      static const T coefficients[] = {
          -4.41534164647933937950E-18, 3.33079451882223809783E-17,
          -2.43127984654795469359E-16, 1.71539128555513303061E-15,
          -1.16853328779934516808E-14, 7.67618549860493561688E-14,
          -4.85644678311192946090E-13, 2.95505266312963983461E-12,
          -1.72682629144155570723E-11, 9.67580903537323691224E-11,
          -5.18979560163526290666E-10, 2.65982372468238665035E-9,
          -1.30002500998624804212E-8,  6.04699502254191894932E-8,
          -2.67079385394061173391E-7,  1.11738753912010371815E-6,
          -4.41673835845875056359E-6,  1.64484480707288970893E-5,
          -5.75419501008210370398E-5,  1.88502885095841655729E-4,
          -5.76375574538582365885E-4,  1.63947561694133579842E-3,
          -4.32430999505057594430E-3,  1.05464603945949983183E-2,
          -2.37374148058994688156E-2,  4.93052842396707084878E-2,
          -9.49010970480476444210E-2,  1.71620901522208775349E-1,
          -3.04682672343198398683E-1,  6.76795274409476084995E-1};

      // 计算 y = (x / 2.0) - 2.0
      T y = (x / T{2.0}) - T{2.0};
      // 使用 chbevl 函数计算多项式展开的结果，系数为 coefficients，长度为 30
      return chbevl(y, coefficients, int{30});
    }

    // 如果 x 大于 8，继续计算另一区间的逼近，此处未完整展示
    // ...
    // 定义静态常量数组 coefficients，存储了 Chebyshev 多项式的系数
    static const T coefficients[] = {
        -7.23318048787475395456E-18, -4.83050448594418207126E-18,
        4.46562142029675999901E-17,  3.46122286769746109310E-17,
        -2.82762398051658348494E-16, -3.42548561967721913462E-16,
        1.77256013305652638360E-15,  3.81168066935262242075E-15,
        -9.55484669882830764870E-15, -4.15056934728722208663E-14,
        1.54008621752140982691E-14,  3.85277838274214270114E-13,
        7.18012445138366623367E-13,  -1.79417853150680611778E-12,
        -1.32158118404477131188E-11, -3.14991652796324136454E-11,
        1.18891471078464383424E-11,  4.94060238822496958910E-10,
        3.39623202570838634515E-9,   2.26666899049817806459E-8,
        2.04891858946906374183E-7,   2.89137052083475648297E-6,
        6.88975834691682398426E-5,   3.36911647825569408990E-3,
        8.04490411014108831608E-1};
    
    // 返回经过 Chebyshev 多项式求值得到的函数值，输入参数为 T{32.0} / x - T{2.0}
    // 使用 coefficients 数组作为多项式的系数，25 表示系数数组的长度
    return chbevl(T{32.0} / x - T{2.0}, coefficients, int{25}) / std::sqrt(x);
    
    
    注释说明了代码中的静态常量数组 `coefficients` 存储了一个 Chebyshev 多项式的系数，并且函数 `chbevl` 被调用来计算给定参数下的多项式值，最后结果被除以 `sqrt(x)` 返回。
/* 定义一个常量，表示中心范围的比例 */
#define CENTRAL_RANGE 0.7

/* 模板函数：计算反误差函数的逆函数。使用有理逼近法进行初始近似，
   然后通过两步牛顿法将其精确到完全精度。代码是对 Matlab 版本 2.0 中
   erfinv.m 文件的直接翻译。
   作者：Gary L. Pavlis, Indiana University
   日期：1996年2月 */
template <typename T>
inline typename std::enable_if<std::is_floating_point<T>::value, T>::type
calc_erfinv(T y) {
    T x, z, num, dem; /* 工作变量 */
    /* 有理展开的系数 */
    T a[4] = { T(0.886226899), T(-1.645349621), T(0.914624893), T(-0.140543331) };
    T b[4] = { T(-2.118377725), T(1.442710462), T(-0.329097515), T(0.012229801) };
    T c[4] = { T(-1.970840454), T(-1.624906493), T(3.429567803), T(1.641345311) };
    T d[2] = { T(3.543889200), T(1.637067800) };
    T y_abs = std::abs(y);
    
    /* 如果输入值的绝对值大于1，则返回 NaN */
    if (y_abs > 1.0) return std::numeric_limits<T>::quiet_NaN();
    
    /* 如果输入值的绝对值等于1，则根据平台返回正无穷或负无穷 */
#ifdef _WIN32
    // 在 Windows 平台上，_copysign 不属于 std 命名空间
    if (y_abs == 1.0) return copysign(std::numeric_limits<T>::infinity(), y);
#else
    if (y_abs == 1.0) return std::copysign(std::numeric_limits<T>::infinity(), y);
#endif
    
    /* 如果输入值的绝对值在中心范围内，则使用有理逼近法计算 */
    if (y_abs <= static_cast<T>(CENTRAL_RANGE)) {
        z = y * y;
        num = (((a[3]*z + a[2])*z + a[1])*z + a[0]);
        dem = ((((b[3]*z + b[2])*z + b[1])*z + b[0]) * z + static_cast<T>(1.0));
        x = y * num / dem;
    }
    /* 如果输入值的绝对值超出中心范围，则使用对数和平方根计算 */
    else {
        z = std::sqrt(-std::log((static_cast<T>(1.0)-y_abs)/static_cast<T>(2.0)));
        num = ((c[3]*z + c[2])*z + c[1]) * z + c[0];
        dem = (d[1]*z + d[0])*z + static_cast<T>(1.0);
#ifdef _WIN32
        // 在 Windows 平台上，_copysign 不属于 std 命名空间
        x = copysign(num, y) / dem;
#else
        x = std::copysign(num, y) / dem;
#endif
    }
    
    /* 通过两步牛顿拉弗逊法进行进一步修正 */
    x = x - (std::erf(x) - y) / ((static_cast<T>(2.0)/static_cast<T>(std::sqrt(c10::pi<double>)))*std::exp(-x*x));
    x = x - (std::erf(x) - y) / ((static_cast<T>(2.0)/static_cast<T>(std::sqrt(c10::pi<double>)))*std::exp(-x*x));

    /* 返回计算结果 */
    return x;
}

/* 取消定义 CENTRAL_RANGE 常量 */
#undef CENTRAL_RANGE
/*
 * Note [3-Clause BSD License for the Cephes Math Library]
 * Code derived from implementations in the Cephes Math Library should mention its derivation and reference
 * this note (ex. 'This function is derived from the implementation of X in the Cephes Math Library. See note
 * [3-Clause BSD License for the Cephes Math Library]. The license is:
 * Copyright (c) 2018, Steven Moshier
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * * Redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer.
 * * Redistributions in binary form must reproduce the above copyright
 * notice, this list of conditions and the following disclaimer in the
 * documentation and/or other materials provided with the distribution.
 * * Neither the name of the nor the
 * names of its contributors may be used to endorse or promote products
 * derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL Steven Moshier BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/*
 * This function is derived from the implementation of the zeta function in the Cephes Math Library.
 * See note [3-Clause BSD License for the Cephes Math Library].
 */
template <typename scalar_t, bool is_cuda=false>
C10_HOST_DEVICE inline scalar_t zeta(scalar_t x, scalar_t q) __ubsan_ignore_float_divide_by_zero__ {
  using acc_t = at::acc_type<scalar_t, is_cuda>;
  定义精度类型 acc_t 为标量类型 scalar_t 的累加类型
  定义常量 MACHEP 表示机器精度
  constexpr acc_t zero = acc_t{0.0};
  constexpr acc_t half = acc_t{0.5};
  constexpr acc_t one = acc_t{1.0};
  static const acc_t A[] = {
      12.0,
      -720.0,
      30240.0,
      -1209600.0,
      47900160.0,
      -1.8924375803183791606e9, /*1.307674368e12/691*/
      7.47242496e10,
      -2.950130727918164224e12, /*1.067062284288e16/3617*/
      1.1646782814350067249e14, /*5.109094217170944e18/43867*/
      -4.5979787224074726105e15, /*8.028576626982912e20/174611*/
      1.8152105401943546773e17, /*1.5511210043330985984e23/854513*/
      -7.1661652561756670113e18 /*1.6938241367317436694528e27/236364091*/
  };

  int i = 0;
  acc_t a, b, k, s, t, w;
  如果 x 等于 one，执行以下操作
*/
  # 返回一个表示正无穷大的值，用于标识计算结果为正无穷的情况
  return std::numeric_limits<scalar_t>::infinity();
}

if (x < one) {
  # 如果输入值 x 小于 1，返回一个表示静默 NaN 的值，用于标识计算结果不是数字的情况
  return std::numeric_limits<scalar_t>::quiet_NaN();
}

if (q <= zero) {
  # 如果 q 小于等于 0
  if (q == std::floor(q)) {
    # 如果 q 是整数，则返回一个表示正无穷大的值
    return std::numeric_limits<scalar_t>::infinity();
  }
  if (x != std::floor(x)) {
    # 如果 x 不是整数，则返回一个表示静默 NaN 的值
    return std::numeric_limits<scalar_t>::quiet_NaN();
  }
}

s = std::pow(q, -x);  # 计算 q 的 -x 次幂并赋值给 s
a = q;  # 将 q 赋值给 a
i = 0;  # 初始化循环计数器 i
b = zero;  # 将零赋值给 b
while ((i < 9) || (a <= acc_t{9.0})) {
  # 循环执行以下操作，直到 i 大于等于 9 或者 a 大于等于 9.0
  i += 1;  # i 加一
  a += one;  # a 加一
  b = ::pow(a, -x);  # 计算 a 的 -x 次幂并赋值给 b
  s += b;  # s 加上 b
  if ((-MACHEP * s < b) && (b < MACHEP * s)) {
    # 如果 b 的绝对值小于等于 MACHEP * s，则返回 s 的值
    return static_cast<scalar_t>(s);
  }
};

w = a;  # 将 a 的值赋给 w
s += b * w / (x - one);  # 计算 s + b * w / (x - 1) 并赋值给 s
s -= half * b;  # s 减去 half * b 的值并赋值给 s
a = one;  # 将一赋值给 a
k = zero;  # 将零赋值给 k
for (int i = 0; i < 12; i++) {
  # 循环执行以下操作，直到 i 大于等于 12
  a *= x + k;  # a 乘以 (x + k)
  b /= w;  # b 除以 w
  t = a * b / A[i];  # 计算 a * b / A[i] 并赋值给 t
  s = s + t;  # s 加上 t 的值并赋值给 s
  t = ::fabs(t / s);  # 计算 t / s 的绝对值并赋值给 t
  if (t < MACHEP) {
    # 如果 t 小于 MACHEP，则返回 s 的值
    return static_cast<scalar_t>(s);
  }
  k += one;  # k 加一
  a *= x + k;  # a 乘以 (x + k)
  b /= w;  # b 除以 w
  k += one;  # k 加一
}
return static_cast<scalar_t>(s);  # 返回 s 的值作为计算结果
/*
 * This function evaluates a polynomial of degree N using the Horner's method.
 * It computes:
 *
 *     y = C0 + C1*x + C2*x^2 + ... + CN*x^N
 *
 * The coefficients A[] are stored in reverse order: A[0] = CN, ..., A[N] = C0.
 *
 * @tparam T The type of input and coefficients (usually float or double).
 * @param x The input value to evaluate the polynomial at.
 * @param A An array of coefficients in reverse order.
 * @param len The length of the coefficient array.
 * @return The evaluated result of the polynomial.
 */
template <typename T>
C10_HOST_DEVICE inline T polevl(const T x, const T A[], size_t len) {
  T result = 0;
  for (size_t i = 0; i <= len; i++) {
    result = result * x + A[i];
  }
  return result;
}

/*
 * This function computes the trigamma function for double precision floating point numbers.
 * It includes handling for values where x < 0.5 and utilizes a polynomial approximation for other cases.
 * Special handling is applied for division by zero errors.
 *
 * @param x The input value for which the trigamma function is computed.
 * @return The computed trigamma value for the input x.
 */
inline double trigamma(double x) __ubsan_ignore_float_divide_by_zero__ {
  double sign = +1;
  double result = 0;
  if (x < 0.5) {
    sign = -1;
    const double sin_pi_x = sin(c10::pi<double> * x);
    result -= (c10::pi<double> * c10::pi<double>) / (sin_pi_x * sin_pi_x);
    x = 1 - x;
  }
  for (int i = 0; i < 6; ++i) {
    result += 1 / (x * x);
    x += 1;
  }
  const double ixx = 1 / (x*x);
  result += (1 + 1 / (2*x) + ixx * (1./6 - ixx * (1./30 - ixx * (1./42)))) / x;
  return sign * result;
}

/*
 * This function computes the trigamma function for single precision floating point numbers.
 * It includes handling for values where x < 0.5f and utilizes a polynomial approximation for other cases.
 * Special handling is applied for division by zero errors.
 *
 * @param x The input value for which the trigamma function is computed.
 * @return The computed trigamma value for the input x.
 */
inline float trigamma(float x) __ubsan_ignore_float_divide_by_zero__ {
  float sign = +1;
  float result = 0;
  if (x < 0.5f) {
    sign = -1;
    const float sin_pi_x = sinf(c10::pi<float> * x);
    result -= (c10::pi<float> * c10::pi<float>) / (sin_pi_x * sin_pi_x);
    x = 1 - x;
  }
  for (int i = 0; i < 6; ++i) {
    result += 1 / (x * x);
    x += 1;
  }
  const float ixx = 1 / (x*x);
  result += (1 + 1 / (2*x) + ixx * (1.f/6 - ixx * (1.f/30 - ixx * (1.f/42)))) / x;
  return sign * result;
}

/*
 * This function computes the digamma function using numerical approximations.
 * It handles different cases of the input x, including negative values and small values.
 *
 * @param x The input value for which the digamma function is computed.
 * @return The computed digamma value for the input x.
 */
inline double calc_digamma(double x) {
  static double PSI_10 = 2.25175258906672110764;
  if (x == 0) {
    // Special case handling for x == 0 as per standard gamma function behavior.
    return std::copysign(INFINITY, -x);
  }

  bool x_is_integer = x == trunc(x);
  if (x < 0) {
    if (x_is_integer) {
      // Special case handling for negative integer x.
      return std::numeric_limits<double>::quiet_NaN();
    }
    // Handling for negative non-integer x using reflection formula and trigonometric functions.
    double q, r;
    r = std::modf(x, &q);
    return calc_digamma(1 - x) - c10::pi<double> / tan(c10::pi<double> * r);
  }

  // For x >= 10, using a polynomial approximation and other numerical methods.
  double result = 0;
  while (x < 10) {
    // 减去 x 的倒数并更新结果
    result -= 1 / x;
    // 增加 x 的值以准备下一次循环
    x += 1;
  }
  // 如果 x 等于 10，则加上预定义常数 PSI_10 并返回结果
  if (x == 10) {
    return result + PSI_10;
  }

  // 计算渐近的对数伽玛函数值

  // 静态常量数组 A，存储系数
  static const double A[] = {
      8.33333333333333333333E-2,
      -2.10927960927960927961E-2,
      7.57575757575757575758E-3,
      -4.16666666666666666667E-3,
      3.96825396825396825397E-3,
      -8.33333333333333333333E-3,
      8.33333333333333333333E-2,
  };

  // 初始化变量 y 为 0
  double y = 0;
  // 如果 x 小于 1.0e17，则进行渐近伽玛函数的计算
  if (x < 1.0e17) {
    // 计算 z，即 x 的倒数的平方
    double z = 1.0 / (x * x);
    // 使用 polevl 函数计算伽玛函数的多项式近似值
    y = z * polevl(z, A, 6);
  }
  // 返回最终的伽玛函数的计算结果
  return result + log(x) - (0.5 / x) - y;
}

/*
 * This function is derived from the implementation of the digamma function in the Cephes Math Library.
 * See note [3-Clause BSD License for the Cephes Math Library].
 */
// 计算对数 gamma 函数的导数，即 digamma 函数的近似值
inline float calc_digamma(float x) {
  // See [C++ Standard Reference: Gamma Function]
  // 预定义常数 PSI_10，用于 x >= 10 的情况
  static float PSI_10 = 2.25175258906672110764f;
  if (x == 0) {
    // 如果 x 为 0，根据标准，返回 ±∞
    return std::copysign(INFINITY, -x);
  }

  bool x_is_integer = x == truncf(x);
  if (x < 0) {
    if (x_is_integer) {
      // 如果 x 是负整数，返回 NaN
      return std::numeric_limits<float>::quiet_NaN();
    }
    // 计算 x 的小数部分 r，并计算 digamma 函数的负值
    double q, r;
    r = std::modf(x, &q);
    float pi_over_tan_pi_x = (float)(c10::pi<double> / tan(c10::pi<double> * r));
    return calc_digamma(1 - x) - pi_over_tan_pi_x;
  }

  // 将 x 推到大于等于 10
  float result = 0;
  while (x < 10) {
    result -= 1 / x;
    x += 1;
  }
  if (x == 10) {
    // 当 x = 10 时，返回预定义的 PSI_10 值
    return result + PSI_10;
  }

  // 计算渐近 digamma
  static const float A[] = {
      8.33333333333333333333E-2f,
      -2.10927960927960927961E-2f,
      7.57575757575757575758E-3f,
      -4.16666666666666666667E-3f,
      3.96825396825396825397E-3f,
      -8.33333333333333333333E-3f,
      8.33333333333333333333E-2f,
  };

  float y = 0;
  if (x < 1.0e17f) {
    // 如果 x 较小，则计算系数 y
    float z = 1 / (x * x);
    y = z * polevl(z, A, 6);
  }
  // 返回最终的 digamma 值
  return result + logf(x) - (0.5f / x) - y;
}

// 计算 BFloat16 类型的 digamma 函数
inline c10::BFloat16 calc_digamma(c10::BFloat16 a) {
  return calc_digamma(static_cast<float>(a));
}

// 计算 Half 类型的 digamma 函数
inline c10::Half calc_digamma(c10::Half a) {
  return calc_digamma(static_cast<float>(a));
}

// 计算 polygamma 函数
template <typename scalar_t, bool is_cuda=false>
// 如果 n <= 1，则直接返回 0
inline C10_HOST_DEVICE scalar_t calc_polygamma(scalar_t x, int n) {
  const auto one = scalar_t{1};
  // 计算 polygamma 函数的值
  return ((n % 2) ? one : -one) *
      std::exp(std::lgamma(static_cast<scalar_t>(n) + one)) *
      zeta<scalar_t, is_cuda>(static_cast<scalar_t>(n + 1), x);
}

// 计算正则化的不完全 Gamma 函数
// 正则化的不完全 Gamma 函数和它们的辅助函数遵循 SciPy 的实现

/* References
 * [igam1] "The Digital Library of Mathematical Functions", dlmf.nist.gov
 * [igam2] Maddock et al., "Incomplete Gamma Functions",
 *     https://www.boost.org/doc/libs/1_61_0/libs/math/doc/html/math_toolkit/sf_gamma/igamma.html
 */
/*
 * This implementation of the regularized incomplete gamma functions and
 * their helper functions are derived from the implementation of SciPy's
 * gammainc, Cephes's igam and igamc, and Boost's Lanczos approximations.
 * See NOTICE for the licenses.
 */
template <typename scalar_t>
scalar_t ratevl(scalar_t x, const scalar_t num[], int64_t M,
    const scalar_t denom[], int64_t N) {
  // evaluating rational function, i.e., the ratio of two polynomials
  // the coefficients for numerator are given by `num` while coeffs for
  // denumerator are given by `denom`

  int64_t i, dir;
  scalar_t y, num_ans, denom_ans;
  scalar_t absx = std::fabs(x);
  const scalar_t *p;

  if (absx > 1) {
    /* Evaluate as a polynomial in 1/x. */
    dir = -1;
    p = num + M;
    y = 1 / x;
  }
  else {
    dir = 1;
    p = num;
    y = x;
  }

  /* Evaluate the numerator */
  num_ans = *p;
  p += dir;
  for (i = 1; i <= M; i++) {
    num_ans = num_ans * y + *p;
    p += dir;
  }
  /* Evaluate the denominator */
  if (absx > 1) {
    p = denom + N;
  }
  else {
    p = denom;
  }

  denom_ans = *p;
  p += dir;
  for (i = 1; i <= N; i++) {
    denom_ans = denom_ans * y + *p;
    p += dir;
  }
  if (absx > 1) {
    i = N - M;
    return std::pow(x, i) * num_ans / denom_ans;
  }
  else {
    return num_ans / denom_ans;
  }
}

// SciPy's lanczos implementation is taken from Boost
/* (C) Copyright John Maddock 2006.
 * Use, modification and distribution are subject to the
 * Boost Software License, Version 1.0. See
 * https://www.boost.org/LICENSE_1_0.txt or see NOTICE.
 */
template <typename scalar_t>
static scalar_t lanczos_sum_expg_scaled(scalar_t x) {
  // lanczos approximation
  static const scalar_t lanczos_sum_expg_scaled_num[13] = {
    0.006061842346248906525783753964555936883222,
    0.5098416655656676188125178644804694509993,
    19.51992788247617482847860966235652136208,
    449.9445569063168119446858607650988409623,
    6955.999602515376140356310115515198987526,
    75999.29304014542649875303443598909137092,
    601859.6171681098786670226533699352302507,
    3481712.15498064590882071018964774556468,
    14605578.08768506808414169982791359218571,
    43338889.32467613834773723740590533316085,
    86363131.28813859145546927288977868422342,
    103794043.1163445451906271053616070238554,
    56906521.91347156388090791033559122686859
  };
  static const scalar_t lanczos_sum_expg_scaled_denom[13] = {
    1.,
    66.,
    1925.,
    32670.,
    357423.,
    2637558.,
    13339535.,
    45995730.,
    105258076.,
    150917976.,
    120543840.,
    39916800.,
    0.
  };
  return ratevl(x, lanczos_sum_expg_scaled_num,
      sizeof(lanczos_sum_expg_scaled_num) / sizeof(lanczos_sum_expg_scaled_num[0]) - 1,
      lanczos_sum_expg_scaled_denom,
      sizeof(lanczos_sum_expg_scaled_denom) / sizeof(lanczos_sum_expg_scaled_denom[0]) - 1);
}
static scalar_t _igam_helper_fac(scalar_t a, scalar_t x) {
    // 计算 x^a * exp(-a) / gamma(a)
    // 修正自 [igam2] 中 (15) 和 (16)，将 exp(x - a) 替换为 exp(a - x)。

    scalar_t ax, fac, res, num, numfac;
    static scalar_t MAXLOG = std::is_same<scalar_t,double>::value ?
        7.09782712893383996843E2 : 88.72283905206835;
    static scalar_t EXP1 = 2.718281828459045;
    static scalar_t lanczos_g = 6.024680040776729583740234375;

    // 如果 |a - x| 大于 0.4 * |a|，则使用简化的计算方式
    if (std::fabs(a - x) > 0.4 * std::fabs(a)) {
        ax = a * std::log(x) - x - std::lgamma(a);
        // 如果 ax 小于 -MAXLOG，则返回 0.0
        if (ax < -MAXLOG) {
            return 0.0;
        }
        // 返回 exp(ax)
        return std::exp(ax);
    }

    // 否则使用更复杂的计算方式
    fac = a + lanczos_g - 0.5;
    res = std::sqrt(fac / EXP1) / lanczos_sum_expg_scaled(a);

    // 根据 a 和 x 的大小分情况计算 res 的值
    if ((a < 200) && (x < 200)) {
        res *= std::exp(a - x) * std::pow(x / fac, a);
    } else {
        num = x - a - lanczos_g + 0.5;
        numfac = num / fac;
        res *= std::exp(a * (std::log1p(numfac) - numfac) + x * (0.5 - lanczos_g) / fac);
    }
    return res;
}

template <typename scalar_t>
static scalar_t _igam_helper_series(scalar_t a, scalar_t x) {
    // 使用 DLMF 8.11.4 [igam1] 计算 igam
    static scalar_t MACHEP = std::is_same<scalar_t, double>::value ?
        1.11022302462515654042E-16 : 5.9604644775390625E-8;
    static int MAXITER = 2000;

    int i;
    scalar_t ans, ax, c, r;

    ax = _igam_helper_fac(a, x);
    // 如果 ax 等于 0.0，则返回 0.0
    if (ax == 0.0) {
        return 0.0;
    }

    /* power series */
    r = a;
    c = 1.0;
    ans = 1.0;

    // 进行幂级数求和计算
    for (i = 0; i < MAXITER; i++) {
        r += 1.0;
        c *= x / r;
        ans += c;
        if (c <= MACHEP * ans) {
            break;
        }
    }
    return (ans * ax / a);
}

template <typename scalar_t>
static scalar_t _igamc_helper_series(scalar_t a, scalar_t x) {
    // 使用 DLMF 8.7.3 [igam1] 计算 igamc
    // 这与 _igam_helper_series 中的级数计算相关，但额外处理以避免抵消

    int n;
    scalar_t fac = 1;
    scalar_t sum = 0;
    scalar_t term, logx;
    static scalar_t MAXITER = 2000;
    static scalar_t MACHEP = std::is_same<scalar_t, double>::value ?
        1.11022302462515654042E-16 : 5.9604644775390625E-8;

    // 进行级数求和计算
    for (n = 1; n < MAXITER; n++) {
        fac *= -x / n;
        term = fac / (a + n);
        sum += term;
        if (std::fabs(term) <= MACHEP * std::fabs(sum)) {
            break;
        }
    }

    logx = std::log(x);
    term = -std::expm1(a * logx - std::lgamma(1+a));
    return term - std::exp(a * logx - std::lgamma(a)) * sum;
}

template <typename scalar_t>
static scalar_t _igam_helper_asymptotic_series(scalar_t a, scalar_t x, bool igam) {
    // 使用 DLMF 8.12.3/8.12.4 [igam1] 计算 igam 或 igamc

    static const scalar_t d[25][25] =
    # 定义一个包含四个子列表的二维数组，每个子列表包含一系列浮点数，代表多项式的系数
    {
        # 第一个子列表
        -3.3333333333333333e-1,  # 系数0
        8.3333333333333333e-2,   # 系数1
        -1.4814814814814815e-2,  # 系数2
        1.1574074074074074e-3,   # 系数3
        3.527336860670194e-4,    # 系数4
        -1.7875514403292181e-4,  # 系数5
        3.9192631785224378e-5,   # 系数6
        -2.1854485106799922e-6,  # 系数7
        -1.85406221071516e-6,    # 系数8
        8.296711340953086e-7,    # 系数9
        -1.7665952736826079e-7,  # 系数10
        6.7078535434014986e-9,   # 系数11
        1.0261809784240308e-8,   # 系数12
        -4.3820360184533532e-9,  # 系数13
        9.1476995822367902e-10,  # 系数14
        -2.551419399494625e-11,  # 系数15
        -5.8307721325504251e-11, # 系数16
        2.4361948020667416e-11,  # 系数17
        -5.0276692801141756e-12, # 系数18
        1.1004392031956135e-13,  # 系数19
        3.3717632624009854e-13,  # 系数20
        -1.3923887224181621e-13, # 系数21
        2.8534893807047443e-14,  # 系数22
        -5.1391118342425726e-16, # 系数23
        -1.9752288294349443e-15  # 系数24
    },
    {
        # 第二个子列表
        -1.8518518518518519e-3,  # 系数0
        -3.4722222222222222e-3,  # 系数1
        2.6455026455026455e-3,   # 系数2
        -9.9022633744855967e-4,  # 系数3
        2.0576131687242798e-4,   # 系数4
        -4.0187757201646091e-7,  # 系数5
        -1.8098550334489978e-5,  # 系数6
        7.6491609160811101e-6,   # 系数7
        -1.6120900894563446e-6,  # 系数8
        4.6471278028074343e-9,   # 系数9
        1.378633446915721e-7,    # 系数10
        -5.752545603517705e-8,   # 系数11
        1.1951628599778147e-8,   # 系数12
        -1.7543241719747648e-11, # 系数13
        -1.0091543710600413e-9,  # 系数14
        4.1627929918425826e-10,  # 系数15
        -8.5639070264929806e-11, # 系数16
        6.0672151016047586e-14,  # 系数17
        7.1624989648114854e-12,  # 系数18
        -2.9331866437714371e-12, # 系数19
        5.9966963656836887e-13,  # 系数20
        -2.1671786527323314e-16, # 系数21
        -4.9783399723692616e-14, # 系数22
        2.0291628823713425e-14,  # 系数23
        -4.13125571381061e-15    # 系数24
    },
    {
        # 第三个子列表
        4.1335978835978836e-3,    # 系数0
        -2.6813271604938272e-3,   # 系数1
        7.7160493827160494e-4,    # 系数2
        2.0093878600823045e-6,    # 系数3
        -1.0736653226365161e-4,   # 系数4
        5.2923448829120125e-5,    # 系数5
        -1.2760635188618728e-5,   # 系数6
        3.4235787340961381e-8,    # 系数7
        1.3721957309062933e-6,    # 系数8
        -6.298992138380055e-7,    # 系数9
        1.4280614206064242e-7,    # 系数10
        -2.0477098421990866e-10,  # 系数11
        -1.4092529910867521e-8,   # 系数12
        6.228974084922022e-9,     # 系数13
        -1.3670488396617113e-9,   # 系数14
        9.4283561590146782e-13,   # 系数15
        1.2872252400089318e-10,   # 系数16
        -5.5645956134363321e-11,  # 系数17
        1.1975935546366981e-11,   # 系数18
        -4.1689782251838635e-15,  # 系数19
        -1.0940640427884594e-12,  # 系数20
        4.6622399463901357e-13,   # 系数21
        -9.905105763906906e-14,   # 系数22
        1.8931876768373515e-17,   # 系数23
        8.8592218725911273e-15    # 系数24
    },
    {
        # 第四个子列表
        6.4943415637860082e-4,    # 系数0
        2.2947209362139918e-4,    # 系数1
        -4.6918949439525571e-4,   # 系数2
        2.6772063206283885e-4,    # 系数3
        -7.5618016718839764e-5,   # 系数4
        -2.3965051138672967e-7,   # 系数5
        1.1082654115347302e-5,    # 系数6
        -5.6749528269915966e-6,   # 系数7
        1.4230900732435884e-6,    # 系数8
        -2.7861080291528142e-11,  # 系数9
        -1.6958404091930277e-7,   # 系数10
        8.0994649053880824e-8,    # 系数11
        -1.9111168485973654e-8,   # 系数12
        2.3928620439808118e-12,   # 系数13
        2.0620131815488798e-9,    # 系数14
        -9.4604966618551322e-10,  # 系数15
        2.1541049775774908e-10,   # 系数16
        -1.388823336813903e-14,   # 系数17
        -2.1894761681963939e-11,  # 系数18
        9.7909989511716851e-12,
    # 定义一个包含多个列表的大列表，每个小列表包含24个浮点数，用作某种系数或数据集合
    [
        {-8.618882909167117e-4, 7.8403922172006663e-4, -2.9907248030319018e-4,
         -1.4638452578843418e-6, 6.6414982154651222e-5, -3.9683650471794347e-5,
         1.1375726970678419e-5, 2.5074972262375328e-10, -1.6954149536558306e-6,
         8.9075075322053097e-7, -2.2929348340008049e-7, 2.956794137544049e-11,
         2.8865829742708784e-8, -1.4189739437803219e-8, 3.4463580499464897e-9,
         -2.3024517174528067e-13, -3.9409233028046405e-10, 1.8602338968504502e-10,
         -4.356323005056618e-11, 1.2786001016296231e-15, 4.6792750266579195e-12,
         -2.1492464706134829e-12, 4.9088156148096522e-13, -6.3385914848915603e-18,
         -5.0453320690800944e-14},
        {-3.3679855336635815e-4, -6.9728137583658578e-5, 2.7727532449593921e-4,
         -1.9932570516188848e-4, 6.7977804779372078e-5, 1.419062920643967e-7,
         -1.3594048189768693e-5, 8.0184702563342015e-6, -2.2914811765080952e-6,
         -3.252473551298454e-10, 3.4652846491085265e-7, -1.8447187191171343e-7,
         4.8240967037894181e-8, -1.7989466721743515e-14, -6.3061945000135234e-9,
         3.1624176287745679e-9, -7.8409242536974293e-10, 5.1926791652540407e-15,
         9.3589442423067836e-11, -4.5134262161632782e-11, 1.0799129993116827e-11,
         -3.661886712685252e-17, -1.210902069055155e-12, 5.6807435849905643e-13,
         -1.3249659916340829e-13},
        {5.3130793646399222e-4, -5.9216643735369388e-4, 2.7087820967180448e-4,
         7.9023532326603279e-7, -8.1539693675619688e-5, 5.6116827531062497e-5,
         -1.8329116582843376e-5, -3.0796134506033048e-9, 3.4651553688036091e-6,
         -2.0291327396058604e-6, 5.7887928631490037e-7, 2.338630673826657e-13,
         -8.8286007463304835e-8, 4.7435958880408128e-8, -1.2545415020710382e-8,
         8.6496488580102925e-14, 1.6846058979264063e-9, -8.5754928235775947e-10,
         2.1598224929232125e-10, -7.6132305204761539e-16, -2.6639822008536144e-11,
         1.3065700536611057e-11, -3.1799163902367977e-12, 4.7109761213674315e-18,
         3.6902800842763467e-13},
        {3.4436760689237767e-4, 5.1717909082605922e-5, -3.3493161081142236e-4,
         2.812695154763237e-4, -1.0976582244684731e-4, -1.2741009095484485e-7,
         2.7744451511563644e-5, -1.8263488805711333e-5, 5.7876949497350524e-6,
         4.9387589339362704e-10, -1.0595367014026043e-6, 6.1667143761104075e-7,
         -1.7562973359060462e-7, -1.2974473287015439e-12, 2.695423606288966e-8,
         -1.4578352908731271e-8, 3.887645959386175e-9, -3.8810022510194121e-17,
         -5.3279941738772867e-10, 2.7437977643314845e-10, -6.9957960920705679e-11,
         2.5899863874868481e-17, 8.8566890996696381e-12, -4.403168815871311e-12,
         1.0865561947091654e-12},
    ]
    # 定义一个二维列表，包含多个子列表，每个子列表都包含24个浮点数
    {
        # 第一个子列表
        {-6.5262391859530942e-4, 8.3949872067208728e-4, -4.3829709854172101e-4,
         -6.969091458420552e-7, 1.6644846642067548e-4, -1.2783517679769219e-4,
         4.6299532636913043e-5, 4.5579098679227077e-9, -1.0595271125805195e-5,
         6.7833429048651666e-6, -2.1075476666258804e-6, -1.7213731432817145e-11,
         3.7735877416110979e-7, -2.1867506700122867e-7, 6.2202288040189269e-8,
         6.5977038267330006e-16, -9.5903864974256858e-9, 5.2132144922808078e-9,
         -1.3991589583935709e-9, 5.382058999060575e-16, 1.9484714275467745e-10,
         -1.0127287556389682e-10, 2.6077347197254926e-11, -5.0904186999932993e-18,
         -3.3721464474854592e-12},
        # 第二个子列表
        {-5.9676129019274625e-4, -7.2048954160200106e-5, 6.7823088376673284e-4,
         -6.4014752602627585e-4, 2.7750107634328704e-4, 1.8197008380465151e-7,
         -8.4795071170685032e-5, 6.105192082501531e-5, -2.1073920183404862e-5,
         -8.8585890141255994e-10, 4.5284535953805377e-6, -2.8427815022504408e-6,
         8.7082341778646412e-7, 3.6886101871706965e-12, -1.5344695190702061e-7,
         8.862466778790695e-8, -2.5184812301826817e-8, -1.0225912098215092e-14,
         3.8969470758154777e-9, -2.1267304792235635e-9, 5.7370135528051385e-10,
         -1.887749850169741e-19, -8.0931538694657866e-11, 4.2382723283449199e-11,
         -1.1002224534207726e-11},
        # 第三个子列表
        {1.3324454494800656e-3, -1.9144384985654775e-3, 1.1089369134596637e-3,
         9.932404122642299e-7, -5.0874501293093199e-4, 4.2735056665392884e-4,
         -1.6858853767910799e-4, -8.1301893922784998e-9, 4.5284402370562147e-5,
         -3.127053674781734e-5, 1.044986828530338e-5, 4.8435226265680926e-11,
         -2.1482565873456258e-6, 1.329369701097492e-6, -4.0295693092101029e-7,
         -1.7567877666323291e-13, 7.0145043163668257e-8, -4.040787734999483e-8,
         1.1474026743371963e-8, 3.9642746853563325e-18, -1.7804938269892714e-9,
         9.7480262548731646e-10, -2.6405338676507616e-10, 5.794875163403742e-18,
         3.7647749553543836e-11},
        # 第四个子列表
        {1.579727660730835e-3, 1.6251626278391582e-4, -2.0633421035543276e-3,
         2.1389686185689098e-3, -1.0108559391263003e-3, -3.9912705529919201e-7,
         3.6235025084764691e-4, -2.8143901463712154e-4, 1.0449513336495887e-4,
         2.1211418491830297e-9, -2.5779417251947842e-5, 1.7281818956040463e-5,
         -5.6413773872904282e-6, -1.1024320105776174e-11, 1.1223224418895175e-6,
         -6.8693396379526735e-7, 2.0653236975414887e-7, 4.6714772409838506e-14,
         -3.5609886164949055e-8, 2.0470855345905963e-8, -5.8091738633283358e-9,
         -1.332821287582869e-16, 9.0354604391335133e-10, -4.9598782517330834e-10,
         1.3481607129399749e-10},
    }
    {-4.0725121195140166e-3, 6.4033628338080698e-3, -4.0410161081676618e-3,
      -2.183732802866233e-6, 2.1740441801254639e-3, -1.9700440518418892e-3,
      8.3595469747962458e-4, 1.9445447567109655e-8, -2.5779387120421696e-4,
      1.9009987368139304e-4, -6.7696499937438965e-5, -1.4440629666426572e-10,
      1.5712512518742269e-5, -1.0304008744776893e-5, 3.304517767401387e-6,
      7.9829760242325709e-13, -6.4097794149313004e-7, 3.8894624761300056e-7,
      -1.1618347644948869e-7, -2.816808630596451e-15, 1.9878012911297093e-8,
      -1.1407719956357511e-8, 3.2355857064185555e-9, 4.1759468293455945e-20,
      -5.0423112718105824e-10},


    # 第一个数组，包含22个浮点数值



    {-5.9475779383993003e-3, -5.4016476789260452e-4, 8.7910413550767898e-3,
      -9.8576315587856125e-3, 5.0134695031021538e-3, 1.2807521786221875e-6,
      -2.0626019342754683e-3, 1.7109128573523058e-3, -6.7695312714133799e-4,
      -6.9011545676562133e-9, 1.8855128143995902e-4, -1.3395215663491969e-4,
      4.6263183033528039e-5, 4.0034230613321351e-11, -1.0255652921494033e-5,
      6.612086372797651e-6, -2.0913022027253008e-6, -2.0951775649603837e-13,
      3.9756029041993247e-7, -2.3956211978815887e-7, 7.1182883382145864e-8,
      8.925574873053455e-16, -1.2101547235064676e-8, 6.9350618248334386e-9,
      -1.9661464453856102e-9},


    # 第二个数组，包含22个浮点数值



    {1.7402027787522711e-2, -2.9527880945699121e-2, 2.0045875571402799e-2,
      7.0289515966903407e-6, -1.2375421071343148e-2, 1.1976293444235254e-2,
      -5.4156038466518525e-3, -6.3290893396418616e-8, 1.8855118129005065e-3,
      -1.473473274825001e-3, 5.5515810097708387e-4, 5.2406834412550662e-10,
      -1.4357913535784836e-4, 9.9181293224943297e-5, -3.3460834749478311e-5,
      -3.5755837291098993e-12, 7.1560851960630076e-6, -4.5516802628155526e-6,
      1.4236576649271475e-6, 1.8803149082089664e-14, -2.6623403898929211e-7,
      1.5950642189595716e-7, -4.7187514673841102e-8, -6.5107872958755177e-17,
      7.9795091026746235e-9},


    # 第三个数组，包含22个浮点数值



    {3.0249124160905891e-2, 2.4817436002649977e-3, -4.9939134373457022e-2,
      5.9915643009307869e-2, -3.2483207601623391e-2, -5.7212968652103441e-6,
      1.5085251778569354e-2, -1.3261324005088445e-2, 5.5515262632426148e-3,
      3.0263182257030016e-8, -1.7229548406756723e-3, 1.2893570099929637e-3,
      -4.6845138348319876e-4, -1.830259937893045e-10, 1.1449739014822654e-4,
      -7.7378565221244477e-5, 2.5625836246985201e-5, 1.0766165333192814e-12,
      -5.3246809282422621e-6, 3.349634863064464e-6, -1.0381253128684018e-6,
      -5.608909920621128e-15, 1.9150821930676591e-7, -1.1418365800203486e-7,
      3.3654425209171788e-8},


    # 第四个数组，包含22个浮点数值
    # 定义一个包含多个数组的列表，每个数组包含一组浮点数作为系数
    {
        # 第一个数组的系数
        -9.9051020880159045e-2, 1.7954011706123486e-1, -1.2989606383463778e-1,
        -3.1478872752284357e-5, 9.0510635276848131e-2, -9.2828824411184397e-2,
        4.4412112839877808e-2, 2.7779236316835888e-7, -1.7229543805449697e-2,
        1.4182925050891573e-2, -5.6214161633747336e-3, -2.39598509186381e-9,
        1.6029634366079908e-3, -1.1606784674435773e-3, 4.1001337768153873e-4,
        1.8365800754090661e-11, -9.5844256563655903e-5, 6.3643062337764708e-5,
        -2.076250624489065e-5, -1.1806020912804483e-13, 4.2131808239120649e-6,
        -2.6262241337012467e-6, 8.0770620494930662e-7, 6.0125912123632725e-16,
        -1.4729737374018841e-7
    },
    {
        # 第二个数组的系数
        -1.9994542198219728e-1, -1.5056113040026424e-2, 3.6470239469348489e-1,
        -4.6435192311733545e-1, 2.6640934719197893e-1, 3.4038266027147191e-5,
        -1.3784338709329624e-1, 1.276467178337056e-1, -5.6213828755200985e-2,
        -1.753150885483011e-7, 1.9235592956768113e-2, -1.5088821281095315e-2,
        5.7401854451350123e-3, 1.0622382710310225e-9, -1.5335082692563998e-3,
        1.0819320643228214e-3, -3.7372510193945659e-4, -6.6170909729031985e-12,
        8.4263617380909628e-5, -5.5150706827483479e-5, 1.7769536448348069e-5,
        3.8827923210205533e-14, -3.53513697488768e-6, 2.1865832130045269e-6,
        -6.6812849447625594e-7
    },
    {
        # 第三个数组的系数
        7.2438608504029431e-1, -1.3918010932653375, 1.0654143352413968,
        1.876173868950258e-4, -8.2705501176152696e-1, 8.9352433347828414e-1,
        -4.4971003995291339e-1, -1.6107401567546652e-6, 1.9235590165271091e-1,
        -1.6597702160042609e-1, 6.8882222681814333e-2, 1.3910091724608687e-8,
        -2.146911561508663e-2, 1.6228980898865892e-2, -5.9796016172584256e-3,
        -1.1287469112826745e-10, 1.5167451119784857e-3, -1.0478634293553899e-3,
        3.5539072889126421e-4, 8.1704322111801517e-13, -7.7773013442452395e-5,
        5.0291413897007722e-5, -1.6035083867000518e-5, 1.2469354315487605e-14,
        3.1369106244517615e-6
    },
    {
        # 第四个数组的系数
        1.6668949727276811, 1.165462765994632e-1, -3.3288393225018906,
        4.4692325482864037, -2.6977693045875807, -2.600667859891061e-4,
        1.5389017615694539, -1.4937962361134612, 6.8881964633233148e-1,
        1.3077482004552385e-6, -2.5762963325596288e-1, 2.1097676102125449e-1,
        -8.3714408359219882e-2, -7.7920428881354753e-9, 2.4267923064833599e-2,
        -1.7813678334552311e-2, 6.3970330388900056e-3, 4.9430807090480523e-11,
        -1.5554602758465635e-3, 1.0561196919903214e-3, -3.5277184460472902e-4,
        9.3002334645022459e-14, 7.5285855026557172e-5, -4.8186515569156351e-5,
        1.5227271505597605e-5
    }
    # 定义一个包含多个列表的列表，每个列表代表一个长度为 25 的浮点数数组
    {
        # 第一个列表
        {-6.6188298861372935, 1.3397985455142589e+1, -1.0789350606845146e+1,
          -1.4352254537875018e-3, 9.2333694596189809, -1.0456552819547769e+1,
          5.5105526029033471, 1.2024439690716742e-5, -2.5762961164755816,
          2.3207442745387179, -1.0045728797216284, -1.0207833290021914e-7,
          3.3975092171169466e-1, -2.6720517450757468e-1, 1.0235252851562706e-1,
          8.4329730484871625e-10, -2.7998284958442595e-2, 2.0066274144976813e-2,
          -7.0554368915086242e-3, 1.9402238183698188e-12, 1.6562888105449611e-3,
          -1.1082898580743683e-3, 3.654545161310169e-4, -5.1290032026971794e-11,
          -7.6340103696869031e-5},
        # 第二个列表
        {-1.7112706061976095e+1, -1.1208044642899116, 3.7131966511885444e+1,
          -5.2298271025348962e+1, 3.3058589696624618e+1, 2.4791298976200222e-3,
          -2.061089403411526e+1, 2.088672775145582e+1, -1.0045703956517752e+1,
          -1.2238783449063012e-5, 4.0770134274221141, -3.473667358470195,
          1.4329352617312006, 7.1359914411879712e-8, -4.4797257159115612e-1,
          3.4112666080644461e-1, -1.2699786326594923e-1, -2.8953677269081528e-10,
          3.3125776278259863e-2, -2.3274087021036101e-2, 8.0399993503648882e-3,
          -1.177805216235265e-9, -1.8321624891071668e-3, 1.2108282933588665e-3,
          -3.9479941246822517e-4},
        # 第三个列表
        {7.389033153567425e+1, -1.5680141270402273e+2, 1.322177542759164e+2,
          1.3692876877324546e-2, -1.2366496885920151e+2, 1.4620689391062729e+2,
          -8.0365587724865346e+1, -1.1259851148881298e-4, 4.0770132196179938e+1,
          -3.8210340013273034e+1, 1.719522294277362e+1, 9.3519707955168356e-7,
          -6.2716159907747034, 5.1168999071852637, -2.0319658112299095,
          -4.9507215582761543e-9, 5.9626397294332597e-1, -4.4220765337238094e-1,
          1.6079998700166273e-1, -2.4733786203223402e-8, -4.0307574759979762e-2,
          2.7849050747097869e-2, -9.4751858992054221e-3, 6.419922235909132e-6,
          2.1250180774699461e-3},
        # 第四个列表
        {2.1216837098382522e+2, 1.3107863022633868e+1, -4.9698285932871748e+2,
          7.3121595266969204e+2, -4.8213821720890847e+2, -2.8817248692894889e-2,
          3.2616720302947102e+2, -3.4389340280087117e+2, 1.7195193870816232e+2,
          1.4038077378096158e-4, -7.52594195897599e+1, 6.651969984520934e+1,
          -2.8447519748152462e+1, -7.613702615875391e-7, 9.5402237105304373,
          -7.5175301113311376, 2.8943997568871961, -4.6612194999538201e-7,
          -8.0615149598794088e-1, 5.8483006570631029e-1, -2.0845408972964956e-1,
          1.4765818959305817e-4, 5.1000433863753019e-2, -3.3066252141883665e-2,
          1.5109265210467774e-2}
    }
  {-9.8959643098322368e+2, 2.1925555360905233e+3, -1.9283586782723356e+3,
    -1.5925738122215253e-1, 1.9569985945919857e+3, -2.4072514765081556e+3,
    1.3756149959336496e+3, 1.2920735237496668e-3, -7.525941715948055e+2,
    7.3171668742208716e+2, -3.4137023466220065e+2, -9.9857390260608043e-6,
    1.3356313181291573e+2, -1.1276295161252794e+2, 4.6310396098204458e+1,
    -7.9237387133614756e-6, -1.4510726927018646e+1, 1.1111771248100563e+1,
    -4.1690817945270892, 3.1008219800117808e-3, 1.1220095449981468,
    -7.6052379926149916e-1, 3.6262236505085254e-1, 2.216867741940747e-1,
    4.8683443692930507e-1}};

这是一个包含25个元素的静态数组，存储了一系列浮点数值。


int k, n, sgn;
int maxpow = 0;
static scalar_t MACHEP = std::is_same<scalar_t, double>::value ?
  1.11022302462515654042E-16 : 5.9604644775390625E-8;
scalar_t lambda = x / a;
scalar_t sigma = (x - a) / a;
scalar_t eta, res, ck, ckterm, term, absterm;
scalar_t absoldterm = INFINITY;
scalar_t etapow[25] = {1};
scalar_t sum = 0;
scalar_t afac = 1;

声明了一些变量和常量：
- `k`, `n`, `sgn`: 用于循环和条件判断的整数变量。
- `maxpow`: 追踪当前使用的最大幂次。
- `MACHEP`: 如果`scalar_t`是`double`类型，则设置为一个特定的极小值；否则设置为另一个极小值。
- `lambda`, `sigma`: 根据输入变量`x`和`a`计算得到的浮点数值。
- `eta`, `res`, `ck`, `ckterm`, `term`, `absterm`: 存储中间计算结果的浮点数变量。
- `absoldterm`: 初始设为正无穷大。
- `etapow`: 长度为25的数组，用于存储`eta`的不同幂次。
- `sum`: 累加变量，用于存储最终求和结果。
- `afac`: 初始化为1，用于调整系数。


if (igam) {
  sgn = -1;
}
else {
  sgn = 1;
}

根据条件`igam`的真假，设置`sgn`的值为1或-1。


if (lambda > 1) {
  eta = std::sqrt(-2 * (std::log1p(sigma) - sigma));
}
else if (lambda < 1) {
  eta = -std::sqrt(-2 * (std::log1p(sigma) - sigma));
}
else {
  eta = 0;
}

根据`lambda`的值，计算`eta`：
- 如果`lambda`大于1，计算`eta`为`sqrt(-2 * (log1p(sigma) - sigma))`。
- 如果`lambda`小于1，计算`eta`为负数的上述表达式。
- 如果`lambda`等于1，设置`eta`为0。


res = 0.5 * std::erfc(sgn * eta * std::sqrt(a / 2));

计算`res`，使用误差函数的补函数`erfc`，输入为`sgn * eta * sqrt(a / 2)`的一半。


for (k = 0; k < 25; k++) {
  ck = d[k][0];
  for (n = 1; n < 25; n++) {
    if (n > maxpow) {
      etapow[n] = eta * etapow[n-1];
      maxpow += 1;
    }
    ckterm = d[k][n]*etapow[n];
    ck += ckterm;
    if (std::fabs(ckterm) < MACHEP * std::fabs(ck)) {
      break;
    }
  }
  term = ck * afac;
  absterm = std::fabs(term);
  if (absterm > absoldterm) {
    break;
  }
  sum += term;
  if (absterm < MACHEP * std::fabs(sum)) {
    break;
  }
  absoldterm = absterm;
  afac /= a;
}

通过两个嵌套循环计算一个和级数，更新`sum`，直到达到精度要求或达到最大迭代次数。


res += sgn * std::exp(-0.5 * a * eta * eta) * sum / std::sqrt(2 * c10::pi<float> * a);

最后计算`res`的最终值，加上一个指数项的乘积和一个常数项。


return res;

返回计算结果`res`。  
// 计算 igamc 使用 DLMF 8.9.2 中的方法 [igam1]
template <typename scalar_t>
static scalar_t _igamc_helper_continued_fraction(scalar_t a, scalar_t x) {
    int i;
    scalar_t ans, ax, c, yc, r, t, y, z;
    scalar_t pk, pkm1, pkm2, qk, qkm1, qkm2;
    int MAXITER = 2000;
    // 机器精度 MACHEP 和大数阈值 BIG 的设定，根据 scalar_t 的类型选择不同的值
    static scalar_t MACHEP = std::is_same<scalar_t, double>::value ?
        1.11022302462515654042E-16 : 5.9604644775390625E-8;
    static scalar_t BIG = std::is_same<scalar_t, double>::value ?
        4.503599627370496e15 : 16777216.;
    static scalar_t BIGINV = std::is_same<scalar_t, double>::value ?
        2.22044604925031308085e-16 : 5.9604644775390625E-8;

    // 计算 igam_helper_fac 函数的结果
    ax = _igam_helper_fac(a, x);
    // 如果结果为零，则直接返回零
    if (ax == 0.0) {
        return 0.0;
    }

    /* 连分式计算 */
    y = 1.0 - a;
    z = x + y + 1.0;
    c = 0.0;
    pkm2 = 1.0;
    qkm2 = x;
    pkm1 = x + 1.0;
    qkm1 = z * x;
    ans = pkm1 / qkm1;

    for (i = 0; i < MAXITER; i++) {
        c += 1.0;
        y += 1.0;
        z += 2.0;
        yc = y * c;
        pk = pkm1 * z - pkm2 * yc;
        qk = qkm1 * z - qkm2 * yc;
        if (qk != 0) {
            r = pk / qk;
            t = std::fabs((ans - r) / r);
            ans = r;
        } else {
            t = 1.0;
        }
        pkm2 = pkm1;
        pkm1 = pk;
        qkm2 = qkm1;
        qkm1 = qk;
        // 若误差小于机器精度，则退出循环
        if (std::fabs(pk) > BIG) {
            pkm2 *= BIGINV;
            pkm1 *= BIGINV;
            qkm2 *= BIGINV;
            qkm1 *= BIGINV;
        }
        if (t <= MACHEP) {
            break;
        }
    }
    return ans * ax;
}

// 计算正则化的上不完全伽玛函数 igammac
template <typename scalar_t>
inline scalar_t calc_igammac(scalar_t a, scalar_t x) {
    /* 根据 a 和 x 的值不同，计算正则化的上不完全伽玛函数的方法也不同：
     * - 如果 x 和/或 a 处于定义域边界，则将结果设为边界处的值
     * - 如果 a 较大且接近 x，则使用大参数的均匀渐近展开式（见 DLMF 8.12.4 [igam1]）
     * - 如果 x > 1.1 且 x < a，则使用从正则化下不完全伽玛函数减法得出
     * - 否则，计算从 [igam2] 方程 (5) 开始的级数
     */
    scalar_t absxma_a;

    static scalar_t SMALL = 20.0;
    static scalar_t LARGE = 200.0;
    static scalar_t SMALLRATIO = 0.3;
    static scalar_t LARGERATIO = 4.5;

    // 注意在 SciPy 中，a 和 x 是非负数，其中 igammac(0, x) = 0.0 当且仅当 x > 0
    if ((x < 0) || (a < 0)) {
        // 函数定义域之外的情况
        return std::numeric_limits<scalar_t>::quiet_NaN();
    } else if (a == 0) {
        if (x > 0) {
            return 0.0;
        } else {
            return std::numeric_limits<scalar_t>::quiet_NaN();
        }
    } else if (x == 0) {
        return 1.0;
    } else if (std::isinf(a)) {
        if (std::isinf(x)) {
            return std::numeric_limits<scalar_t>::quiet_NaN();
        }
        return 1.0;
    } else if (std::isinf(x)) {
    # 如果函数执行到这里，说明输入的参数不符合预期，返回浮点数 0.0
    return 0.0;
  }

  # 计算 x 和 a 之间的相对误差
  absxma_a = std::fabs(x - a) / a;
  # 检查是否满足使用渐近级数的条件
  if ((a > SMALL) && (a < LARGE) && (absxma_a < SMALLRATIO)) {
     return _igam_helper_asymptotic_series(a, x, 0);
  }
  # 检查是否满足使用大数渐近估计的条件
  else if ((a > LARGE) && (absxma_a < LARGERATIO / std::sqrt(a))) {
     return _igam_helper_asymptotic_series(a, x, 0);
  }

  # 根据 x 的值和 a 的值选择不同的计算路径
  if (x > 1.1) {
    if (x < a) {
      # 如果 x < a，返回 1 减去正则化不完全 Gamma 函数级数部分的计算结果
      return 1.0 - _igam_helper_series(a, x);
    }
    else {
      # 如果 x >= a，返回正则化不完全 Gamma 函数的连分式计算结果
      return _igamc_helper_continued_fraction(a, x);
    }
  }
  else if (x <= 0.5) {
    # 如果 x <= 0.5，根据条件选择使用级数或者连分式计算
    if (-0.4 / std::log(x) < a) {
      return 1.0 - _igam_helper_series(a, x);
    }
    else {
      return _igamc_helper_series(a, x);
    }
  }
  else {
    # 如果 0.5 < x <= 1.1，根据条件选择使用级数或者连分式计算
    if (x * 1.1 < a) {
      return 1.0 - _igam_helper_series(a, x);
    }
    else {
      return _igamc_helper_series(a, x);
    }
  }
}

template <typename scalar_t>
scalar_t calc_igamma(scalar_t a, scalar_t x) {
  /* 计算正则化的下不完全伽玛函数，根据参数 a 和 x 的不同情况进行不同的计算：
   * - 如果 x 和/或 a 处于定义区域的边界，则返回边界结果
   * - 如果 a 较大且接近 x，使用大参数的均匀渐近展开（参见 DLMF 8.12.3 [igam1]）
   * - 如果 x > 1 且 x > a，则使用正则化的上不完全伽玛函数的减法
   * - 否则，计算 [igam2] 式 (4) 的级数
   */
  scalar_t absxma_a;
  static scalar_t SMALL = 20.0;
  static scalar_t LARGE = 200.0;
  static scalar_t SMALLRATIO = 0.3;
  static scalar_t LARGERATIO = 4.5;

  // 边界值，参考 SciPy
  // 注意在 SciPy 中，a 和 x 都是非负的，且不同时为 0（即它们中最多一个可以为 0），
  // 其中 igamma(0, x) = 1.0 当且仅当 x > 0。
  if ((x < 0) || (a < 0)) {
    // 函数定义区域之外
    return std::numeric_limits<scalar_t>::quiet_NaN();
  }
  else if (a == 0) {
    if (x > 0) {
      return 1.0;
    }
    else {
      return std::numeric_limits<scalar_t>::quiet_NaN();
    }
  }
  else if (x == 0) {
    return 0.0; // 积分下限为零
  }
  else if (std::isinf(a)) {
    if (std::isinf(x)) {
      return std::numeric_limits<scalar_t>::quiet_NaN();
    }
    return 0.0;
  }
  else if (std::isinf(x)) {
    return 1.0;
  }

  /* 当 a ~ x 时的渐近区域。参见 [igam2] */
  absxma_a = std::fabs(x - a) / a;
  if ((a > SMALL) && (a < LARGE) && (absxma_a < SMALLRATIO)) {
    return _igam_helper_asymptotic_series(a, x, 1);
  }
  else if ((a > LARGE) && (absxma_a < LARGERATIO / std::sqrt(a))) {
    return _igam_helper_asymptotic_series(a, x, 1);
  }

  if ((x > 1.0) && (x > a)) {
    return 1.0 - calc_igammac(a, x);
  }

  return _igam_helper_series(a, x);
}

template <>
C10_UNUSED inline c10::BFloat16 calc_igamma<c10::BFloat16>(c10::BFloat16 a, c10::BFloat16 x) {
  return calc_igamma<float>(float(a), float(x));
}

template <>
C10_UNUSED inline c10::Half calc_igamma<c10::Half>(c10::Half a, c10::Half x) {
  return calc_igamma<float>(float(a), float(x));
}

template <>
C10_UNUSED inline c10::BFloat16 calc_igammac<c10::BFloat16>(c10::BFloat16 a, c10::BFloat16 x) {
  return calc_igammac<float>(float(a), float(x));
}

template <>
C10_UNUSED inline c10::Half calc_igammac<c10::Half>(c10::Half a, c10::Half x) {
  return calc_igammac<float>(float(a), float(x));
}

inline c10::BFloat16 calc_erfinv(c10::BFloat16 a) { return calc_erfinv(float(a)); }

template <typename T>
inline T abs_impl(T v) {
  return std::abs(v);
}

template <>
C10_UNUSED inline uint8_t abs_impl(uint8_t v) {
  return v;
}

template <typename T>
inline typename std::enable_if<std::is_integral<T>::value, T>::type
calc_gcd(T a, T b) {
  a = abs_impl(a);
  b = abs_impl(b);
  while (a != 0) {
    T c = a;
    a = b % a;
    b = c;
  }
  return b;
}

template <typename T>
/*
 * Computes the base 2 exponential function for a given floating-point number.
 * Uses std::exp2 from the C++ standard library.
 */
C10_HOST_DEVICE T exp2_impl(T x) {
  return std::exp2(x);
}

/*
 * Computes the base 2 exponential function for a given complex number.
 * Since there is no std::exp2 overload for complex numbers, it uses the identity
 * exp(2^x) = e^(ln(2) * x) where ln(2) is the natural logarithm of 2.
 */
template <typename T>
C10_HOST_DEVICE c10::complex<T> exp2_impl(c10::complex<T> x) {
  constexpr auto ln2 = c10::ln_2<T>; // ln(2) for the complex type T
  return std::exp(ln2 * x);
}

/*
 * This function is derived from the implementation of the chbevl function in the Cephes Math Library.
 * See note [3-Clause BSD License for the Cephes Math Library].
 *
 * Evaluates the series
 *
 *       len-1
 *         - '
 *  y  =   >   array[i] T (x/2)
 *         -             i
 *        i=0
 *
 * of Chebyshev polynomials Ti at argument x/2.
 *
 * Coefficients are stored in reverse order, i.e. the zero order term is last in the array.  Note len is the number of
 * coefficients, not the order.
 *
 * If coefficients are for the interval a to b, x must have been transformed to x -> 2(2x - b - a)/(b-a) before
 * entering the routine.  This maps x from (a, b) to (-1, 1), over which the Chebyshev polynomials are defined.
 *
 * If the coefficients are for the inverted interval, in which (a, b) is mapped to (1/b, 1/a), the transformation
 * required is x -> 2(2ab/x - b - a)/(b-a).  If b is infinity, this becomes x -> 4a/x - 1.
 */
template <typename T>
inline typename std::enable_if<std::is_floating_point<T>::value, T>::type
chbevl(const T x, const T array[], size_t len) {
  T b0, b1, b2;

  b0 = array[0]; // Initialize b0 with the first coefficient
  b1 = static_cast<T>(0.0); // Initialize b1 with 0.0

  // Evaluate the Chebyshev series
  for (size_t i = 1; i < len; ++i) {
    b2 = b1; // Shift coefficients
    b1 = b0;
    b0 = x * b1 - b2 + array[i]; // Update b0 using the recurrence relation for Chebyshev polynomials
  }

  return (static_cast<T>(0.5) * (b0 - b2)); // Return the computed value of the series
}

/*
 * This function is derived from the implementation of the i0 function in the Cephes Math Library.
 * See note [3-Clause BSD License for the Cephes Math Library].
 *
 * Computes an approximation of the zeroth order modified Bessel function of the first kind.
 * The approximation is actually two (sub)approximations, both using a Chebyshev polynomial expansion.
 * One approximates the function over [0, 8], and the other over (8, infinity). This function takes the absolute value
 * of all inputs to convert them into the domain of the approximation.
 */
template <typename T>
inline std::tuple<const T*, size_t> chebyshev_coefficients_i0e_A() {
  /* Chebyshev coefficients for exp(-x) I0(x)
   * in the interval [0,8].
   *
   * lim(x->0){ exp(-x) I0(x) } = 1.
   */
  static const T coeff[] = {
      -4.41534164647933937950E-18, 3.33079451882223809783E-17,
      -2.43127984654795469359E-16, 1.71539128555513303061E-15,
      -1.16853328779934516808E-14, 7.67618549860493561688E-14,
      -4.85644678311192946090E-13, 2.95505266312963983461E-12,
      -1.72682629144155570723E-11, 9.67580903537323691224E-11,
      -5.18979560163526290666E-10, 2.65982372468238665035E-9,
      -1.30002500998624804212E-8,  6.04699502254191894932E-8,
      -2.67079385394061173391E-7,  1.11738753912010371815E-6,
      -4.41673835845875056359E-6,  1.64484480707288970893E-5,
      -5.75419501008210370398E-5,  1.88502885095841655729E-4,
      -5.76375574538582365885E-4,  1.63947561694133579842E-3,
      -4.32430999505057594430E-3,  1.05464603945949983183E-2,
      -2.37374148058994688156E-2,  4.93052842396707084878E-2,
      -9.49010970480476444210E-2,  1.71620901522208775349E-1,
      -3.04682672343198398683E-1,  6.76795274409476084995E-1};
  return std::make_tuple(coeff, 30);
};

template <typename T>
inline std::tuple<const T*, size_t> chebyshev_coefficients_i0e_B() {
  /* Chebyshev coefficients for exp(-x) sqrt(x) I0(x)
   * in the inverted interval [8,infinity].
   *
   * lim(x->inf){ exp(-x) sqrt(x) I0(x) } = 1/sqrt(2pi).
   */
  static const T coeff[] = {
      -7.23318048787475395456E-18, -4.83050448594418207126E-18,
      4.46562142029675999901E-17,  3.46122286769746109310E-17,
      -2.82762398051658348494E-16, -3.42548561967721913462E-16,
      1.77256013305652638360E-15,  3.81168066935262242075E-15,
      -9.55484669882830764870E-15, -4.15056934728722208663E-14,
      1.54008621752140982691E-14,  3.85277838274214270114E-13,
      7.18012445138366623367E-13,  -1.79417853150680611778E-12,
      -1.32158118404477131188E-11, -3.14991652796324136454E-11,
      1.18891471078464383424E-11,  4.94060238822496958910E-10,
      3.39623202570838634515E-9,   2.26666899049817806459E-8,
      2.04891858946906374183E-7,   2.89137052083475648297E-6,
      6.88975834691682398426E-5,   3.36911647825569408990E-3,
      8.04490411014108831608E-1};

  return std::make_tuple(coeff, 25);
};
chebyshev_coefficients_i1e_A() {
  /* Chebyshev coefficients for exp(-x) I1(x)
   * in the interval [0,8].
   *
   * lim(x->0){ exp(-x) I1(x) / x } = 1/2.
   */
  static const T coeff[] = {
      // Chebyshev coefficients for double precision calculation
      2.77791411276104639959E-18, -2.11142121435816608115E-17,
      1.55363195773620046921E-16, -1.10559694773538630805E-15,
      7.60068429473540693410E-15, -5.04218550472791168711E-14,
      3.22379336594557470981E-13, -1.98397439776494371520E-12,
      1.17361862988909016308E-11, -6.66348972350202774223E-11,
      3.62559028155211703701E-10, -1.88724975172282928790E-9,
      9.38153738649577178388E-9,  -4.44505912879632808065E-8,
      2.00329475355213526229E-7,  -8.56872026469545474066E-7,
      3.47025130813767847674E-6,  -1.32731636560394358279E-5,
      4.78156510755005422638E-5,  -1.61760815825896745588E-4,
      5.12285956168575772895E-4,  -1.51357245063125314899E-3,
      4.15642294431288815669E-3,  -1.05640848946261981558E-2,
      2.47264490306265168283E-2,  -5.29459812080949914269E-2,
      1.02643658689847095384E-1,  -1.76416518357834055153E-1,
      2.52587186443633654823E-1};
  return std::make_tuple(coeff, 29);
};

template <typename T>
inline typename std::enable_if<std::is_same<float, T>::value, std::tuple<const T*, size_t>>::type
chebyshev_coefficients_i1e_A() {
  /* Chebyshev coefficients for exp(-x) I1(x)
   * in the interval [0,8].
   *
   * lim(x->0){ exp(-x) I1(x) / x } = 1/2.
   */
  static const T coeff[] = {
      // Chebyshev coefficients for single precision calculation
      9.38153738649577178388E-9f,
      -4.44505912879632808065E-8f,
      2.00329475355213526229E-7f,
      -8.56872026469545474066E-7f,
      3.47025130813767847674E-6f,
      -1.32731636560394358279E-5f,
      4.78156510755005422638E-5f,
      -1.61760815825896745588E-4f,
      5.12285956168575772895E-4f,
      -1.51357245063125314899E-3f,
      4.15642294431288815669E-3f,
      -1.05640848946261981558E-2f,
      2.47264490306265168283E-2f,
      -5.29459812080949914269E-2f,
      1.02643658689847095384E-1f,
      -1.76416518357834055153E-1f,
      2.52587186443633654823E-1f};
  return std::make_tuple(coeff, 17);
};

template <typename T>
inline typename std::enable_if<std::is_same<double, T>::value, std::tuple<const T*, size_t>>::type
// 定义函数 chebyshev_coefficients_i1e_B，返回 Chebyshev 多项式系数以及数组长度
chebyshev_coefficients_i1e_B() {
  /* Chebyshev coefficients for exp(-x) sqrt(x) I1(x)
   * in the inverted interval [8,infinity].
   *
   * lim(x->inf){ exp(-x) sqrt(x) I1(x) } = 1/sqrt(2pi).
   */
  // 静态常量数组，存储 Chebyshev 多项式系数
  static const T coeff[] = {
      7.51729631084210481353E-18,  4.41434832307170791151E-18,
      -4.65030536848935832153E-17, -3.20952592199342395980E-17,
      2.96262899764595013876E-16,  3.30820231092092828324E-16,
      -1.88035477551078244854E-15, -3.81440307243700780478E-15,
      1.04202769841288027642E-14,  4.27244001671195135429E-14,
      -2.10154184277266431302E-14, -4.08355111109219731823E-13,
      -7.19855177624590851209E-13, 2.03562854414708950722E-12,
      1.41258074366137813316E-11,  3.25260358301548823856E-11,
      -1.89749581235054123450E-11, -5.58974346219658380687E-10,
      -3.83538038596423702205E-9,  -2.63146884688951950684E-8,
      -2.51223623787020892529E-7,  -3.88256480887769039346E-6,
      -1.10588938762623716291E-4,  -9.76109749136146840777E-3,
      7.78576235018280120474E-1};

  // 返回 Chebyshev 多项式系数数组及其长度
  return std::make_tuple(coeff, 25);
};

// 模板函数，仅当 T 为 float 类型时有效，返回 Chebyshev 多项式系数以及数组长度
template <typename T>
inline typename std::enable_if<std::is_same<float, T>::value, std::tuple<const T*, size_t>>::type
chebyshev_coefficients_i1e_B() {
  /* Chebyshev coefficients for exp(-x) sqrt(x) I1(x)
   * in the inverted interval [8,infinity].
   *
   * lim(x->inf){ exp(-x) sqrt(x) I1(x) } = 1/sqrt(2pi).
   */
  // 静态常量数组，存储 Chebyshev 多项式系数（仅包含 float 类型）
  static const T coeff[] = {
      -3.83538038596423702205E-9f,
      -2.63146884688951950684E-8f,
      -2.51223623787020892529E-7f,
      -3.88256480887769039346E-6f,
      -1.10588938762623716291E-4f,
      -9.76109749136146840777E-3f,
      7.78576235018280120474E-1f};

  // 返回 Chebyshev 多项式系数数组及其长度
  return std::make_tuple(coeff, 7);
};

// 模板函数，计算修正的 Bessel 函数 I0(x)，其中 x 可以是任意浮点数类型 T
template <typename T>
inline typename std::enable_if<std::is_floating_point<T>::value, T>::type
calc_i0(T _x) {
  // 取 x 的绝对值
  T x = std::abs(_x);

  // 如果 x <= 8.0，则使用 chebyshev_coefficients_i0e_A 返回的系数计算
  if (x <= T{8.0}) {
    auto coeff_pair = chebyshev_coefficients_i0e_A<T>();
    auto A = std::get<0>(coeff_pair);
    auto len = std::get<1>(coeff_pair);
    T y = (x / T{2.0}) - T{2.0};
    // 返回 exp(x) * chbevl(y, A, len) 的计算结果
    return static_cast<T>(std::exp(x) * chbevl(y, A, len));
  }
  // 否则使用 chebyshev_coefficients_i0e_B 返回的系数计算
  auto coeff_pair = chebyshev_coefficients_i0e_B<T>();
  auto B = std::get<0>(coeff_pair);
  auto len = std::get<1>(coeff_pair);
  // 返回 exp(x) * chbevl(32.0 / x - 2.0, B, len) / sqrt(x) 的计算结果
  return std::exp(x) * chbevl(T{32.0} / x - T{2.0}, B, len) / std::sqrt(x);
}

// 将 bfloat16 类型的输入向上转型为 float 类型，以提高数值精度
inline c10::BFloat16 calc_i0(c10::BFloat16 a) { return calc_i0(static_cast<float>(a)); }
/*
 * This function is derived from the implementation of the i1 function in the Cephes Math Library.
 * See note [3-Clause BSD License for the Cephes Math Library].
 *
 * Computes an approximation of the first order modified Bessel function of the first kind.
 * The approximation is actually two (sub)approximations, both using a Chebyshev polynomial expansion.
 * One approximates the function over [0, 8], and the other over (8, infinity). This function takes the absolute value
 * of all inputs to convert them into the domain of the approximation.
 */
template <typename T>
inline typename std::enable_if<std::is_floating_point<T>::value, T>::type
calc_i1(T _x) {
  // Take the absolute value of the input to ensure it is within the domain of approximation
  T x = std::abs(_x);

  if (x <= T{8.0}) {
    // Retrieve coefficients for the Chebyshev polynomial expansion from A coefficients
    auto coeff_pair = chebyshev_coefficients_i1e_A<T>();
    auto A = std::get<0>(coeff_pair);
    auto len = std::get<1>(coeff_pair);
    // Compute y for the approximation within [0, 8]
    T y = (x / T{2.0}) - T{2.0};
    // Evaluate the approximation using Chebyshev polynomial
    const T out = std::exp(x) * x * chbevl(y, A, len);
    // Return the computed value with correct sign based on the original input
    return (_x < T{0.0}) ? -out : out;
  }
  // For x > 8, retrieve coefficients for the Chebyshev polynomial expansion from B coefficients
  auto coeff_pair = chebyshev_coefficients_i1e_B<T>();
  auto B = std::get<0>(coeff_pair);
  auto len = std::get<1>(coeff_pair);
  // Evaluate the approximation for x > 8 using Chebyshev polynomial and exponential function
  const T out = (std::exp(x) * chbevl(T{32.0} / x - T{2.0}, B, len)) / std::sqrt(x);
  // Return the computed value with correct sign based on the original input
  return (_x < T{0.0}) ? -out : out;
}

/*
 * This function is derived from the implementation of the i1e function in the Cephes Math Library.
 * See note [3-Clause BSD License for the Cephes Math Library].
 *
 * Computes an approximation of the exponentially scaled first order modified Bessel function of the first kind.
 * The approximation is actually two (sub)approximations, both using a Chebyshev polynomial expansion.
 * One approximates the function over [0, 8], and the other over (8, infinity). This function takes the absolute value
 * of all inputs to convert them into the domain of the approximation.
 */
template <typename T>
inline typename std::enable_if<std::is_floating_point<T>::value, T>::type
calc_i1e(T _x) {
  // Take the absolute value of the input to ensure it is within the domain of approximation
  T x = std::abs(_x);

  if (x <= T{8.0}) {
    // Retrieve coefficients for the Chebyshev polynomial expansion from A coefficients
    auto coeff_pair = chebyshev_coefficients_i1e_A<T>();
    auto A = std::get<0>(coeff_pair);
    auto len = std::get<1>(coeff_pair);
    // Compute y for the approximation within [0, 8]
    T y = (x / T{2.0}) - T{2.0};
    // Evaluate the approximation using Chebyshev polynomial
    const T out = chbevl(y, A, len) * x;
    // Return the computed value with correct sign based on the original input
    return (_x < T{0.0}) ? -out : out;
  }
  // For x > 8, retrieve coefficients for the Chebyshev polynomial expansion from B coefficients
  auto coeff_pair = chebyshev_coefficients_i1e_B<T>();
  auto B = std::get<0>(coeff_pair);
  auto len = std::get<1>(coeff_pair);
  // Evaluate the approximation for x > 8 using Chebyshev polynomial and exponential function
  const auto out = chbevl(T{32.0} / x - T{2.0}, B, len) / std::sqrt(x);
  // Return the computed value with correct sign based on the original input
  return (_x < T{0.0}) ? -out : out;
}
  // 定义常数 sqrt(2pi)，用于后续计算
  constexpr T s2pi = 2.50662827463100050242E0;
  // 定义常数 1
  constexpr T one = 1;
  // 定义常数 0
  constexpr T zero = 0;

  // 近似计算 0 <= |y - 0.5| <= 3/8 区间的正态分布逆函数的系数
  static const T P0[5] = {
      -5.99633501014107895267E1,
      9.80010754185999661536E1,
      -5.66762857469070293439E1,
      1.39312609387279679503E1,
      -1.23916583867381258016E0,
  };

  // 近似计算 0 <= |y - 0.5| <= 3/8 区间的正态分布逆函数的系数
  static const T Q0[9] = {
      1.00000000000000000000E0,
      1.95448858338141759834E0,
      4.67627912898881538453E0,
      8.63602421390890590575E1,
      -2.25462687854119370527E2,
      2.00260212380060660359E2,
      -8.20372256168333339912E1,
      1.59056225126211695515E1,
      -1.18331621121330003142E0,
  };

  // 近似计算对应区间的正态分布逆函数的系数，区间 z = sqrt(-2 log y) 在 2 到 8 之间
  static const T P1[9] = {
      4.05544892305962419923E0,
      3.15251094599893866154E1,
      5.71628192246421288162E1,
      4.40805073893200834700E1,
      1.46849561928858024014E1,
      2.18663306850790267539E0,
      -1.40256079171354495875E-1,
      -3.50424626827848203418E-2,
      -8.57456785154685413611E-4,
  };

  // 近似计算对应区间的正态分布逆函数的系数，区间 z = sqrt(-2 log y) 在 2 到 8 之间
  static const T Q1[9] = {
      1.00000000000000000000E0,
      1.57799883256466749731E1,
      4.53907635128879210584E1,
      4.13172038254672030440E1,
      1.50425385692907503408E1,
      2.50464946208309415979E0,
      -1.42182922854787788574E-1,
      -3.80806407691578277194E-2,
      -9.33259480895457427372E-4,
  };

  // 近似计算对应区间的正态分布逆函数的系数，区间 z = sqrt(-2 log y) 在 8 到 64 之间
  static const T P2[9] = {
      3.23774891776946035970E0,
      6.91522889068984211695E0,
      3.93881025292474443415E0,
      1.33303460815807542389E0,
      2.01485389549179081538E-1,
      1.23716634817820021358E-2,
      3.01581553508235416007E-4,
      2.65806974686737550832E-6,
      6.23974539184983293730E-9,
  };

  // 近似计算对应区间的正态分布逆函数的系数，区间 z = sqrt(-2 log y) 在 8 到 64 之间
  static const T Q2[9] = {
      1.00000000000000000000E0,
      6.02427039364742014255E0,
      3.67983563856160859403E0,
      1.37702099489081330271E0,
      2.16236993594496635890E-1,
      1.34204006088543189037E-2,
      3.28014464682127739104E-4,
      2.89247864745380683936E-6,
      6.79019408009981274425E-9,
  };

  // 若 y0 等于 0，则返回负无穷
  if (y0 == zero) {
    return -std::numeric_limits<T>::infinity();
  }
  // 若 y0 等于 1，则返回正无穷
  if (y0 == one) {
    return std::numeric_limits<T>::infinity();
  }
  // 若 y0 小于 0 或大于 1，则返回 NaN
  if (y0 < zero || y0 > one) {
    return std::numeric_limits<T>::quiet_NaN();
  }

  // 代码标志位，默认为 true
  bool code = true;
  // 初始化变量 y 为 y0
  T y = y0;
  // 若 y 大于 1 - exp(-2) ≈ 0.135，将 y 转换为 1 - y 并修改 code 标志位
  if (y > one - T{0.13533528323661269189}) { /* 0.135... = exp(-2) */
    y = one - y;
    code = false;
  }

  // 若 y 大于 0.135，将 y 调整为 y - 0.5，并计算额外的变量 x
  if (y > T{0.13533528323661269189}) {
    y = y - T{0.5};
    const T y2 = y * y;
    // 计算正态分布逆函数的近似值 x
    T x = y + y * (y2 * polevl(y2, P0, 4) / polevl(y2, Q0, 8));
    // 返回 x 乘以 s2pi 的结果
    return (x * s2pi);
  }

  // 计算变量 x，其值为负2.0乘以自然对数 y 的平方根
  T x = ::sqrt(T{-2.0} * ::log(y));
  // 计算常量 x0，其值为 x 减去自然对数 x 除以 x 的结果
  const T x0 = x - ::log(x) / x;

  // 计算常量 z，其值为 1 除以 x
  const T z = one / x;
  T x1;
  // 如果 x 小于 8.0，则执行以下操作；注释表明 y 大于 exp(-32) = 1.2664165549e-14
  if (x < T{8.0}) /* y > exp(-32) = 1.2664165549e-14 */
  {
    // 计算 x1，其值为 z 乘以 polevl(z, P1, 8) 的结果，再除以 polevl(z, Q1, 8) 的结果
    x1 = z * polevl(z, P1, 8) / polevl(z, Q1, 8);
  } else {
    // 否则，计算 x1，其值为 z 乘以 polevl(z, P2, 8) 的结果，再除以 polevl(z, Q2, 8) 的结果
    x1 = z * polevl(z, P2, 8) / polevl(z, Q2, 8);
  }
  // 更新 x 的值，其为 x0 减去 x1 的结果
  x = x0 - x1;
  // 如果 code 不为 0，则将 x 取相反数
  if (code) {
    x = -x;
  }
  // 返回计算后的 x 值
  return x;
"""
/* erfcx(x) = exp(x^2) erfc(x) function, for real x, written by
   Steven G. Johnson, October 2012.

   This function combines a few different ideas.

   First, for x > 50, it uses a continued-fraction expansion (same as
   for the Faddeeva function, but with algebraic simplifications for z=i*x).

   Second, for 0 <= x <= 50, it uses Chebyshev polynomial approximations,
   but with two twists:

      a) It maps x to y = 4 / (4+x) in [0,1].  This simple transformation,
         inspired by a similar transformation in the octave-forge/specfun
         erfcx by Soren Hauberg, results in much faster Chebyshev convergence
         than other simple transformations I have examined.

      b) Instead of using a single Chebyshev polynomial for the entire
         [0,1] y interval, we break the interval up into 100 equal
         subintervals, with a switch/lookup table, and use much lower
         degree Chebyshev polynomials in each subinterval. This greatly
         improves performance in my tests.

   For x < 0, we use the relationship erfcx(-x) = 2 exp(x^2) - erfc(x),
   with the usual checks for overflow etcetera.

   Performance-wise, it seems to be substantially faster than either
   the SLATEC DERFC function [or an erfcx function derived therefrom]
   or Cody's CALERF function (from netlib.org/specfun), while
   retaining near machine precision in accuracy.  */
"""
/* 给定 y100=100*y，其中 y = 4/(4+x)，x >= 0，计算 erfc(x)。

   使用一个包含 100 个不同的切比雪夫多项式的查找表，
   这些多项式分别适用于 y 区间 [0,0.01], [0.01,0.02], ...., [0.99,1]。
   这些多项式是通过 Maple 和一个小型 shell 脚本生成的。
   相比于使用单个多项式拟合整个 [0,1] 区间，这样可以显著降低多项式的次数（约为 1/4）。
*/

template <typename T>
C10_HOST_DEVICE  inline typename std::enable_if<std::is_floating_point<T>::value, T>::type
erfcx_y100(T y100)
{
  switch (static_cast<int>(y100)) {
case 0: {
T t = 2*y100 - 1;
return 0.70878032454106438663e-3 + (0.71234091047026302958e-3 + (0.35779077297597742384e-5 + (0.17403143962587937815e-7 + (0.81710660047307788845e-10 + (0.36885022360434957634e-12 + 0.15917038551111111111e-14 * t) * t) * t) * t) * t) * t;
}
case 1: {
T t = 2*y100 - 3;
return 0.21479143208285144230e-2 + (0.72686402367379996033e-3 + (0.36843175430938995552e-5 + (0.18071841272149201685e-7 + (0.85496449296040325555e-10 + (0.38852037518534291510e-12 + 0.16868473576888888889e-14 * t) * t) * t) * t) * t) * t;
}
case 2: {
T t = 2*y100 - 5;
return 0.36165255935630175090e-2 + (0.74182092323555510862e-3 + (0.37948319957528242260e-5 + (0.18771627021793087350e-7 + (0.89484715122415089123e-10 + (0.40935858517772440862e-12 + 0.17872061464888888889e-14 * t) * t) * t) * t) * t) * t;
}
case 3: {
T t = 2*y100 - 7;
return 0.51154983860031979264e-2 + (0.75722840734791660540e-3 + (0.39096425726735703941e-5 + (0.19504168704300468210e-7 + (0.93687503063178993915e-10 + (0.43143925959079664747e-12 + 0.18939926435555555556e-14 * t) * t) * t) * t) * t) * t;
}
case 4: {
T t = 2*y100 - 9;
return 0.66457513172673049824e-2 + (0.77310406054447454920e-3 + (0.40289510589399439385e-5 + (0.20271233238288381092e-7 + (0.98117631321709100264e-10 + (0.45484207406017752971e-12 + 0.20076352213333333333e-14 * t) * t) * t) * t) * t) * t;
}
case 5: {
T t = 2*y100 - 11;
return 0.82082389970241207883e-2 + (0.78946629611881710721e-3 + (0.41529701552622656574e-5 + (0.21074693344544655714e-7 + (0.10278874108587317989e-9 + (0.47965201390613339638e-12 + 0.21285907413333333333e-14 * t) * t) * t) * t) * t) * t;
}
case 6: {
T t = 2*y100 - 13;
return 0.98039537275352193165e-2 + (0.80633440108342840956e-3 + (0.42819241329736982942e-5 + (0.21916534346907168612e-7 + (0.10771535136565470914e-9 + (0.50595972623692822410e-12 + 0.22573462684444444444e-14 * t) * t) * t) * t) * t) * t;
}
case 7: {
T t = 2*y100 - 15;
return 0.11433927298290302370e-1 + (0.82372858383196561209e-3 + (0.44160495311765438816e-5 + (0.22798861426211986056e-7 + (0.11291291745879239736e-9 + (0.53386189365816880454e-12 + 0.23944209546666666667e-14 * t) * t) * t) * t) * t) * t;
}
case 8: {
T t = 2*y100 - 17;
# 返回一个多项式的值，t 是输入变量
return 0.13099232878814653979e-1 + (0.84167002467906968214e-3 + (0.45555958988457506002e-5 + (0.23723907357214175198e-7 + (0.11839789326602695603e-9 + (0.56346163067550237877e-12 + 0.25403679644444444444e-14 * t) * t) * t) * t) * t) * t;
}

# 当 case 为 9 时，根据输入变量计算并返回一个特定的多项式值
case 9: {
    T t = 2*y100 - 19;
    return 0.14800987015587535621e-1 + (0.86018092946345943214e-3 + (0.47008265848816866105e-5 + (0.24694040760197315333e-7 + (0.12418779768752299093e-9 + (0.59486890370320261949e-12 + 0.26957764568888888889e-14 * t) * t) * t) * t) * t) * t;
}

# 当 case 为 10 时，根据输入变量计算并返回一个特定的多项式值
case 10: {
    T t = 2*y100 - 21;
    return 0.16540351739394069380e-1 + (0.87928458641241463952e-3 + (0.48520195793001753903e-5 + (0.25711774900881709176e-7 + (0.13030128534230822419e-9 + (0.62820097586874779402e-12 + 0.28612737351111111111e-14 * t) * t) * t) * t) * t) * t;
}

# 当 case 为 11 时，根据输入变量计算并返回一个特定的多项式值
case 11: {
    T t = 2*y100 - 23;
    return 0.18318536789842392647e-1 + (0.89900542647891721692e-3 + (0.50094684089553365810e-5 + (0.26779777074218070482e-7 + (0.13675822186304615566e-9 + (0.66358287745352705725e-12 + 0.30375273884444444444e-14 * t) * t) * t) * t) * t) * t;
}

# 当 case 为 12 时，根据输入变量计算并返回一个特定的多项式值
case 12: {
    T t = 2*y100 - 25;
    return 0.20136801964214276775e-1 + (0.91936908737673676012e-3 + (0.51734830914104276820e-5 + (0.27900878609710432673e-7 + (0.14357976402809042257e-9 + (0.70114790311043728387e-12 + 0.32252476000000000000e-14 * t) * t) * t) * t) * t) * t;
}

# 当 case 为 13 时，根据输入变量计算并返回一个特定的多项式值
case 13: {
    T t = 2*y100 - 27;
    return 0.21996459598282740954e-1 + (0.94040248155366777784e-3 + (0.53443911508041164739e-5 + (0.29078085538049374673e-7 + (0.15078844500329731137e-9 + (0.74103813647499204269e-12 + 0.34251892320000000000e-14 * t) * t) * t) * t) * t) * t;
}

# 当 case 为 14 时，根据输入变量计算并返回一个特定的多项式值
case 14: {
    T t = 2*y100 - 29;
    return 0.23898877187226319502e-1 + (0.96213386835900177540e-3 + (0.55225386998049012752e-5 + (0.30314589961047687059e-7 + (0.15840826497296335264e-9 + (0.78340500472414454395e-12 + 0.36381553564444444445e-14 * t) * t) * t) * t) * t) * t;
}

# 当 case 为 15 时，根据输入变量计算并返回一个特定的多项式值
case 15: {
    T t = 2*y100 - 31;
    return 0.25845480155298518485e-1 + (0.98459293067820123389e-3 + (0.57082915920051843672e-5 + (0.31613782169164830118e-7 + (0.16646478745529630813e-9 + (0.82840985928785407942e-12 + 0.38649975768888888890e-14 * t) * t) * t) * t) * t) * t;
}

# 当 case 为 16 时，根据输入变量计算并返回一个特定的多项式值
case 16: {
    T t = 2*y100 - 33;
    return 0.27837754783474696598e-1 + (0.10078108563256892757e-2 + (0.59020366493792212221e-5 + (0.32979263553246520417e-7 + (0.17498524159268458073e-9 + (0.87622459124842525110e-12 + 0.41066206488888888890e-14 * t) * t) * t) * t) * t) * t;
}

# 当 case 为 17 时，根据输入变量计算并返回一个特定的多项式值
case 17: {
    T t = 2*y100 - 35;
    return 0.29877251304899307550e-1 + (0.10318204245057349310e-2 + (0.61041829697162055093e-5 + (0.34414860359542720579e-7 + (0.18399863072934089607e-9 + (0.92703227366365046533e-12 + 0.43639844053333333334e-14 * t) * t) * t) * t) * t) * t;
}

# 当 case 为 18 时，根据输入变量计算并返回一个特定的多项式值
case 18: {
    T t = 2*y100 - 37;
    return 0.31965587178596443475e-1 + (0.10566560976716574401e-2 + (0.63151633192414586770e-5 + (0.35924638339521924242e-7 + (0.19353584758781174038e-9 + (0.98102783859889264382e-12 + 0.46381060817777777779e-14 * t) * t) * t) * t) * t) * t;
}

# 当 case 为 19 时，根据输入变量计算并返回一个特定的多项式值
case 19: {
    T t = 2*y100 - 39;
# 返回一个多项式的值，具体的多项式表达式如下:
# 0.34104450552588334840e-1 + (0.10823541191350532574e-2 + (0.65354356159553934436e-5 +
# (0.37512918348533521149e-7 + (0.20362979635817883229e-9 + (0.10384187833037282363e-11 +
# 0.49300625262222222221e-14 * t) * t) * t) * t) * t) * t
return 0.34104450552588334840e-1 + (0.10823541191350532574e-2 + (0.65354356159553934436e-5 +
       (0.37512918348533521149e-7 + (0.20362979635817883229e-9 + (0.10384187833037282363e-11 +
       0.49300625262222222221e-14 * t) * t) * t) * t) * t) * t;
}

case 20: {
    # 计算特定情况下的 t 值
    T t = 2*y100 - 41;
    # 返回一个特定 t 值下的多项式的值，具体的多项式表达式如下:
    # 0.36295603928292425716e-1 + (0.11089526167995268200e-2 + (0.67654845095518363577e-5 +
    # (0.39184292949913591646e-7 + (0.21431552202133775150e-9 + (0.10994259106646731797e-11 +
    # 0.52409949102222222221e-14 * t) * t) * t) * t) * t) * t
    return 0.36295603928292425716e-1 + (0.11089526167995268200e-2 + (0.67654845095518363577e-5 +
           (0.39184292949913591646e-7 + (0.21431552202133775150e-9 + (0.10994259106646731797e-11 +
           0.52409949102222222221e-14 * t) * t) * t) * t) * t) * t;
}

case 21: {
    # 计算特定情况下的 t 值
    T t = 2*y100 - 43;
    # 返回一个特定 t 值下的多项式的值，具体的多项式表达式如下:
    # 0.38540888038840509795e-1 + (0.11364917134175420009e-2 + (0.70058230641246312003e-5 +
    # (0.40943644083718586939e-7 + (0.22563034723692881631e-9 + (0.11642841011361992885e-11 +
    # 0.55721092871111111110e-14 * t) * t) * t) * t) * t) * t
    return 0.38540888038840509795e-1 + (0.11364917134175420009e-2 + (0.70058230641246312003e-5 +
           (0.40943644083718586939e-7 + (0.22563034723692881631e-9 + (0.11642841011361992885e-11 +
           0.55721092871111111110e-14 * t) * t) * t) * t) * t) * t;
}

case 22: {
    # 计算特定情况下的 t 值
    T t = 2*y100 - 45;
    # 返回一个特定 t 值下的多项式的值，具体的多项式表达式如下:
    # 0.40842225954785960651e-1 + (0.11650136437945673891e-2 + (0.72569945502343006619e-5 +
    # (0.42796161861855042273e-7 + (0.23761401711005024162e-9 + (0.12332431172381557035e-11 +
    # 0.59246802364444444445e-14 * t) * t) * t) * t) * t) * t
    return 0.40842225954785960651e-1 + (0.11650136437945673891e-2 + (0.72569945502343006619e-5 +
           (0.42796161861855042273e-7 + (0.23761401711005024162e-9 + (0.12332431172381557035e-11 +
           0.59246802364444444445e-14 * t) * t) * t) * t) * t) * t;
}

case 23: {
    # 计算特定情况下的 t 值
    T t = 2*y100 - 47;
    # 返回一个特定 t 值下的多项式的值，具体的多项式表达式如下:
    # 0.43201627431540222422e-1 + (0.11945628793917272199e-2 + (0.75195743532849206263e-5 +
    # (0.44747364553960993492e-7 + (0.25030885216472953674e-9 + (0.13065684400300476484e-11 +
    # 0.63000532853333333334e-14 * t) * t) * t) * t) * t) * t
    return 0.43201627431540222422e-1 + (0.11945628793917272199e-2 + (0.75195743532849206263e-5 +
           (0.44747364553960993492e-7 + (0.25030885216472953674e-9 + (0.13065684400300476484e-11 +
           0.63000532853333333334e-14 * t) * t) * t) * t) * t) * t;
}

case 24: {
    # 计算特定情况下的 t 值
    T t = 2*y100 - 49;
    # 返回一个特定 t 值下的多项式的值，具体的多项式表达式如下:
    # 0.45621193513810471438e-1 + (0.12251862608067529503e-2 + (0.77941720055551920319e-5 +
    # (0.46803119830954460212e-7 + (0.26375990983978426273e-9 + (0.13845421370977119765e-11 +
    # 0.66996477404444444445e-14 * t) * t) * t) * t) * t) * t
    return 0.45621193513810471438e-1 + (0.12251862608067529503e-2 + (0.77941720055551920319e-5 +
           (0.46803119830954460212e-7 + (0.26375990983978426273e-9 + (0.13845421370977119765e-11 +
           0.66996477404444444445e-14 * t) * t) * t) * t) * t) * t;
}

case 25: {
    # 计算特定情况下的 t 值
    T t = 2*y100 - 51;
    # 返回一个特定 t 值下的多项式的值，具体的多项式表达式如下:
    # 0.48103121413299865517e-1 + (0.12569331386432195113e-2 + (0.80814333496367673980e-5 +
    # (0.48969667335682018324e-7 + (0.27801515481905748484e-9 + (0.14674637611609884208e-11 +
    # 0.71249589351111111110e-14 * t) * t) * t) * t) * t) * t
    return 0.48103121413299865517e-1 + (0.12569331386432195113e-2 + (0.80814333496367673980e-5 +
           (0.48969667335682018324e-7 + (0.27801515481905748484e-9 + (0.
# 对于给定的 y100 值，计算对应的 Legendre 多项式的值，分别返回不同阶数的多项式结果

case 31: {
    # 计算 t 值，这里使用了 2*y100 - 63 的公式
    T t = 2*y100 - 63;
    # 计算 Legendre 多项式 P_31(t) 的值
    return 0.64440817576653297993e-1 + (0.14741275456383131151e-2 + (0.10112293819576437838e-4 + (0.64698236605933246196e-7 + (0.38353412915303665586e-9 + (0.20881176114385120186e-11 + 0.10310784480000000000e-13 * t) * t) * t) * t) * t) * t;
}

case 32: {
    # 计算 t 值，这里使用了 2*y100 - 65 的公式
    T t = 2*y100 - 65;
    # 计算 Legendre 多项式 P_32(t) 的值
    return 0.67430045633130393282e-1 + (0.15153655418916540370e-2 + (0.10509857606888328667e-4 + (0.67851706529363332855e-7 + (0.40504602194811140006e-9 + (0.22157325110542534469e-11 + 0.10964842115555555556e-13 * t) * t) * t) * t) * t) * t;
}

case 33: {
    # 计算 t 值，这里使用了 2*y100 - 67 的公式
    T t = 2*y100 - 67;
    # 计算 Legendre 多项式 P_33(t) 的值
    return 0.70503365513338850709e-1 + (0.15582323336495709827e-2 + (0.10926868866865231089e-4 + (0.71182482239613507542e-7 + (0.42787405890153386710e-9 + (0.23514379522274416437e-11 + 0.11659571751111111111e-13 * t) * t) * t) * t) * t) * t;
}

case 34: {
    # 计算 t 值，这里使用了 2*y100 - 69 的公式
    T t = 2*y100 - 69;
    # 计算 Legendre 多项式 P_34(t) 的值
    return 0.73664114037944596353e-1 + (0.16028078812438820413e-2 + (0.11364423678778207991e-4 + (0.74701423097423182009e-7 + (0.45210162777476488324e-9 + (0.24957355004088569134e-11 + 0.12397238257777777778e-13 * t) * t) * t) * t) * t) * t;
}

case 35: {
    # 计算 t 值，这里使用了 2*y100 - 71 的公式
    T t = 2*y100 - 71;
    # 计算 Legendre 多项式 P_35(t) 的值
    return 0.76915792420819562379e-1 + (0.16491766623447889354e-2 + (0.11823685320041302169e-4 + (0.78420075993781544386e-7 + (0.47781726956916478925e-9 + (0.26491544403815724749e-11 + 0.13180196462222222222e-13 * t) * t) * t) * t) * t) * t;
}

case 36: {
    # 计算 t 值，这里使用了 2*y100 - 73 的公式
    T t = 2*y100 - 73;
    # 计算 Legendre 多项式 P_36(t) 的值
    return 0.80262075578094612819e-1 + (0.16974279491709504117e-2 + (0.12305888517309891674e-4 + (0.82350717698979042290e-7 + (0.50511496109857113929e-9 + (0.28122528497626897696e-11 + 0.14010889635555555556e-13 * t) * t) * t) * t) * t) * t;
}

case 37: {
    # 计算 t 值，这里使用了 2*y100 - 75 的公式
    T t = 2*y100 - 75;
    # 计算 Legendre 多项式 P_37(t) 的值
    return 0.83706822008980357446e-1 + (0.17476561032212656962e-2 + (0.12812343958540763368e-4 + (0.86506399515036435592e-7 + (0.53409440823869467453e-9 + (0.29856186620887555043e-11 + 0.14891851591111111111e-13 * t) * t) * t) * t) * t) * t;
}

case 38: {
    # 计算 t 值，这里使用了 2*y100 - 77 的公式
    T t = 2*y100 - 77;
    # 计算 Legendre 多项式 P_38(t) 的值
    return 0.87254084284461718231e-1 + (0.17999608886001962327e-2 + (0.13344443080089492218e-4 + (0.90900994316429008631e-7 + (0.56486134972616465316e-9 + (0.31698707080033956934e-11 + 0.15825697795555555556e-13 * t) * t) * t) * t) * t) * t;
}

case 39: {
    # 计算 t 值，这里使用了 2*y100 - 79 的公式
    T t = 2*y100 - 79;
    # 计算 Legendre 多项式 P_39(t) 的值
    return 0.90908120182172748487e-1 + (0.18544478050657699758e-2 + (0.13903663143426120077e-4 + (0.95549246062549906177e-7 + (0.59752787125242054315e-9 + (0.33656597366099099413e-11 + 0.16815130613333333333e-13 * t) * t) * t) * t) * t) * t;
}

case 40: {
    # 计算 t 值，这里使用了 2*y100 - 81 的公式
    T t = 2*y100 - 81;
    # 计算 Legendre 多项式 P_40(t) 的值
    return 0.94673404508075481121e-1 + (0.19112284419887303347e-2 + (0.14491572616545004930e-4 + (0.10046682186333613697e-6 + (0.63221272959791000515e-9 + (0.35736693975589130818e-11 + 0.17862931591111111111e-13 * t) * t) * t) * t) * t) * t;
}

case 41: {
    # 计算 t 值，这里使用了 2*y100 - 83 的公式
    T t = 2*y100 - 83;
    # 计算 Legendre 多项式 P_41(t) 的值
    return 0.98584827706987241877e-1 + (0.19717719882359187211e-2 + (0.15142155816418725021e-4 + (0.10521499311862569365e-6 + (0.67104773145851668974e-9 + (0.37772264481776599163e-11 + 0.18981478257777777778e-13 * t) * t) * t) * t) * t) * t;
}
# 返回一个多项式函数的值，根据不同的 case 选择不同的多项式系数
return 0.98554641648004456555e-1 + (0.19704208544725622126e-2 + (0.15109836875625443935e-4 + (0.10567036667675984067e-6 + (0.66904168640019354565e-9 + (0.37946171850824333014e-11 + 0.18971959040000000000e-13 * t) * t) * t) * t) * t) * t;
}
case 42: {
# 计算 t 的值
T t = 2*y100 - 85;
# 返回一个多项式函数的值，根据 t 选择不同的多项式系数
return 0.10255677889470089531e0 + (0.20321499629472857418e-2 + (0.15760224242962179564e-4 + (0.11117756071353507391e-6 + (0.70814785110097658502e-9 + (0.40292553276632563925e-11 + 0.20145143075555555556e-13 * t) * t) * t) * t) * t) * t;
}
case 43: {
# 计算 t 的值
T t = 2*y100 - 87;
# 返回一个多项式函数的值，根据 t 选择不同的多项式系数
return 0.10668502059865093318e0 + (0.20965479776148731610e-2 + (0.16444612377624983565e-4 + (0.11700717962026152749e-6 + (0.74967203250938418991e-9 + (0.42783716186085922176e-11 + 0.21385479360000000000e-13 * t) * t) * t) * t) * t) * t;
}
case 44: {
# 计算 t 的值
T t = 2*y100 - 89;
# 返回一个多项式函数的值，根据 t 选择不同的多项式系数
return 0.11094484319386444474e0 + (0.21637548491908170841e-2 + (0.17164995035719657111e-4 + (0.12317915750735938089e-6 + (0.79376309831499633734e-9 + (0.45427901763106353914e-11 + 0.22696025653333333333e-13 * t) * t) * t) * t) * t) * t;
}
case 45: {
# 计算 t 的值
T t = 2*y100 - 91;
# 返回一个多项式函数的值，根据 t 选择不同的多项式系数
return 0.11534201115268804714e0 + (0.22339187474546420375e-2 + (0.17923489217504226813e-4 + (0.12971465288245997681e-6 + (0.84057834180389073587e-9 + (0.48233721206418027227e-11 + 0.24079890062222222222e-13 * t) * t) * t) * t) * t) * t;
}
case 46: {
# 计算 t 的值
T t = 2*y100 - 93;
# 返回一个多项式函数的值，根据 t 选择不同的多项式系数
return 0.11988259392684094740e0 + (0.23071965691918689601e-2 + (0.18722342718958935446e-4 + (0.13663611754337957520e-6 + (0.89028385488493287005e-9 + (0.51210161569225846701e-11 + 0.25540227111111111111e-13 * t) * t) * t) * t) * t) * t;
}
case 47: {
# 计算 t 的值
T t = 2*y100 - 95;
# 返回一个多项式函数的值，根据 t 选择不同的多项式系数
return 0.12457298393509812907e0 + (0.23837544771809575380e-2 + (0.19563942105711612475e-4 + (0.14396736847739470782e-6 + (0.94305490646459247016e-9 + (0.54366590583134218096e-11 + 0.27080225920000000000e-13 * t) * t) * t) * t) * t) * t;
}
case 48: {
# 计算 t 的值
T t = 2*y100 - 97;
# 返回一个多项式函数的值，根据 t 选择不同的多项式系数
return 0.12941991566142438816e0 + (0.24637684719508859484e-2 + (0.20450821127475879816e-4 + (0.15173366280523906622e-6 + (0.99907632506389027739e-9 + (0.57712760311351625221e-11 + 0.28703099555555555556e-13 * t) * t) * t) * t) * t) * t;
}
case 49: {
# 计算 t 的值
T t = 2*y100 - 99;
# 返回一个多项式函数的值，根据 t 选择不同的多项式系数
return 0.13443048593088696613e0 + (0.25474249981080823877e-2 + (0.21385669591362915223e-4 + (0.15996177579900443030e-6 + (0.10585428844575134013e-8 + (0.61258809536787882989e-11 + 0.30412080142222222222e-13 * t) * t) * t) * t) * t) * t;
}
case 50: {
# 计算 t 的值
T t = 2*y100 - 101;
# 返回一个多项式函数的值，根据 t 选择不同的多项式系数
return 0.13961217543434561353e0 + (0.26349215871051761416e-2 + (0.22371342712572567744e-4 + (0.16868008199296822247e-6 + (0.11216596910444996246e-8 + (0.65015264753090890662e-11 + 0.32210394506666666666e-13 * t) * t) * t) * t) * t) * t;
}
case 51: {
# 计算 t 的值
T t = 2*y100 - 103;
# 返回一个多项式函数的值，根据 t 选择不同的多项式系数
return 0.14497287157673800690e0 + (0.27264675383982439814e-2 + (0.23410870961050950197e-4 + (0.17791863939526376477e-6 + (0.11886425714330958106e-8 + (0.68993039665054288034e-11 + 0.34101266222222222221e-13 * t) * t) * t) * t) * t) * t;
}
case 52: {
# 计算 t 的值
T t = 2*y100 - 105;
# 返回一个多项式函数的值，根据不同的情况选择不同的多项式
return 0.15052089272774618151e0 + (0.28222846410136238008e-2 + (0.24507470422713397006e-4 + (0.18770927679626136909e-6 + (0.12597184587583370712e-8 + (0.73203433049229821618e-11 + 0.36087889048888888890e-13 * t) * t) * t) * t) * t) * t;
}
case 53: {
# 设置变量 t 为 2*y100 - 107
T t = 2*y100 - 107;
# 返回特定条件下的多项式函数值
return 0.15626501395774612325e0 + (0.29226079376196624949e-2 + (0.25664553693768450545e-4 + (0.19808568415654461964e-6 + (0.13351257759815557897e-8 + (0.77658124891046760667e-11 + 0.38173420035555555555e-13 * t) * t) * t) * t) * t) * t;
}
case 54: {
# 设置变量 t 为 2*y100 - 109
T t = 2*y100 - 109;
# 返回特定条件下的多项式函数值
return 0.16221449434620737567e0 + (0.30276865332726475672e-2 + (0.26885741326534564336e-4 + (0.20908350604346384143e-6 + (0.14151148144240728728e-8 + (0.82369170665974313027e-11 + 0.40360957457777777779e-13 * t) * t) * t) * t) * t) * t;
}
case 55: {
# 设置变量 t 为 2*y100 - 111
T t = 2*y100 - 111;
# 返回特定条件下的多项式函数值
return 0.16837910595412130659e0 + (0.31377844510793082301e-2 + (0.28174873844911175026e-4 + (0.22074043807045782387e-6 + (0.14999481055996090039e-8 + (0.87348993661930809254e-11 + 0.42653528977777777779e-13 * t) * t) * t) * t) * t) * t;
}
case 56: {
# 设置变量 t 为 2*y100 - 113
T t = 2*y100 - 113;
# 返回特定条件下的多项式函数值
return 0.17476916455659369953e0 + (0.32531815370903068316e-2 + (0.29536024347344364074e-4 + (0.23309632627767074202e-6 + (0.15899007843582444846e-8 + (0.92610375235427359475e-11 + 0.45054073102222222221e-13 * t) * t) * t) * t) * t) * t;
}
case 57: {
# 设置变量 t 为 2*y100 - 115
T t = 2*y100 - 115;
# 返回特定条件下的多项式函数值
return 0.18139556223643701364e0 + (0.33741744168096996041e-2 + (0.30973511714709500836e-4 + (0.24619326937592290996e-6 + (0.16852609412267750744e-8 + (0.98166442942854895573e-11 + 0.47565418097777777779e-13 * t) * t) * t) * t) * t) * t;
}
case 58: {
# 设置变量 t 为 2*y100 - 117
T t = 2*y100 - 117;
# 返回特定条件下的多项式函数值
return 0.18826980194443664549e0 + (0.35010775057740317997e-2 + (0.32491914440014267480e-4 + (0.26007572375886319028e-6 + (0.17863299617388376116e-8 + (0.10403065638343878679e-10 + 0.50190265831111111110e-13 * t) * t) * t) * t) * t) * t;
}
case 59: {
# 设置变量 t 为 2*y100 - 119
T t = 2*y100 - 119;
# 返回特定条件下的多项式函数值
return 0.19540403413693967350e0 + (0.36342240767211326315e-2 + (0.34096085096200907289e-4 + (0.27479061117017637474e-6 + (0.18934228504790032826e-8 + (0.11021679075323598664e-10 + 0.52931171733333333334e-13 * t) * t) * t) * t) * t) * t;
}
case 60: {
# 设置变量 t 为 2*y100 - 121
T t = 2*y100 - 121;
# 返回特定条件下的多项式函数值
return 0.20281109560651886959e0 + (0.37739673859323597060e-2 + (0.35791165457592409054e-4 + (0.29038742889416172404e-6 + (0.20068685374849001770e-8 + (0.11673891799578381999e-10 + 0.55790523093333333334e-13 * t) * t) * t) * t) * t) * t;
}
case 61: {
# 设置变量 t 为 2*y100 - 123
T t = 2*y100 - 123;
# 返回特定条件下的多项式函数值
return 0.21050455062669334978e0 + (0.39206818613925652425e-2 + (0.37582602289680101704e-4 + (0.30691836231886877385e-6 + (0.21270101645763677824e-8 + (0.12361138551062899455e-10 + 0.58770520160000000000e-13 * t) * t) * t) * t) * t) * t;
}
case 62: {
# 设置变量 t 为 2*y100 - 125
T t = 2*y100 - 125;
# 返回特定条件下的多项式函数值
return 0.21849873453703332479e0 + (0.40747643554689586041e-2 + (0.39476163820986711501e-4 + (0.32443839970139918836e-6 + (0.22542053491518680200e-8 + (0.13084879235290858490e-10 + 0.61873153262222222221e-13 * t) * t) * t) * t) * t) * t;
}
case 63: {
# 设置变量 t 为 2*y100 - 127
T t = 2*y100 - 127;
# 返回一个多项式的值，使用传入的参数 t 计算
return 0.22680879990043229327e0 + (0.42366354648628516935e-2 + (0.41477956909656896779e-4 + (0.34300544894502810002e-6 + (0.23888264229264067658e-8 + (0.13846596292818514601e-10 + 0.65100183751111111110e-13 * t) * t) * t) * t) * t) * t;
}
case 64: {
# 计算特定情况下的多项式值，t 的值通过给定的数学表达式计算得到
T t = 2*y100 - 129;
return 0.23545076536988703937e0 + (0.44067409206365170888e-2 + (0.43594444916224700881e-4 + (0.36268045617760415178e-6 + (0.25312606430853202748e-8 + (0.14647791812837903061e-10 + 0.68453122631111111110e-13 * t) * t) * t) * t) * t) * t;
}
case 65: {
# 计算特定情况下的多项式值，t 的值通过给定的数学表达式计算得到
T t = 2*y100 - 131;
return 0.24444156740777432838e0 + (0.45855530511605787178e-2 + (0.45832466292683085475e-4 + (0.38352752590033030472e-6 + (0.26819103733055603460e-8 + (0.15489984390884756993e-10 + 0.71933206364444444445e-13 * t) * t) * t) * t) * t) * t;
}
case 66: {
# 计算特定情况下的多项式值，t 的值通过给定的数学表达式计算得到
T t = 2*y100 - 133;
return 0.25379911500634264643e0 + (0.47735723208650032167e-2 + (0.48199253896534185372e-4 + (0.40561404245564732314e-6 + (0.28411932320871165585e-8 + (0.16374705736458320149e-10 + 0.75541379822222222221e-13 * t) * t) * t) * t) * t) * t;
}
case 67: {
# 计算特定情况下的多项式值，t 的值通过给定的数学表达式计算得到
T t = 2*y100 - 135;
return 0.26354234756393613032e0 + (0.49713289477083781266e-2 + (0.50702455036930367504e-4 + (0.42901079254268185722e-6 + (0.30095422058900481753e-8 + (0.17303497025347342498e-10 + 0.79278273368888888890e-13 * t) * t) * t) * t) * t) * t;
}
case 68: {
# 计算特定情况下的多项式值，t 的值通过给定的数学表达式计算得到
T t = 2*y100 - 137;
return 0.27369129607732343398e0 + (0.51793846023052643767e-2 + (0.53350152258326602629e-4 + (0.45379208848865015485e-6 + (0.31874057245814381257e-8 + (0.18277905010245111046e-10 + 0.83144182364444444445e-13 * t) * t) * t) * t) * t) * t;
}
case 69: {
# 计算特定情况下的多项式值，t 的值通过给定的数学表达式计算得到
T t = 2*y100 - 139;
return 0.28426714781640316172e0 + (0.53983341916695141966e-2 + (0.56150884865255810638e-4 + (0.48003589196494734238e-6 + (0.33752476967570796349e-8 + (0.19299477888083469086e-10 + 0.87139049137777777779e-13 * t) * t) * t) * t) * t) * t;
}
case 70: {
# 计算特定情况下的多项式值，t 的值通过给定的数学表达式计算得到
T t = 2*y100 - 141;
return 0.29529231465348519920e0 + (0.56288077305420795663e-2 + (0.59113671189913307427e-4 + (0.50782393781744840482e-6 + (0.35735475025851713168e-8 + (0.20369760937017070382e-10 + 0.91262442613333333334e-13 * t) * t) * t) * t) * t) * t;
}
case 71: {
# 计算特定情况下的多项式值，t 的值通过给定的数学表达式计算得到
T t = 2*y100 - 143;
return 0.30679050522528838613e0 + (0.58714723032745403331e-2 + (0.62248031602197686791e-4 + (0.53724185766200945789e-6 + (0.37827999418960232678e-8 + (0.21490291930444538307e-10 + 0.95513539182222222221e-13 * t) * t) * t) * t) * t) * t;
}
case 72: {
# 计算特定情况下的多项式值，t 的值通过给定的数学表达式计算得到
T t = 2*y100 - 145;
return 0.31878680111173319425e0 + (0.61270341192339103514e-2 + (0.65564012259707640976e-4 + (0.56837930287837738996e-6 + (0.40035151353392378882e-8 + (0.22662596341239294792e-10 + 0.99891109760000000000e-13 * t) * t) * t) * t) * t) * t;
}
case 73: {
# 计算特定情况下的多项式值，t 的值通过给定的数学表达式计算得到
T t = 2*y100 - 147;
return 0.33130773722152622027e0 + (0.63962406646798080903e-2 + (0.69072209592942396666e-4 + (0.60133006661885941812e-6 + (0.42362183765883466691e-8 + (0.23888182347073698382e-10 + 0.10439349811555555556e-12 * t) * t) * t) * t) * t) * t;
}
case 74: {
# 计算特定情况下的多项式值，t 的值通过给定的数学表达式计算得到
T t = 2*y100 - 149;
// 返回一个多项式的值，其中 t 是输入的参数
return 0.34438138658041336523e0 + (0.66798829540414007258e-2 + (0.72783795518603561144e-4 + (0.63619220443228800680e-6 + (0.44814499336514453364e-8 + (0.25168535651285475274e-10 + 0.10901861383111111111e-12 * t) * t) * t) * t) * t) * t;
}
case 75: {
// 计算 t 并使用其值计算另一个多项式，t 的计算基于输入 y100
T t = 2*y100 - 151;
return 0.35803744972380175583e0 + (0.69787978834882685031e-2 + (0.76710543371454822497e-4 + (0.67306815308917386747e-6 + (0.47397647975845228205e-8 + (0.26505114141143050509e-10 + 0.11376390933333333333e-12 * t) * t) * t) * t) * t) * t;
}
case 76: {
// 计算 t 并使用其值计算另一个多项式，t 的计算基于输入 y100
T t = 2*y100 - 153;
return 0.37230734890119724188e0 + (0.72938706896461381003e-2 + (0.80864854542670714092e-4 + (0.71206484718062688779e-6 + (0.50117323769745883805e-8 + (0.27899342394100074165e-10 + 0.11862637614222222222e-12 * t) * t) * t) * t) * t) * t;
}
case 77: {
// 计算 t 并使用其值计算另一个多项式，t 的计算基于输入 y100
T t = 2*y100 - 155;
return 0.38722432730555448223e0 + (0.76260375162549802745e-2 + (0.85259785810004603848e-4 + (0.75329383305171327677e-6 + (0.52979361368388119355e-8 + (0.29352606054164086709e-10 + 0.12360253370666666667e-12 * t) * t) * t) * t) * t) * t;
}
case 78: {
// 计算 t 并使用其值计算另一个多项式，t 的计算基于输入 y100
T t = 2*y100 - 157;
return 0.40282355354616940667e0 + (0.79762880915029728079e-2 + (0.89909077342438246452e-4 + (0.79687137961956194579e-6 + (0.55989731807360403195e-8 + (0.30866246101464869050e-10 + 0.12868841946666666667e-12 * t) * t) * t) * t) * t) * t;
}
case 79: {
// 计算 t 并使用其值计算另一个多项式，t 的计算基于输入 y100
T t = 2*y100 - 159;
return 0.41914223158913787649e0 + (0.83456685186950463538e-2 + (0.94827181359250161335e-4 + (0.84291858561783141014e-6 + (0.59154537751083485684e-8 + (0.32441553034347469291e-10 + 0.13387957943111111111e-12 * t) * t) * t) * t) * t) * t;
}
case 80: {
// 计算 t 并使用其值计算另一个多项式，t 的计算基于输入 y100
T t = 2*y100 - 161;
return 0.43621971639463786896e0 + (0.87352841828289495773e-2 + (0.10002929142066799966e-3 + (0.89156148280219880024e-6 + (0.62480008150788597147e-8 + (0.34079760983458878910e-10 + 0.13917107176888888889e-12 * t) * t) * t) * t) * t) * t;
}
case 81: {
// 计算 t 并使用其值计算另一个多项式，t 的计算基于输入 y100
T t = 2*y100 - 163;
return 0.45409763548534330981e0 + (0.91463027755548240654e-2 + (0.10553137232446167258e-3 + (0.94293113464638623798e-6 + (0.65972492312219959885e-8 + (0.35782041795476563662e-10 + 0.14455745872000000000e-12 * t) * t) * t) * t) * t) * t;
}
case 82: {
// 计算 t 并使用其值计算另一个多项式，t 的计算基于输入 y100
T t = 2*y100 - 165;
return 0.47282001668512331468e0 + (0.95799574408860463394e-2 + (0.11135019058000067469e-3 + (0.99716373005509038080e-6 + (0.69638453369956970347e-8 + (0.37549499088161345850e-10 + 0.15003280712888888889e-12 * t) * t) * t) * t) * t) * t;
}
case 83: {
// 计算 t 并使用其值计算另一个多项式，t 的计算基于输入 y100
T t = 2*y100 - 167;
return 0.49243342227179841649e0 + (0.10037550043909497071e-1 + (0.11750334542845234952e-3 + (0.10544006716188967172e-5 + (0.73484461168242224872e-8 + (0.39383162326435752965e-10 + 0.15559069118222222222e-12 * t) * t) * t) * t) * t) * t;
}
case 84: {
// 计算 t 并使用其值计算另一个多项式，t 的计算基于输入 y100
T t = 2*y100 - 169;
return 0.51298708979209258326e0 + (0.10520454564612427224e-1 + (0.12400930037494996655e-3 + (0.11147886579371265246e-5 + (0.77517184550568711454e-8 + (0.41283980931872622611e-10 + 0.16122419680000000000e-12 * t) * t) * t) * t) * t) * t;
}
case 85: {
// 计算 t 并使用其值计算另一个多项式，t 的计算基于输入 y100
T t = 2*y100 - 171;
# 返回一个多项式表达式的值，其中 t 是输入的参数
return 0.53453307979101369843e0 + (0.11030120618800726938e-1 + (0.13088741519572269581e-3 + (0.11784797595374515432e-5 + (0.81743383063044825400e-8 + (0.43252818449517081051e-10 + 0.16692592640000000000e-12 * t) * t) * t) * t) * t) * t;
}
case 86: {
# 计算 t 值，然后返回对应的多项式表达式的值
T t = 2*y100 - 173;
return 0.55712643071169299478e0 + (0.11568077107929735233e-1 + (0.13815797838036651289e-3 + (0.12456314879260904558e-5 + (0.86169898078969313597e-8 + (0.45290446811539652525e-10 + 0.17268801084444444444e-12 * t) * t) * t) * t) * t) * t;
}
case 87: {
# 计算 t 值，然后返回对应的多项式表达式的值
T t = 2*y100 - 175;
return 0.58082532122519320968e0 + (0.12135935999503877077e-1 + (0.14584223996665838559e-3 + (0.13164068573095710742e-5 + (0.90803643355106020163e-8 + (0.47397540713124619155e-10 + 0.17850211608888888889e-12 * t) * t) * t) * t) * t) * t;
}
case 88: {
# 计算 t 值，然后返回对应的多项式表达式的值
T t = 2*y100 - 177;
return 0.60569124025293375554e0 + (0.12735396239525550361e-1 + (0.15396244472258863344e-3 + (0.13909744385382818253e-5 + (0.95651595032306228245e-8 + (0.49574672127669041550e-10 + 0.18435945564444444444e-12 * t) * t) * t) * t) * t) * t;
}
case 89: {
# 计算 t 值，然后返回对应的多项式表达式的值
T t = 2*y100 - 179;
return 0.63178916494715716894e0 + (0.13368247798287030927e-1 + (0.16254186562762076141e-3 + (0.14695084048334056083e-5 + (0.10072078109604152350e-7 + (0.51822304995680707483e-10 + 0.19025081422222222222e-12 * t) * t) * t) * t) * t) * t;
}
case 90: {
# 计算 t 值，然后返回对应的多项式表达式的值
T t = 2*y100 - 181;
return 0.65918774689725319200e0 + (0.14036375850601992063e-1 + (0.17160483760259706354e-3 + (0.15521885688723188371e-5 + (0.10601827031535280590e-7 + (0.54140790105837520499e-10 + 0.19616655146666666667e-12 * t) * t) * t) * t) * t) * t;
}
case 91: {
# 计算 t 值，然后返回对应的多项式表达式的值
T t = 2*y100 - 183;
return 0.68795950683174433822e0 + (0.14741765091365869084e-1 + (0.18117679143520433835e-3 + (0.16392004108230585213e-5 + (0.11155116068018043001e-7 + (0.56530360194925690374e-10 + 0.20209663662222222222e-12 * t) * t) * t) * t) * t) * t;
}
case 92: {
# 计算 t 值，然后返回对应的多项式表达式的值
T t = 2*y100 - 185;
return 0.71818103808729967036e0 + (0.15486504187117112279e-1 + (0.19128428784550923217e-3 + (0.17307350969359975848e-5 + (0.11732656736113607751e-7 + (0.58991125287563833603e-10 + 0.20803065333333333333e-12 * t) * t) * t) * t) * t) * t;
}
case 93: {
# 计算 t 值，然后返回对应的多项式表达式的值
T t = 2*y100 - 187;
return 0.74993321911726254661e0 + (0.16272790364044783382e-1 + (0.20195505163377912645e-3 + (0.18269894883203346953e-5 + (0.12335161021630225535e-7 + (0.61523068312169087227e-10 + 0.21395783431111111111e-12 * t) * t) * t) * t) * t) * t;
}
case 94: {
# 计算 t 值，然后返回对应的多项式表达式的值
T t = 2*y100 - 189;
return 0.78330143531283492729e0 + (0.17102934132652429240e-1 + (0.21321800585063327041e-3 + (0.19281661395543913713e-5 + (0.12963340087354341574e-7 + (0.64126040998066348872e-10 + 0.21986708942222222222e-12 * t) * t) * t) * t) * t) * t;
}
case 95: {
# 计算 t 值，然后返回对应的多项式表达式的值
T t = 2*y100 - 191;
return 0.81837581041023811832e0 + (0.17979364149044223802e-1 + (0.22510330592753129006e-3 + (0.20344732868018175389e-5 + (0.13617902941839949718e-7 + (0.66799760083972474642e-10 + 0.22574701262222222222e-12 * t) * t) * t) * t) * t) * t;
}
case 96: {
# 计算 t 值，然后返回对应的多项式表达式的值
T t = 2*y100 - 193;
// 根据输入的参数 x，计算对应的 airy 气函数前向传播值
template<typename T>
inline C10_HOST_DEVICE T airy_ai_forward(T x) {
    // airy 气函数的系数 AN 数组，用于计算函数值
    static const T AN[] = {
            +3.46538101525629032477e-01,
            +1.20075952739645805542e+01,
            +7.62796053615234516538e+01,
            +1.68089224934630576269e+02,
            +1.59756391350164413639e+02,
            +7.05360906840444183113e+01,
            +1.40264691163389668864e+01,
            +9.99999999999999995305e-01,
    };

    // 返回 airy 气函数的前向传播值
    static const T y100 = 100;
    switch (int(x * y100)) {
        case 97: {
            // 计算 t 值，用于 airy 气函数的系数 AN 的应用
            T t = 2 * y100 - 195;
            return 0.89402868170849933734e0 + (0.19881418399127202569e-1 + (0.25086793128395995798e-3 + (0.22633402747585233180e-5 + (0.15008997042116532283e-7 + (0.72357609075043941261e-10 + 0.23737194737777777778e-12 * t) * t) * t) * t) * t) * t;
        }
        case 98: {
            // 计算 t 值，用于 airy 气函数的系数 AN 的应用
            T t = 2 * y100 - 197;
            return 0.93481333942870796363e0 + (0.20912536329780368893e-1 + (0.26481403465998477969e-3 + (0.23863447359754921676e-5 + (0.15746923065472184451e-7 + (0.75240468141720143653e-10 + 0.24309291271111111111e-12 * t) * t) * t) * t) * t) * t;
        }
        case 99: {
            // 计算 t 值，用于 airy 气函数的系数 AN 的应用
            T t = 2 * y100 - 199;
            return 0.97771701335885035464e0 + (0.22000938572830479551e-1 + (0.27951610702682383001e-3 + (0.25153688325245314530e-5 + (0.16514019547822821453e-7 + (0.78191526829368231251e-10 + 0.24873652355555555556e-12 * t) * t) * t) * t) * t) * t;
        }
    }
    // 如果 y = 1，即 |x| < 4*eps 时，返回 1.0，表示 erfcx 在 1e-15 以内接近 1
    return 1.0;
}

/*
 * Logarithm of Gaussian cumulative distribution function.
 *
 * This implementation of log_ndtr and its helper functions
 * follow SciPy's implementation
 * See NOTICE for the licenses.
 */
template <typename T>
inline C10_HOST_DEVICE T calc_log_ndtr(T x) {
    // 计算参数 t，用于简化计算 log_ndtr 函数
    T t = x * c10::frac_sqrt_2<T>;
    // 判断 x 的值，根据不同情况返回对数正态分布函数的值
    if (x < T{-1.0}) {
        return std::log(calc_erfcx(-t) / 2) - t * t;
    } else {
        return std::log1p(-std::erfc(t) / 2);
    }
}
    // 系数数组 AD，存储第一组系数数据
    static const T AD[] = {
            +5.67594532638770212846e-01,
            +1.47562562584847203173e+01,
            +8.45138970141474626562e+01,
            +1.77318088145400459522e+02,
            +1.64234692871529701831e+02,
            +7.14778400825575695274e+01,
            +1.40959135607834029598e+01,
            +1.00000000000000000470e+00,
    };

    // 系数数组 AFN，存储第二组系数数据
    static const T AFN[] = {
            -1.31696323418331795333e-01,
            -6.26456544431912369773e-01,
            -6.93158036036933542233e-01,
            -2.79779981545119124951e-01,
            -4.91900132609500318020e-02,
            -4.06265923594885404393e-03,
            -1.59276496239262096340e-04,
            -2.77649108155232920844e-06,
            -1.67787698489114633780e-08,
    };

    // 系数数组 AFD，存储第三组系数数据
    static const T AFD[] = {
            +1.33560420706553243746e+01,
            +3.26825032795224613948e+01,
            +2.67367040941499554804e+01,
            +9.18707402907259625840e+00,
            +1.47529146771666414581e+00,
            +1.15687173795188044134e-01,
            +4.40291641615211203805e-03,
            +7.54720348287414296618e-05,
            +4.51850092970580378464e-07,
    };

    // 系数数组 AGN，存储第四组系数数据
    static const T AGN[] = {
            +1.97339932091685679179e-02,
            +3.91103029615688277255e-01,
            +1.06579897599595591108e+00,
            +9.39169229816650230044e-01,
            +3.51465656105547619242e-01,
            +6.33888919628925490927e-02,
            +5.85804113048388458567e-03,
            +2.82851600836737019778e-04,
            +6.98793669997260967291e-06,
            +8.11789239554389293311e-08,
            +3.41551784765923618484e-10,
    };

    // 系数数组 AGD，存储第五组系数数据
    static const T AGD[] = {
            +9.30892908077441974853e+00,
            +1.98352928718312140417e+01,
            +1.55646628932864612953e+01,
            +5.47686069422975497931e+00,
            +9.54293611618961883998e-01,
            +8.64580826352392193095e-02,
            +4.12656523824222607191e-03,
            +1.01259085116509135510e-04,
            +1.17166733214413521882e-06,
            +4.91834570062930015649e-09,
    };

    // 域标志初始化为零
    int domain_flag = 0;

    // 声明变量 ai，用于后续计算
    T ai;

    // 如果 x 是无穷大，返回 NaN
    if (std::isinf(x)) {
        return std::numeric_limits<T>::quiet_NaN();
    }

    // 如果 x 大于 103.892，返回 0.0
    if (x > T(103.892)) {
        return T(0.0);
    }

    // 声明变量 f，g，k，用于后续计算
    T f;
    T g;
    T k;
    # 如果 x 小于 -2.09，进入该条件分支
    if (x < T(-2.09)) {
        # 计算 z 值，z = 1 / (-2 * x * sqrt(-x) / 3)
        T z = T(1.0) / (T(-2.0) * x * std::sqrt(-x) / T(3.0));

        # 初始化 afn 为 0
        T afn = 0.0;

        # 计算 AFN 多项式的值
        for (uint8_t index = 0; index <= 8; index++) {
            afn = afn * (z * z) + AFN[index];
        }

        # 初始化 afd 为 0
        T afd = 0.0;

        # 计算 AFD 多项式的值
        for (uint8_t index = 0; index <= 8; index++) {
            afd = afd * (z * z) + AFD[index];
        }

        # 初始化 agn 为 0
        T agn = 0.0;

        # 计算 AGN 多项式的值
        for (uint8_t index = 0; index <= 10 + 0; index++) {
            agn = agn * (z * z) + AGN[index];
        }

        # 初始化 agd 为 0
        T agd = 0.0;

        # 计算 AGD 多项式的值
        for (uint8_t index = 0; index <= 10 - 1; index++) {
            agd = agd * (z * z) + AGD[index];
        }

        # 计算 t 值，t = -2 * x * sqrt(-x) / 3 + 0.25 * pi
        T t = T(-2.0) * x * std::sqrt(-x) / T(3.0) + T(0.25) * c10::pi<T>;

        # 返回表达式的值，使用各项计算得到最终结果
        return T(5.64189583547756286948e-01) / std::sqrt(std::sqrt(-x)) * (std::sin(t) * (T(1.0) + z * z * afn / afd) - std::cos(t) * (z * agn / agd));
    }

    # 如果 x 大于等于 2.09，进入该条件分支
    if (x >= T(2.09)) {
        # 设置 domain_flag 标志为 5
        domain_flag = 5;

        # 计算 zeta 值，zeta = 2 * x * sqrt(x) / 3
        T zeta = T(2.0) * x * std::sqrt(x) / T(3.0);

        # 初始化 an 为 0
        T an = 0.0;

        # 计算 AN 多项式的值
        for (uint8_t index = 0; index <= 7; index++) {
            an = an * (T(1.0) / zeta) + AN[index];
        }

        # 初始化 ad 为 0
        T ad = 0.0;

        # 计算 AD 多项式的值
        for (uint8_t index = 0; index <= 7; index++) {
            ad = ad * (T(1.0) / zeta) + AD[index];
        }

        # 计算 ai 值，ai = 0.564189583547756286948 * (an / ad) / (2 * sqrt(sqrt(x)) * exp(zeta))
        ai = T(5.64189583547756286948e-01) * (an / ad) / (T(2.0) * std::sqrt(std::sqrt(x)) * std::exp(zeta));

        # 如果 x 大于 8.3203353，直接返回 ai
        if (x > T(8.3203353)) {
            return ai;
        }
    }

    # 初始化 f 为 1，g 为 x，k 为 1
    f = 1.0;
    g = x;
    k = 1.0;

    # 初始化 m 为 1，n 为 x，t 为 1，z 为 x 的三次方
    T m = 1.0;
    T n = x;
    T t = 1.0;
    T z = x * x * x;

    # 进入迭代循环，直到 t 小于 std::numeric_limits<T>::epsilon()
    while (t > std::numeric_limits<T>::epsilon()) {
        m *= z;
        k += T(1.0);
        m /= k;
        n *= z;
        k += T(1.0);
        n /= k;
        m /= k;
        f += m;
        k += T(1.0);
        n /= k;
        g += n;

        # 更新 t 值为 m/f 的绝对值
        t = std::abs(m / f);
    }

    # 如果 domain_flag 的最低位是 0，返回特定表达式的值
    if ((domain_flag & 1) == 0) {
        return T(0.355028053887817239260) * f - T(0.258819403792806798405) * g;
    }

    # 否则返回 ai 的值
    return ai;
    // 定义并初始化用于计算第一区间内贝塞尔函数 J0(x) 的系数数组 PP
    static const T PP[] = {
            +7.96936729297347051624e-04,
            +8.28352392107440799803e-02,
            +1.23953371646414299388e+00,
            +5.44725003058768775090e+00,
            +8.74716500199817011941e+00,
            +5.30324038235394892183e+00,
            +9.99999999999999997821e-01,
    };

    // 定义并初始化用于计算第一区间内贝塞尔函数 J0(x) 的系数数组 PQ
    static const T PQ[] = {
            +9.24408810558863637013e-04,
            +8.56288474354474431428e-02,
            +1.25352743901058953537e+00,
            +5.47097740330417105182e+00,
            +8.76190883237069594232e+00,
            +5.30605288235394617618e+00,
            +1.00000000000000000218e+00,
    };

    // 定义并初始化用于计算第二区间内贝塞尔函数 J0(x) 的系数数组 QP
    static const T QP[] = {
            -1.13663838898469149931e-02,
            -1.28252718670509318512e+00,
            -1.95539544257735972385e+01,
            -9.32060152123768231369e+01,
            -1.77681167980488050595e+02,
            -1.47077505154951170175e+02,
            -5.14105326766599330220e+01,
            -6.05014350600728481186e+00,
    };

    // 定义并初始化用于计算第二区间内贝塞尔函数 J0(x) 的系数数组 QQ
    static const T QQ[] = {
            +6.43178256118178023184e+01,
            +8.56430025976980587198e+02,
            +3.88240183605401609683e+03,
            +7.24046774195652478189e+03,
            +5.93072701187316984827e+03,
            +2.06209331660327847417e+03,
            +2.42005740240291393179e+02,
    };

    // 定义并初始化用于计算 J0(x) 的高阶系数数组 RP
    static const T RP[] = {
            -4.79443220978201773821e+09,
            +1.95617491946556577543e+12,
            -2.49248344360967716204e+14,
            +9.70862251047306323952e+15,
    };

    // 定义并初始化用于计算 J0(x) 的高阶系数数组 RQ
    static const T RQ[] = {
            +4.99563147152651017219e+02,
            +1.73785401676374683123e+05,
            +4.84409658339962045305e+07,
            +1.11855537045356834862e+10,
            +2.11277520115489217587e+12,
            +3.10518229857422583814e+14,
            +3.18121955943204943306e+16,
            +1.71086294081043136091e+18,
    };

    // 如果 x 小于 0，则取其相反数，以确保 x 为非负值
    if (x < T(0)) {
        x = -x;
    }

    // 当 x <= 5.0 时，使用低阶近似计算 J0(x)
    if (x <= T(5.0)) {
        // 如果 x 非常接近 0，返回 J0(x) 的近似值
        if (x < T(0.00001)) {
            return T(1.0) - x * x / T(4.0);
        }

        T rp = 0.0;

        // 计算低阶近似的分子部分 rp
        for (uint8_t index = 0; index <= 3; index++) {
            rp = rp * (x * x) + RP[index];
        }

        T rq = 0.0;

        // 计算低阶近似的分母部分 rq
        for (uint8_t index = 0; index <= 7; index++) {
            rq = rq * (x * x) + RQ[index];
        }

        // 计算 J0(x) 的近似值并返回
        return (x * x - T(5.78318596294678452118e+00)) * (x * x - T(3.04712623436620863991e+01)) * rp / rq;
    }

    // 当 x > 5.0 时，使用高阶近似计算 J0(x)
    T pp = 0.0;

    // 计算高阶近似的分子部分 pp
    for (uint8_t index = 0; index <= 6; index++) {
        pp = pp * (T(25.0) / (x * x)) + PP[index];
    }

    T pq = 0.0;

    // 计算高阶近似的分母部分 pq
    for (uint8_t index = 0; index <= 6; index++) {
        pq = pq * (T(25.0) / (x * x)) + PQ[index];
    }

    T qp = 0.0;

    // 计算高阶近似的分子部分 qp
    for (uint8_t index = 0; index <= 7; index++) {
        qp = qp * (T(25.0) / (x * x)) + QP[index];
    }

    T qq = 0.0;
    // 循环计算多项式的值
    for (uint8_t index = 0; index <= 6; index++) {
        qq = qq * (T(25.0) / (x * x)) + QQ[index];
    }

    // 返回函数的最终计算结果，组合了多项式的值、三角函数和常数的运算
    return (pp / pq * std::cos(x - T(0.785398163397448309615660845819875721))
            - T(5.0) / x * (qp / qq) * std::sin(x - T(0.785398163397448309615660845819875721)))
           * T(0.797884560802865355879892119868763737) / std::sqrt(x);
} // 结束函数 bessel_j1_forward

template<typename T>
// 计算第一类贝塞尔函数 J1(x) 的近似值
inline C10_HOST_DEVICE T bessel_j1_forward(T x) {
    // 定义第一组系数 PP
    static const T PP[] = {
            +7.62125616208173112003e-04,
            +7.31397056940917570436e-02,
            +1.12719608129684925192e+00,
            +5.11207951146807644818e+00,
            +8.42404590141772420927e+00,
            +5.21451598682361504063e+00,
            +1.00000000000000000254e+00,
    };

    // 定义第二组系数 PQ
    static const T PQ[] = {
            +5.71323128072548699714e-04,
            +6.88455908754495404082e-02,
            +1.10514232634061696926e+00,
            +5.07386386128601488557e+00,
            +8.39985554327604159757e+00,
            +5.20982848682361821619e+00,
            +9.99999999999999997461e-01,
    };

    // 定义第三组系数 QP
    static const T QP[] = {
            +5.10862594750176621635e-02,
            +4.98213872951233449420e+00,
            +7.58238284132545283818e+01,
            +3.66779609360150777800e+02,
            +7.10856304998926107277e+02,
            +5.97489612400613639965e+02,
            +2.11688757100572135698e+02,
            +2.52070205858023719784e+01,
    };

    // 定义第四组系数 QQ
    static const T QQ[] = {
            +7.42373277035675149943e+01,
            +1.05644886038262816351e+03,
            +4.98641058337653607651e+03,
            +9.56231892404756170795e+03,
            +7.99704160447350683650e+03,
            +2.82619278517639096600e+03,
            +3.36093607810698293419e+02,
    };

    // 定义第五组系数 RP
    static const T RP[] = {
            -8.99971225705559398224e+08,
            +4.52228297998194034323e+11,
            -7.27494245221818276015e+13,
            +3.68295732863852883286e+15,
    };

    // 定义第六组系数 RQ
    static const T RQ[] = {
            +6.20836478118054335476e+02,
            +2.56987256757748830383e+05,
            +8.35146791431949253037e+07,
            +2.21511595479792499675e+10,
            +4.74914122079991414898e+12,
            +7.84369607876235854894e+14,
            +8.95222336184627338078e+16,
            +5.32278620332680085395e+18,
    };

    // 如果 x 小于 0，返回负数 J1(-x)
    if (x < T(0.0)) {
        return -bessel_j1_forward(-x);
    }

    // 如果 x 小于等于 5，计算 RP/RQ 的近似值
    if (x <= T(5.0)) {
        T rp = 0.0;
        // 计算 RP 的多项式
        for (uint8_t index = 0; index <= 3; index++) {
            rp = rp * (x * x) + RP[index];
        }

        T rq = 0.0;
        // 计算 RQ 的多项式
        for (uint8_t index = 0; index <= 7; index++) {
            rq = rq * (x * x) + RQ[index];
        }

        // 返回计算结果
        return rp / rq * x * (x * x - T(1.46819706421238932572e+01)) * (x * x - T(4.92184563216946036703e+01));
    }

    // 如果 x 大于 5，计算 PP/PQ/QP/QQ 的近似值
    T pp = 0.0;
    // 计算 PP 的多项式
    for (uint8_t index = 0; index <= 6; index++) {
        pp = pp * (T(5.0) / x * (T(5.0) / x)) + PP[index];
    }

    T pq = 0.0;
    // 计算 PQ 的多项式
    for (uint8_t index = 0; index <= 6; index++) {
        pq = pq * (T(5.0) / x * (T(5.0) / x)) + PQ[index];
    }

    T qp = 0.0;
    // 计算 QP 的多项式
    for (uint8_t index = 0; index <= 7; index++) {
        qp = qp * (T(5.0) / x * (T(5.0) / x)) + QP[index];
    }

    T qq = 0.0;
    // 计算 QQ 的多项式
    for (uint8_t index = 0; index <= 6; index++) {
        qq = qq * (T(5.0) / x * (T(5.0) / x)) + QQ[index];
    }
    }



    // 函数的结尾，结束当前函数的定义或代码块
    return (pp / pq * std::cos(x - T(2.356194490192344928846982537459627163)) - T(5.0) / x * (qp / qq) * std::sin(x - T(2.356194490192344928846982537459627163))) * T(0.797884560802865355879892119868763737) / std::sqrt(x);
    // 定义常量数组 PP，存储 Y0 函数近似多项式的系数
    static const T PP[] = {
            +7.96936729297347051624e-04,
            +8.28352392107440799803e-02,
            +1.23953371646414299388e+00,
            +5.44725003058768775090e+00,
            +8.74716500199817011941e+00,
            +5.30324038235394892183e+00,
            +9.99999999999999997821e-01,
    };

    // 定义常量数组 PQ，存储 Y0 函数近似多项式的系数
    static const T PQ[] = {
            +9.24408810558863637013e-04,
            +8.56288474354474431428e-02,
            +1.25352743901058953537e+00,
            +5.47097740330417105182e+00,
            +8.76190883237069594232e+00,
            +5.30605288235394617618e+00,
            +1.00000000000000000218e+00,
    };

    // 定义常量数组 QP，存储 Y0 函数近似多项式的系数
    static const T QP[] = {
            -1.13663838898469149931e-02,
            -1.28252718670509318512e+00,
            -1.95539544257735972385e+01,
            -9.32060152123768231369e+01,
            -1.77681167980488050595e+02,
            -1.47077505154951170175e+02,
            -5.14105326766599330220e+01,
            -6.05014350600728481186e+00,
    };

    // 定义常量数组 QQ，存储 Y0 函数近似多项式的系数
    static const T QQ[] = {
            +6.43178256118178023184e+01,
            +8.56430025976980587198e+02,
            +3.88240183605401609683e+03,
            +7.24046774195652478189e+03,
            +5.93072701187316984827e+03,
            +2.06209331660327847417e+03,
            +2.42005740240291393179e+02,
    };

    // 定义常量数组 YP，存储 Y0 函数在较大 x 值下的近似多项式的系数
    static const T YP[] = {
            +1.55924367855235737965e+04,
            -1.46639295903971606143e+07,
            +5.43526477051876500413e+09,
            -9.82136065717911466409e+11,
            +8.75906394395366999549e+13,
            -3.46628303384729719441e+15,
            +4.42733268572569800351e+16,
            -1.84950800436986690637e+16,
    };

    // 定义常量数组 YQ，存储 Y0 函数在较大 x 值下的近似多项式的系数
    static const T YQ[] = {
            +1.04128353664259848412e+03,
            +6.26107330137134956842e+05,
            +2.68919633393814121987e+08,
            +8.64002487103935000337e+10,
            +2.02979612750105546709e+13,
            +3.17157752842975028269e+15,
            +2.50596256172653059228e+17,
    };

    // 如果 x 小于等于 5.0
    if (x <= T(5.0)) {
        // 如果 x 等于 0.0，返回负无穷
        if (x == T(0.0)) {
            return -std::numeric_limits<T>::infinity();
        }

        // 如果 x 小于 0.0，返回 NaN
        if (x < T(0.0)) {
            return std::numeric_limits<T>::quiet_NaN();
        }

        // 初始化 yp 为 0.0
        T yp = 0.0;

        // 计算 YP 的近似多项式值
        for (uint8_t index = 0; index <= 7; index++) {
            yp = yp * (x * x) + YP[index];
        }

        // 初始化 yq 为 0.0
        T yq = 0.0;

        // 计算 YQ 的近似多项式值
        for (uint8_t index = 0; index <= 6; index++) {
            yq = yq * (x * x) + YQ[index];
        }

        // 返回 Y0 函数的近似计算结果
        return yp / yq + (T(0.636619772367581343075535053490057448) * std::log(x) * bessel_j0_forward(x));
    }

    // 如果 x 大于 5.0
    T pp = 0.0;

    // 计算 PP 的近似多项式值
    for (uint8_t index = 0; index <= 6; index++) {
        pp = pp * (T(25.0) / (x * x)) + PP[index];
    }

    // 初始化 pq 为 0.0
    T pq = 0.0;

    // 计算 PQ 的近似多项式值
    for (uint8_t index = 0; index <= 6; index++) {
        pq = pq * (T(25.0) / (x * x)) + PQ[index];
    }

    // 初始化 qp 为 0.0
    T qp = 0.0;
    # 循环计算第一个多项式 qp
    for (uint8_t index = 0; index <= 7; index++) {
        qp = qp * (T(25.0) / (x * x)) + QP[index];
    }

    # 初始化第二个多项式 qq
    T qq = 0.0;

    # 循环计算第二个多项式 qq
    for (uint8_t index = 0; index <= 6; index++) {
        qq = qq * (T(25.0) / (x * x)) + QQ[index];
    }

    # 计算并返回最终表达式的值
    return (pp / pq * std::sin(x - T(0.785398163397448309615660845819875721))
            + T(5.0) / x * (qp / qq) * std::cos(x - T(0.785398163397448309615660845819875721)))
           * T(0.797884560802865355879892119868763737) / std::sqrt(x);
    // 定义系数数组 PP，用于计算低于 5.0 的情况下的贝塞尔函数 Y1
    static const T PP[] = {
            +7.62125616208173112003e-04,
            +7.31397056940917570436e-02,
            +1.12719608129684925192e+00,
            +5.11207951146807644818e+00,
            +8.42404590141772420927e+00,
            +5.21451598682361504063e+00,
            +1.00000000000000000254e+00,
    };

    // 定义系数数组 PQ，用于计算低于 5.0 的情况下的贝塞尔函数 Y1
    static const T PQ[] = {
            +5.71323128072548699714e-04,
            +6.88455908754495404082e-02,
            +1.10514232634061696926e+00,
            +5.07386386128601488557e+00,
            +8.39985554327604159757e+00,
            +5.20982848682361821619e+00,
            +9.99999999999999997461e-01,
    };

    // 定义系数数组 QP，用于计算低于 5.0 的情况下的贝塞尔函数 Y1
    static const T QP[] = {
            +5.10862594750176621635e-02,
            +4.98213872951233449420e+00,
            +7.58238284132545283818e+01,
            +3.66779609360150777800e+02,
            +7.10856304998926107277e+02,
            +5.97489612400613639965e+02,
            +2.11688757100572135698e+02,
            +2.52070205858023719784e+01,
    };

    // 定义系数数组 QQ，用于计算低于 5.0 的情况下的贝塞尔函数 Y1
    static const T QQ[] = {
            +7.42373277035675149943e+01,
            +1.05644886038262816351e+03,
            +4.98641058337653607651e+03,
            +9.56231892404756170795e+03,
            +7.99704160447350683650e+03,
            +2.82619278517639096600e+03,
            +3.36093607810698293419e+02,
    };

    // 定义系数数组 YP，用于计算高于 5.0 的情况下的贝塞尔函数 Y1
    static const T YP[] = {
            +1.26320474790178026440e+09,
            -6.47355876379160291031e+11,
            +1.14509511541823727583e+14,
            -8.12770255501325109621e+15,
            +2.02439475713594898196e+17,
            -7.78877196265950026825e+17,
    };

    // 定义系数数组 YQ，用于计算高于 5.0 的情况下的贝塞尔函数 Y1
    static const T YQ[] = {
            +5.94301592346128195359e+02,
            +2.35564092943068577943e+05,
            +7.34811944459721705660e+07,
            +1.87601316108706159478e+10,
            +3.88231277496238566008e+12,
            +6.20557727146953693363e+14,
            +6.87141087355300489866e+16,
            +3.97270608116560655612e+18,
    };

    // 如果 x 小于或等于 5.0，执行以下逻辑
    if (x <= T(5.0)) {
        // 如果 x 等于 0，返回负无穷大
        if (x == T(0.0)) {
            return -std::numeric_limits<T>::infinity();
        }

        // 如果 x 小于等于 0，返回 NaN
        if (x <= T(0.0)) {
            return std::numeric_limits<T>::quiet_NaN();
        }

        // 初始化 yp 为 0.0，计算 YP 数组的贝塞尔函数 Y1 近似值
        T yp = 0.0;
        for (uint8_t index = 0; index <= 5; index++) {
            yp = yp * (x * x) + YP[index];
        }

        // 初始化 yq 为 0.0，计算 YQ 数组的贝塞尔函数 Y1 近似值
        T yq = 0.0;
        for (uint8_t index = 0; index <= 7; index++) {
            yq = yq * (x * x) + YQ[index];
        }

        // 返回 Y1(x) 的近似值
        return x * (yp / yq) + (T(0.636619772367581343075535053490057448) * (bessel_j1_forward(x) * std::log(x) - T(1.0) / x));
    }

    // 如果 x 大于 5.0，执行以下逻辑
    T pp = 0.0;
    // 计算 PP 数组的贝塞尔函数 Y1 近似值
    for (uint8_t index = 0; index <= 6; index++) {
        pp = pp * (T(5.0) / x * (T(5.0) / x)) + PP[index];
    }

    // 计算 PQ 数组的贝塞尔函数 Y1 近似值
    T pq = 0.0;
    for (uint8_t index = 0; index <= 6; index++) {
        pq = pq * (T(5.0) / x * (T(5.0) / x)) + PQ[index];
    }

    // 初始化 qp 为 0.0，计算 QP 数组的贝塞尔函数 Y1 近似值
    T qp = 0.0;
    # 循环计算 QP 值，初始为 0
    for (uint8_t index = 0; index <= 7; index++) {
        # 使用 Horner 方法计算 QP 多项式
        qp = qp * (T(5.0) / x * (T(5.0) / x)) + QP[index];
    }

    # 初始化 qq 值为 0.0
    T qq = 0.0;

    # 循环计算 QQ 值，初始为 0
    for (uint8_t index = 0; index <= 6; index++) {
        # 使用 Horner 方法计算 QQ 多项式
        qq = qq * (T(5.0) / x * (T(5.0) / x)) + QQ[index];
    }

    # 计算最终表达式的值并返回
    return (pp / pq * std::sin(x - T(2.356194490192344928846982537459627163)) +
            T(5.0) / x * (qp / qq) * std::cos(x - T(2.356194490192344928846982537459627163))) *
           T(0.797884560802865355879892119868763737) / std::sqrt(x);
// 计算第一类 Chebyshev 多项式的值 T_n(x)，其中 n 是整数
template<typename T>
inline C10_HOST_DEVICE T chebyshev_polynomial_t_forward(T x, int64_t n) {
    // 如果 n 小于 0，返回 0.0
    if (n < 0) {
        return T(0.0);
    }

    // 如果 x 的绝对值等于 1.0
    if (std::abs(x) == T(1.0)) {
        // 如果 x 大于 0.0 或者 n 是偶数，则返回 1.0
        if (x > T(0.0) || n % 2 == 0) {
            return T(1.0);
        }
        // 否则返回 -1.0
        return T(-1.0);
    }

    // 如果 n 大于 6 并且 x 的绝对值小于 1.0
    if ((n > 6) && (std::abs(x) < T(1.0))) {
        // 返回 cos(n * acos(x))
        return std::cos(n * std::acos(x));
    }

    // 如果 n 等于 0，返回 1.0
    if (n == 0) {
        return T(1.0);
    }

    // 如果 n 等于 1，返回 x
    if (n == 1) {
        return x;
    }

    T p = T(1.0);
    T q = x;
    T r;

    // 计算 T_n(x) 的递推关系
    for (int64_t k = 2; k <= n; k++) {
        r = (x + x) * q - p;
        p = q;
        q = r;
    }

    return r; // 返回 T_n(x)
} // chebyshev_polynomial_t_forward(T x, int64_t n)

// 转换函数，将浮点数 n 转换为整数并调用上面的函数计算 T_n(x)
template<typename T, bool is_cuda=false>
inline C10_HOST_DEVICE T chebyshev_polynomial_t_forward(T x, T n) {
    return chebyshev_polynomial_t_forward(x, static_cast<int64_t>(n));
} // chebyshev_polynomial_t_forward(T x, T n)

// 计算第二类 Chebyshev 多项式的值 U_n(x)，其中 n 是整数
template<typename T>
inline C10_HOST_DEVICE T chebyshev_polynomial_u_forward(T x, int64_t n) {
    // 如果 n 小于 0，返回 0.0
    if (n < 0) {
        return T(0.0);
    }

    // 如果 x 的绝对值等于 1.0
    if (std::abs(x) == T(1.0)) {
        // 如果 x 大于 0.0 或者 n 是偶数，则返回 n + 1
        if (x > T(0.0) || n % 2 == 0) {
            return n + 1;
        }
        // 否则返回 -(n + 1)
        return -(n + 1);
    }

    // 如果 n 大于 8 并且 x 的绝对值小于 1.0
    if ((n > 8) && (std::abs(x) < T(1.0))) {
        // 如果 sin(acos(x)) 不等于 0.0
        if (std::sin(std::acos(x)) != T(0.0)) {
            // 返回 sin((n + 1) * acos(x)) / sin(acos(x))
            return std::sin((n + 1) * std::acos(x)) / std::sin(std::acos(x));
        }
        // 否则返回 (n + 1) * cos((n + 1) * acos(x)) / x
        return (n + 1) * std::cos((n + 1) * std::acos(x)) / x;
    }

    // 如果 n 等于 0，返回 1.0
    if (n == 0) {
        return T(1.0);
    }

    // 如果 n 等于 1，返回 x + x
    if (n == 1) {
        return x + x;
    }

    T p = T(1.0);
    T q = x + x;
    T r;

    // 计算 U_n(x) 的递推关系
    for (int64_t k = 2; k <= n; k++) {
        r = (x + x) * q - p;
        p = q;
        q = r;
    }

    return r; // 返回 U_n(x)
} // chebyshev_polynomial_u_forward(T x, int64_t n)

// 转换函数，将浮点数 n 转换为整数并调用上面的函数计算 U_n(x)
template<typename T, bool is_cuda=false>
inline C10_HOST_DEVICE T chebyshev_polynomial_u_forward(T x, T n) {
    return chebyshev_polynomial_u_forward(x, static_cast<int64_t>(n));
} // chebyshev_polynomial_u_forward(T x, T n)

// 计算第三类 Chebyshev 多项式的值 V_n(x)，其中 n 是整数
template<typename T>
inline C10_HOST_DEVICE T chebyshev_polynomial_v_forward(T x, int64_t n) {
    // 如果 n 小于 0，返回 0.0
    if (n < 0) {
        return T(0.0);
    }

    // 如果 x 的绝对值等于 1.0
    if (std::abs(x) == T(1.0)) {
        // 如果 x 大于 0.0，则返回 1.0
        if (x > T(0.0)) {
            return T(1.0);
        }
        // 如果 n 是偶数，则返回 n + n + 1
        if (n % 2 == 0) {
            return n + n + 1;
        }
        // 否则返回 -(n + n + 1)
        return -(n + n + 1);
    }

    // 如果 n 大于 8 并且 x 的绝对值小于 1.0
    if ((n > 8) && (std::abs(x) < T(1.0))) {
        // 如果 sin(acos(x) / 2.0) 不等于 1.0
        if (std::sin(std::acos(x) / T(2.0)) != T(1.0)) {
            // 返回 cos((n + 0.5) * acos(x)) / cos(acos(x) / 2.0)
            return std::cos((n + T(0.5)) * std::acos(x)) / std::cos(std::acos(x) / T(2.0));
        }
        // 如果 n 是偶数，则返回 n + n + 1
        if (n % 2 == 0) {
            return n + n + 1;
        }
        // 否则返回 -(n + n + 1)
        return -(n + n + 1);
    }

    // 如果 n 等于 0，返回 1.0
    if (n == 0) {
        return T(1.0);
    }

    // 如果 n 等于 1，返回 x + x - 1.0
    if (n == 1) {
        return x + x - T(1.0);
    }

    T p = T(1.0);
    T q = x + x - T(1.0);
    T r;

    // 计算 V_n(x) 的递推关系
    for (int64_t k = 2; k <= n; k++) {
        r = (x + x) * q - p;
        p = q;
        q = r;
    }

    return r; // 返回 V_n(x)
} // chebyshev_polynomial_v_forward(T x, int64_t n)
// 定义一个模板函数 chebyshev_polynomial_v_forward，返回类型为 T，接受两个参数 x 和 n
template<typename T, bool is_cuda=false>
inline C10_HOST_DEVICE T chebyshev_polynomial_v_forward(T x, T n) {
    // 调用重载函数 chebyshev_polynomial_v_forward(x, static_cast<int64_t>(n)) 并返回结果
    return chebyshev_polynomial_v_forward(x, static_cast<int64_t>(n));
} // chebyshev_polynomial_v_forward(T x, T n)

// 定义一个模板函数 chebyshev_polynomial_w_forward，返回类型为 T，接受两个参数 x 和 n
template<typename T>
inline C10_HOST_DEVICE T chebyshev_polynomial_w_forward(T x, int64_t n) {
    // 如果 n 小于 0，返回 T(0.0)
    if (n < 0) {
        return T(0.0);
    }

    // 如果 x 的绝对值为 1.0
    if (std::abs(x) == T(1.0)) {
        // 如果 x 大于 0.0，返回 n + n + 1
        if (x > T(0.0)) {
            return n + n + 1;
        }
        // 如果 n 是偶数，返回 1.0；否则返回 -1.0
        if (n % 2 == 0) {
            return T(1.0);
        }
        return T(-1.0);
    }

    // 如果 n 大于 8 并且 x 的绝对值小于 1.0
    if ((n > 8) && (std::abs(x) < T(1.0))) {
        // 如果 cos(acos(x) / 2.0) 不等于 1.0，返回 sin((n + 0.5) * acos(x)) / sin(acos(x) / 2.0)
        if (std::cos(std::acos(x) / T(2.0)) != T(1.0)) {
            return std::sin((n + T(0.5)) * std::acos(x)) / std::sin(std::acos(x) / T(2.0));
        }
        // 如果 x 大于 0.0，返回 n + n + 1
        if (x > T(0.0)) {
            return n + n + 1;
        }
        // 如果 n 是偶数，返回 1.0；否则返回 -1.0
        if (n % 2 == 0) {
            return T(1.0);
        }
        return T(-1.0);
    }

    // 如果 n 等于 0，返回 1.0
    if (n == 0) {
        return T(1.0);
    }

    // 如果 n 等于 1，返回 x + x + 1.0
    if (n == 1) {
        return x + x + T(1.0);
    }

    // 初始化变量 p 和 q
    T p = T(1.0);
    T q = x + x + T(1.0);
    T r;

    // 循环计算 n 次 Chebyshev 多项式的值
    for (int64_t k = 2; k <= n; k++) {
        r = (x + x) * q - p;
        p = q;
        q = r;
    }

    // 返回计算结果 r
    return r;
} // chebyshev_polynomial_w_forward(T x, int64_t n)

// 定义一个模板函数 chebyshev_polynomial_w_forward，返回类型为 T，接受两个参数 x 和 n
template<typename T, bool is_cuda=false>
inline C10_HOST_DEVICE T chebyshev_polynomial_w_forward(T x, T n) {
    // 调用重载函数 chebyshev_polynomial_w_forward(x, static_cast<int64_t>(n)) 并返回结果
    return chebyshev_polynomial_w_forward(x, static_cast<int64_t>(n));
} // chebyshev_polynomial_w_forward(T x, T n)

// 定义一个模板函数 hermite_polynomial_h_forward，返回类型为 T，接受两个参数 x 和 n
template<typename T>
inline C10_HOST_DEVICE T hermite_polynomial_h_forward(T x, int64_t n) {
    // 如果 n 小于 0，返回 0.0
    if (n < 0) {
        return T(0.0);
    }

    // 如果 n 等于 0，返回 1.0
    if (n == 0) {
        return T(1.0);
    }

    // 如果 n 等于 1，返回 x + x
    if (n == 1) {
        return x + x;
    }

    // 初始化变量 p, q, r
    T p = T(1.0);
    T q = x + x;
    T r = T(0.0);

    // 循环计算 Hermite 多项式的值
    for (int64_t k = 2; k < n + n; k += 2) {
        r = (x + x) * q - k * p;
        p = q;
        q = r;
    }

    // 返回计算结果 r
    return r;
} // hermite_polynomial_h_forward(T x, int64_t n)

// 定义一个模板函数 hermite_polynomial_h_forward，返回类型为 T，接受两个参数 x 和 n
template<typename T, bool is_cuda=false, std::enable_if_t<!std::is_floating_point<T>::value, int> = 0>
inline C10_HOST_DEVICE T hermite_polynomial_h_forward(T x, T n) {
    // 调用重载函数 hermite_polynomial_h_forward(x, static_cast<int64_t>(n)) 并返回结果
    return hermite_polynomial_h_forward(x, static_cast<int64_t>(n));
} // hermite_polynomial_h_forward(T x, T n)

// 定义一个模板函数 hermite_polynomial_h_forward，返回类型为 T，接受两个参数 x 和 n
template<typename T, bool is_cuda=false, std::enable_if_t<std::is_floating_point<T>::value, int> = 0>
inline C10_HOST_DEVICE T hermite_polynomial_h_forward(T x, T n) {
    // 如果 n 不是无穷大且不是 NaN，调用 hermite_polynomial_h_forward(x, static_cast<int64_t>(n)) 并返回结果
    return hermite_polynomial_h_forward(x, ((!std::isinf(n)) && (!std::isnan(n))) ? static_cast<int64_t>(n) : static_cast<int64_t>(-1));
} // hermite_polynomial_h_forward(T x, T n)

// 定义一个模板函数 hermite_polynomial_he_forward，返回类型为 T，接受两个参数 x 和 n
template<typename T>
inline C10_HOST_DEVICE T hermite_polynomial_he_forward(T x, int64_t n) {
    // 如果 n 小于 0，返回 0.0
    if (n < 0) {
        return T(0.0);
    }

    // 如果 n 等于 0，返回 1.0
    if (n == 0) {
        return T(1.0);
    }

    // 如果 n 等于 1，返回 x
    if (n == 1) {
        return x;
    }

    // 初始化变量 p, q, r
    T p = T(1.0);
    T q = x;
    T r;

    // 循环计算 Hermite 多项式的值
    for (int64_t k = 1; k < n; k++) {
        r = x * q - k * p;
        p = q;
        q = r;
    }

    // 返回计算结果 r
    return r;
} // hermite_polynomial_he_forward(T x, int64_t n)
// hermite_polynomial_he_forward 函数模板的实现，将传入的参数 n 转换为 int64_t 后再调用对应的函数
} // hermite_polynomial_he_forward(T x, int64_t n)

// 模板函数，用于处理泛型 T 类型的参数 x 和 n，调用 hermite_polynomial_he_forward 函数
template<typename T, bool is_cuda=false>
inline C10_HOST_DEVICE T hermite_polynomial_he_forward(T x, T n) {
    return hermite_polynomial_he_forward(x, static_cast<int64_t>(n));
} // hermite_polynomial_he_forward(T x, T n)

// laguerre_polynomial_l_forward 函数模板的实现，计算拉盖尔多项式的正向递归计算
template<typename T>
inline C10_HOST_DEVICE T laguerre_polynomial_l_forward(T x, int64_t n) {
    // 处理 n 小于 0 的情况，返回 0
    if (n < 0) {
        return T(0.0);
    }

    // 处理 x 的绝对值为 0 的情况，返回 1
    if (std::abs(x) == T(0.0)) {
        return T(1.0);
    }

    // 处理 n 等于 0 的情况，返回 1
    if (n == 0) {
        return T(1.0);
    }

    // 处理 n 等于 1 的情况，返回 1 - x
    if (n == 1) {
        return T(1.0) - x;
    }

    T p = T(1.0);
    T q = T(1.0) - x;
    T r;

    // 使用循环计算拉盖尔多项式的值，直到达到 n
    for (int64_t k = 1; k < n; k++) {
        r = (((k + k) + (T(1.0) - x)) * q - k * p) / (k + 1);
        p = q;
        q = r;
    }

    return r;
} // laguerre_polynomial_l_forward(T x, int64_t n)

// 模板函数，用于处理泛型 T 类型的参数 x 和 n，调用 laguerre_polynomial_l_forward 函数
template<typename T, bool is_cuda=false>
inline C10_HOST_DEVICE T laguerre_polynomial_l_forward(T x, T n) {
    return laguerre_polynomial_l_forward(x, static_cast<int64_t>(n));
} // laguerre_polynomial_l_forward(T x, T n)

// legendre_polynomial_p_forward 函数模板的实现，计算勒让德多项式的正向递归计算
template<typename T>
inline C10_HOST_DEVICE T legendre_polynomial_p_forward(T x, int64_t n) {
    // 处理 n 小于 0 的情况，返回 0
    if (n < 0) {
        return T(0.0);
    }

    // 处理 x 的绝对值为 1 的情况
    if (std::abs(x) == T(1.0)) {
        // 如果 x 大于 0 或者 n 是偶数，则返回 1
        if (x > T(0.0) || n % 2 == 0) {
            return T(1.0);
        }
        // 否则返回 -1
        return T(-1.0);
    }

    // 处理 n 等于 0 的情况，返回 1
    if (n == 0) {
        return T(1.0);
    }

    // 处理 n 等于 1 的情况，返回 x
    if (n == 1) {
        return x;
    }

    T p = T(1.0);
    T q = x;
    T r;

    // 使用循环计算勒让德多项式的值，直到达到 n
    for (int64_t k = 1; k < n; k++) {
        r = ((k + k + 1) * x * q - k * p) / (k + 1);
        p = q;
        q = r;
    }

    return r;
} // legendre_polynomial_p_forward(T x, int64_t n)

// 模板函数，用于处理泛型 T 类型的参数 x 和 n，调用 legendre_polynomial_p_forward 函数
template<typename T, bool is_cuda=false>
inline C10_HOST_DEVICE T legendre_polynomial_p_forward(T x, T n) {
    return legendre_polynomial_p_forward(x, static_cast<int64_t>(n));
} // legendre_polynomial_p_forward(T x, T n)

// modified_bessel_i0_forward 函数模板的声明，用于计算修正贝塞尔函数 I0 的正向计算
template<typename T>
inline C10_HOST_DEVICE T modified_bessel_i0_forward(T x) {
    // A 数组存储了 Chebyshev 多项式的系数，用于近似函数计算
    static const T A[] = {
            -4.41534164647933937950e-18,
            +3.33079451882223809783e-17,
            -2.43127984654795469359e-16,
            +1.71539128555513303061e-15,
            -1.16853328779934516808e-14,
            +7.67618549860493561688e-14,
            -4.85644678311192946090e-13,
            +2.95505266312963983461e-12,
            -1.72682629144155570723e-11,
            +9.67580903537323691224e-11,
            -5.18979560163526290666e-10,
            +2.65982372468238665035e-09,
            -1.30002500998624804212e-08,
            +6.04699502254191894932e-08,
            -2.67079385394061173391e-07,
            +1.11738753912010371815e-06,
            -4.41673835845875056359e-06,
            +1.64484480707288970893e-05,
            -5.75419501008210370398e-05,
            +1.88502885095841655729e-04,
            -5.76375574538582365885e-04,
            +1.63947561694133579842e-03,
            -4.32430999505057594430e-03,
            +1.05464603945949983183e-02,
            -2.37374148058994688156e-02,
            +4.93052842396707084878e-02,
            -9.49010970480476444210e-02,
            +1.71620901522208775349e-01,
            -3.04682672343198398683e-01,
            +6.76795274409476084995e-01,
    };

    // B 数组存储了另一个 Chebyshev 多项式的系数，用于在不同范围内的近似计算
    static const T B[] = {
            -7.23318048787475395456e-18,
            -4.83050448594418207126e-18,
            +4.46562142029675999901e-17,
            +3.46122286769746109310e-17,
            -2.82762398051658348494e-16,
            -3.42548561967721913462e-16,
            +1.77256013305652638360e-15,
            +3.81168066935262242075e-15,
            -9.55484669882830764870e-15,
            -4.15056934728722208663e-14,
            +1.54008621752140982691e-14,
            +3.85277838274214270114e-13,
            +7.18012445138366623367e-13,
            -1.79417853150680611778e-12,
            -1.32158118404477131188e-11,
            -3.14991652796324136454e-11,
            +1.18891471078464383424e-11,
            +4.94060238822496958910e-10,
            +3.39623202570838634515e-09,
            +2.26666899049817806459e-08,
            +2.04891858946906374183e-07,
            +2.89137052083475648297e-06,
            +6.88975834691682398426e-05,
            +3.36911647825569408990e-03,
            +8.04490411014108831608e-01,
    };

    // 初始化 p 和 q，用于迭代计算
    T p;
    T q = 0.0;

    // 根据 x 的绝对值大小选择不同的系数数组进行近似计算
    if (std::abs(x) <= T(8.0)) {
        // 在较小的 x 范围内，使用 A 数组进行近似计算
        T a = A[0];

        // 进行 Chebyshev 多项式的迭代计算，生成近似值
        for (uint8_t index = 1; index < 30; index++) {
            p = q;
            q = a;
            a = ((std::abs(x) / T(2.0)) - T(2.0)) * q - p + A[index];
        }

        // 返回近似值乘以指数函数的结果
        return std::exp(std::abs(x)) * (T(0.5) * (a - p));
    }

    // 在较大的 x 范围内，使用 B 数组进行近似计算
    T b = B[0];

    // 进行另一组 Chebyshev 多项式的迭代计算，生成近似值
    for (uint8_t index = 1; index < 25; index++) {
        p = q;
        q = b;
        b = (T(32.0) / std::abs(x) - T(2.0)) * q - p + B[index];
    }

    // 返回近似值乘以指数函数的结果除以 x 的绝对值的平方根
    return std::exp(std::abs(x)) * (T(0.5) * (b - p)) / std::sqrt(std::abs(x));
// 定义模板函数 modified_bessel_i1_forward，计算修正贝塞尔函数 I_1(x) 的近似值
template<typename T>
inline C10_HOST_DEVICE T modified_bessel_i1_forward(T x) {
    // 定义常量数组 A，存储修正贝塞尔函数 I_1(x) 的系数
    static const T A[] = {
            +2.77791411276104639959e-18,
            -2.11142121435816608115e-17,
            +1.55363195773620046921e-16,
            -1.10559694773538630805e-15,
            +7.60068429473540693410e-15,
            -5.04218550472791168711e-14,
            +3.22379336594557470981e-13,
            -1.98397439776494371520e-12,
            +1.17361862988909016308e-11,
            -6.66348972350202774223e-11,
            +3.62559028155211703701e-10,
            -1.88724975172282928790e-09,
            +9.38153738649577178388e-09,
            -4.44505912879632808065e-08,
            +2.00329475355213526229e-07,
            -8.56872026469545474066e-07,
            +3.47025130813767847674e-06,
            -1.32731636560394358279e-05,
            +4.78156510755005422638e-05,
            -1.61760815825896745588e-04,
            +5.12285956168575772895e-04,
            -1.51357245063125314899e-03,
            +4.15642294431288815669e-03,
            -1.05640848946261981558e-02,
            +2.47264490306265168283e-02,
            -5.29459812080949914269e-02,
            +1.02643658689847095384e-01,
            -1.76416518357834055153e-01,
            +2.52587186443633654823e-01,
    };

    // 定义常量数组 B，存储修正贝塞尔函数 I_1(x) 的系数
    static const T B[] = {
            +7.51729631084210481353e-18,
            +4.41434832307170791151e-18,
            -4.65030536848935832153e-17,
            -3.20952592199342395980e-17,
            +2.96262899764595013876e-16,
            +3.30820231092092828324e-16,
            -1.88035477551078244854e-15,
            -3.81440307243700780478e-15,
            +1.04202769841288027642e-14,
            +4.27244001671195135429e-14,
            -2.10154184277266431302e-14,
            -4.08355111109219731823e-13,
            -7.19855177624590851209e-13,
            +2.03562854414708950722e-12,
            +1.41258074366137813316e-11,
            +3.25260358301548823856e-11,
            -1.89749581235054123450e-11,
            -5.58974346219658380687e-10,
            -3.83538038596423702205e-09,
            -2.63146884688951950684e-08,
            -2.51223623787020892529e-07,
            -3.88256480887769039346e-06,
            -1.10588938762623716291e-04,
            -9.76109749136146840777e-03,
            +7.78576235018280120474e-01,
    };

    // 初始化变量 p 和 q
    T p;
    T q = 0.0;

    // 如果输入参数 x 的绝对值小于等于 8.0，使用 A 数组计算修正贝塞尔函数近似值
    if (std::abs(x) <= T(8.0)) {
        T a = A[0];

        // 循环计算修正贝塞尔函数的递推公式
        for (uint8_t index = 1; index < 29; index++) {
            p = q;
            q = a;
            a = ((std::abs(x) / T(2.0)) - T(2.0)) * q - p + A[index];
        }

        // 根据 x 的符号返回修正贝塞尔函数的近似值
        if (x < T(0.0)) {
            return -(T(0.5) * (a - p) * std::abs(x) * std::exp(std::abs(x)));
        }

        return T(0.5) * (a - p) * std::abs(x) * std::exp(std::abs(x));
    }

    // 如果 x 的绝对值大于 8.0，使用 B 数组计算修正贝塞尔函数的近似值
    T b = B[0];
    // 循环计算巴尔根多项式的值，索引从1到24
    for (uint8_t index = 1; index < 25; index++) {
        // 保存前一个和前两个多项式值
        p = q;
        q = b;
        // 计算下一个巴尔根多项式的值
        b = (T(32.0) / std::abs(x) - T(2.0)) * q - p + B[index];
    }

    // 如果 x 小于零，返回负的巴尔根多项式的导数值
    if (x < T(0.0)) {
        return -(std::exp(std::abs(x)) * (T(0.5) * (b - p)) / std::sqrt(std::abs(x)));
    }

    // 如果 x 大于等于零，返回巴尔根多项式的导数值
    return std::exp(std::abs(x)) * (T(0.5) * (b - p)) / std::sqrt(std::abs(x));
// 计算修正贝塞尔函数 K0(x) 的前向计算
template<typename T>
inline C10_HOST_DEVICE T modified_bessel_k0_forward(T x) {
    // A 数组，用于计算 K0(x) 的系数
    static const T A[] = {
            +1.37446543561352307156e-16,
            +4.25981614279661018399e-14,
            +1.03496952576338420167e-11,
            +1.90451637722020886025e-09,
            +2.53479107902614945675e-07,
            +2.28621210311945178607e-05,
            +1.26461541144692592338e-03,
            +3.59799365153615016266e-02,
            +3.44289899924628486886e-01,
            -5.35327393233902768720e-01,
    };

    // B 数组，用于计算 K0(x) 的系数
    static const T B[] = {
            +5.30043377268626276149e-18,
            -1.64758043015242134646e-17,
            +5.21039150503902756861e-17,
            -1.67823109680541210385e-16,
            +5.51205597852431940784e-16,
            -1.84859337734377901440e-15,
            +6.34007647740507060557e-15,
            -2.22751332699166985548e-14,
            +8.03289077536357521100e-14,
            -2.98009692317273043925e-13,
            +1.14034058820847496303e-12,
            -4.51459788337394416547e-12,
            +1.85594911495471785253e-11,
            -7.95748924447710747776e-11,
            +3.57739728140030116597e-10,
            -1.69753450938905987466e-09,
            +8.57403401741422608519e-09,
            -4.66048989768794782956e-08,
            +2.76681363944501510342e-07,
            -1.83175552271911948767e-06,
            +1.39498137188764993662e-05,
            -1.28495495816278026384e-04,
            +1.56988388573005337491e-03,
            -3.14481013119645005427e-02,
            +2.44030308206595545468e+00,
    };

    // 如果 x 等于 0，则返回正无穷
    if (x == T(0.0)) {
        return std::numeric_limits<T>::infinity();
    }

    // 如果 x 小于 0，则返回 NaN
    if (x < T(0.0)) {
        return std::numeric_limits<T>::quiet_NaN();
    }

    T p;
    T q = 0.0;

    // 当 x <= 2.0 时，使用 A 数组计算 K0(x)
    if (x <= T(2.0)) {
        T a = A[0];

        for (uint8_t index = 1; index < 10; index++) {
            p = q;
            q = a;
            a = (x * x - T(2.0)) * q - p + A[index];
        }

        // 返回 K0(x) 的计算结果
        return T(0.5) * (a - p) - std::log(0.5 * x) * modified_bessel_i0_forward(x);
    }

    // 当 x > 2.0 时，使用 B 数组计算 K0(x)
    T b = B[0];

    for (uint8_t index = 1; index < 25; index++) {
        p = q;
        q = b;
        b = (T(8.0) / x - T(2.0)) * q - p + B[index];
    }

    // 返回 K0(x) 的计算结果
    return std::exp(-x) * (T(0.5) * (b - p)) / std::sqrt(x);
} // modified_bessel_k0_forward(T x)

// 计算修正贝塞尔函数 K1(x) 的前向计算
template<typename T>
inline C10_HOST_DEVICE T modified_bessel_k1_forward(T x) {
    // A 数组，用于计算 K1(x) 的系数
    static const T A[] = {
            -7.02386347938628759343e-18,
            -2.42744985051936593393e-15,
            -6.66690169419932900609e-13,
            -1.41148839263352776110e-10,
            -2.21338763073472585583e-08,
            -2.43340614156596823496e-06,
            -1.73028895751305206302e-04,
            -6.97572385963986435018e-03,
            -1.22611180822657148235e-01,
            -3.53155960776544875667e-01,
            +1.52530022733894777053e+00,
    };
    // 定义静态常量数组 B，存储系数用于计算修正贝塞尔函数 I0(x)
    static const T B[] = {
            -5.75674448366501715755e-18,
            +1.79405087314755922667e-17,
            -5.68946255844285935196e-17,
            +1.83809354436663880070e-16,
            -6.05704724837331885336e-16,
            +2.03870316562433424052e-15,
            -7.01983709041831346144e-15,
            +2.47715442448130437068e-14,
            -8.97670518232499435011e-14,
            +3.34841966607842919884e-13,
            -1.28917396095102890680e-12,
            +5.13963967348173025100e-12,
            -2.12996783842756842877e-11,
            +9.21831518760500529508e-11,
            -4.19035475934189648750e-10,
            +2.01504975519703286596e-09,
            -1.03457624656780970260e-08,
            +5.74108412545004946722e-08,
            -3.50196060308781257119e-07,
            +2.40648494783721712015e-06,
            -1.93619797416608296024e-05,
            +1.95215518471351631108e-04,
            -2.85781685962277938680e-03,
            +1.03923736576817238437e-01,
            +2.72062619048444266945e+00,
    };

    // 如果 x 等于 0.0，返回 T 类型的正无穷大
    if (x == T(0.0)) {
        return std::numeric_limits<T>::infinity();
    }

    // 如果 x 小于 0.0，返回 T 类型的静默 NaN
    if (x < T(0.0)) {
        return std::numeric_limits<T>::quiet_NaN();
    }

    T p;
    T q = 0.0;

    // 如果 x 小于等于 2.0
    if (x <= T(2.0)) {
        T a = A[0];

        // 使用前11个系数计算修正贝塞尔函数 I0(x) 的近似值
        for (uint8_t index = 1; index < 11; index++) {
            p = q;
            q = a;
            a = (x * x - T(2.0)) * q - p + A[index];
        }

        // 返回修正贝塞尔函数 I1(x) 的正向逼近值
        return std::log(T(0.5) * x) * modified_bessel_i1_forward(x) + T(0.5) * (a - p) / x;
    }

    // 如果 x 大于 2.0
    T b = B[0];

    // 使用后24个系数计算修正贝塞尔函数 I0(x) 的近似值
    for (uint8_t index = 1; index < 25; index++) {
        p = q;
        q = b;
        b = (T(8.0) / x - T(2.0)) * q - p + B[index];
    }

    // 返回修正贝塞尔函数 I0(x) 的近似值
    return std::exp(-x) * (T(0.5) * (b - p)) / std::sqrt(x);
// 计算经过缩放的修正贝塞尔函数 K1(x) 的前向传播结果，函数模板定义
template<typename T>
inline C10_HOST_DEVICE T scaled_modified_bessel_k1_forward(T x) {
    // A 数组定义，存储修正贝塞尔函数 K1(x) 前向传播系数
    static const T A[] = {
            -7.02386347938628759343e-18,
            -2.42744985051936593393e-15,
            -6.66690169419932900609e-13,
            -1.41148839263352776110e-10,
            -2.21338763073472585583e-08,
            -2.43340614156596823496e-06,
            -1.73028895751305206302e-04,
            -6.97572385963986435018e-03,
            -1.22611180822657148235e-01,
            -3.53155960776544875667e-01,
            +1.52530022733894777053e+00,
    };

    // 如果输入 x 为零，则返回正无穷
    if (x == T(0.0)) {
        return std::numeric_limits<T>::infinity();
    }

    // 如果输入 x 小于零，则返回静默 NaN
    if (x < T(0.0)) {
        return std::numeric_limits<T>::quiet_NaN();
    }

    T p;
    T q = 0.0;

    // 如果 x <= 2.0，使用 A 数组中的系数进行计算
    if (x <= T(2.0)) {
        T a = A[0];

        // 使用递归关系计算修正贝塞尔函数 K1(x) 的近似值
        for (uint64_t index = 1; index < 10; index++) {
            p = q;
            q = a;
            a = (x * x - T(2.0)) * q - p + A[index];
        }

        // 返回修正贝塞尔函数 K1(x) 的前向传播结果
        return (T(0.5) * (a - p) - std::log(T(0.5) * x) * modified_bessel_i0_forward(x)) * std::exp(x);
    }

    // 如果 x > 2.0，使用 B 数组中的系数进行计算
    static const T B[] = {
            +5.30043377268626276149e-18,
            -1.64758043015242134646e-17,
            +5.21039150503902756861e-17,
            -1.67823109680541210385e-16,
            +5.51205597852431940784e-16,
            -1.84859337734377901440e-15,
            +6.34007647740507060557e-15,
            -2.22751332699166985548e-14,
            +8.03289077536357521100e-14,
            -2.98009692317273043925e-13,
            +1.14034058820847496303e-12,
            -4.51459788337394416547e-12,
            +1.85594911495471785253e-11,
            -7.95748924447710747776e-11,
            +3.57739728140030116597e-10,
            -1.69753450938905987466e-09,
            +8.57403401741422608519e-09,
            -4.66048989768794782956e-08,
            +2.76681363944501510342e-07,
            -1.83175552271911948767e-06,
            +1.39498137188764993662e-05,
            -1.28495495816278026384e-04,
            +1.56988388573005337491e-03,
            -3.14481013119645005427e-02,
            +2.44030308206595545468e+00,
    };

    T b = B[0];

    // 使用递归关系计算修正贝塞尔函数 K1(x) 的近似值
    for (uint64_t index = 1; index < 25; index++) {
        p = q;
        q = b;
        b = (T(8.0) / x - T(2.0)) * q - p + B[index];
    }

    // 返回修正贝塞尔函数 K1(x) 的前向传播结果
    return T(0.5) * (b - p) / std::sqrt(x);
} // T scaled_modified_bessel_k1_forward(T x)
    // 定义静态常量数组 B，存储系数值
    static const T B[] = {
            -5.75674448366501715755e-18,
            +1.79405087314755922667e-17,
            -5.68946255844285935196e-17,
            +1.83809354436663880070e-16,
            -6.05704724837331885336e-16,
            +2.03870316562433424052e-15,
            -7.01983709041831346144e-15,
            +2.47715442448130437068e-14,
            -8.97670518232499435011e-14,
            +3.34841966607842919884e-13,
            -1.28917396095102890680e-12,
            +5.13963967348173025100e-12,
            -2.12996783842756842877e-11,
            +9.21831518760500529508e-11,
            -4.19035475934189648750e-10,
            +2.01504975519703286596e-09,
            -1.03457624656780970260e-08,
            +5.74108412545004946722e-08,
            -3.50196060308781257119e-07,
            +2.40648494783721712015e-06,
            -1.93619797416608296024e-05,
            +1.95215518471351631108e-04,
            -2.85781685962277938680e-03,
            +1.03923736576817238437e-01,
            +2.72062619048444266945e+00,
    };

    // 如果输入参数 x 等于 0，返回正无穷大
    if (x == T(0.0)) {
        return std::numeric_limits<T>::infinity();
    }

    // 如果输入参数 x 小于 0，返回 quiet NaN（非信号 NaN）
    if (x < T(0.0)) {
        return std::numeric_limits<T>::quiet_NaN();
    }

    // 初始化变量 p 和 q 为 0.0
    T p;
    T q = 0.0;

    // 如果 x 小于等于 2.0，使用近似表达式计算修正 Bessel 函数 I_1(x)
    if (x <= T(2.0)) {
        // 初始化系数 a 为 A 数组的第一个元素
        T a = A[0];

        // 循环计算修正 Bessel 函数 I_1(x) 的近似值
        for (uint64_t index = 1; index < 11; index++) {
            p = q;
            q = a;
            a = (x * x - T(2.0)) * q - p + A[index];
        }

        // 返回修正 Bessel 函数 I_1(x) 近似值的对数乘以 0.5*x 的指数形式
        return (std::log(T(0.5) * x) * modified_bessel_i1_forward(x) + T(0.5) * (a - p) / x) * std::exp(x);
    }

    // 如果 x 大于 2.0，使用系数数组 B 计算修正 Bessel 函数 I_1(x) 的近似值
    T b = B[0];

    // 循环计算修正 Bessel 函数 I_1(x) 的近似值
    for (uint64_t index = 1; index < 25; index++) {
        p = q;
        q = b;
        b = (T(8.0) / x - T(2.0)) * q - p + B[index];
    }

    // 返回修正 Bessel 函数 I_1(x) 近似值的 0.5*(b - p)/sqrt(x)
    return (T(0.5) * (b - p) / std::sqrt(x));
// 返回修正切比雪夫多项式 T_n(x) 的值，当 n 为负数时返回 0
template<typename T>
inline C10_HOST_DEVICE T shifted_chebyshev_polynomial_t_forward(T x, int64_t n) {
    // 如果 n 小于 0，返回 0
    if (n < 0) {
        return T(0.0);
    }

    // 当 x 等于 1 时，返回 1
    if (x == T(1.0)) {
        return T(1.0);
    }

    // 当 x 等于 0 时，
    if (x == T(0.0)) {
        // 如果 n 是偶数，返回 1
        if (n % 2 == 0) {
            return T(1.0);
        }
        // 如果 n 是奇数，返回 -1
        return T(-1.0);
    }

    // 当 n 大于 6 且 abs(x + x - 1.0) 小于 1 时，
    if ((n > 6) && (std::abs(x + x - T(1.0)) < T(1.0))) {
        // 返回 cos(n * acos(x + x - 1.0))
        return std::cos(n * std::acos(x + x - T(1.0)));
    }

    // 当 n 等于 0 时，返回 1
    if (n == 0) {
        return T(1.0);
    }

    // 当 n 等于 1 时，返回 x + x - 1.0
    if (n == 1) {
        return x + x - T(1.0);
    }

    // 初始化 p 为 1，q 为 x + x - 1.0
    T p = T(1.0);
    T q = x + x - T(1.0);
    T r;

    // 循环计算 T_n(x) 的值，直到 k 等于 n
    for (int64_t k = 2; k <= n; k++) {
        r = (x + x - T(1.0) + (x + x - T(1.0))) * q - p;
        p = q;
        q = r;
    }

    // 返回 T_n(x) 的值
    return r;
} // shifted_chebyshev_polynomial_t_forward(T x, int64_t n)

// 返回修正切比雪夫多项式 T_n(x) 的值，当 n 为实数时转换为整数后计算
template<typename T, bool is_cuda=false>
inline C10_HOST_DEVICE T shifted_chebyshev_polynomial_t_forward(T x, T n) {
    return shifted_chebyshev_polynomial_t_forward(x, static_cast<int64_t>(n));
} // shifted_chebyshev_polynomial_t_forward(T x, T n)

// 返回修正切比雪夫多项式 U_n(x) 的值，当 n 为负数时返回 0
template<typename T>
inline C10_HOST_DEVICE T shifted_chebyshev_polynomial_u_forward(T x, int64_t n) {
    // 如果 n 小于 0，返回 0
    if (n < 0) {
        return T(0.0);
    }

    // 当 x 等于 1 时，返回 n + 1
    if (x == T(1.0)) {
        return n + 1;
    }

    // 当 x 等于 0 时，
    if (x == T(0.0)) {
        // 如果 n 是偶数，返回 n + 1
        if (n % 2 == 0) {
            return n + 1;
        }
        // 如果 n 是奇数，返回 -(n + 1)
        return -(n + 1);
    }

    // 当 n 大于 6 且 abs(x + x - 1.0) 小于 1 时，
    if ((n > 6) && (std::abs(x + x - T(1.0)) < T(1.0))) {
        // 如果 sin(acos(x + x - 1.0)) 不等于 0，
        if (std::sin(std::acos(x + x - T(1.0))) != T(0.0)) {
            // 返回 sin((n + 1) * acos(x + x - 1.0)) / sin(acos(x + x - 1.0))
            return std::sin((n + 1) * std::acos(x + x - T(1.0))) / std::sin(std::acos(x + x - T(1.0)));
        }
        // 返回 (n + 1) * cos((n + 1) * acos(x + x - 1.0)) / (x + x - 1.0)
        return (n + 1) * std::cos((n + 1) * std::acos(x + x - T(1.0))) / (x + x - T(1.0));
    }

    // 当 n 等于 0 时，返回 1
    if (n == 0) {
        return T(1.0);
    }

    // 当 n 等于 1 时，返回 x + x - 1.0 + (x + x - 1.0)
    if (n == 1) {
        return x + x - T(1.0) + (x + x - T(1.0));
    }

    // 初始化 p 为 1，q 为 x + x - 1.0 + (x + x - 1.0)
    T p = T(1.0);
    T q = x + x - T(1.0) + (x + x - T(1.0));
    T r;

    // 循环计算 U_n(x) 的值，直到 k 等于 n
    for (int64_t k = 2; k <= n; k++) {
        r = (x + x - T(1.0) + (x + x - T(1.0))) * q - p;
        p = q;
        q = r;
    }

    // 返回 U_n(x) 的值
    return r;
} // shifted_chebyshev_polynomial_u_forward(T x, int64_t n)

// 返回修正切比雪夫多项式 U_n(x) 的值，当 n 为实数时转换为整数后计算
template<typename T, bool is_cuda=false>
inline C10_HOST_DEVICE T shifted_chebyshev_polynomial_u_forward(T x, T n) {
    return shifted_chebyshev_polynomial_u_forward(x, static_cast<int64_t>(n));
} // shifted_chebyshev_polynomial_u_forward(T x, T n)

// 返回修正切比雪夫多项式 V_n(x) 的值，当 n 为负数时返回 0
template<typename T>
inline C10_HOST_DEVICE T shifted_chebyshev_polynomial_v_forward(T x, int64_t n) {
    // 如果 n 小于 0，返回 0
    if (n < 0) {
        return T(0.0);
    }

    // 当 x 等于 1 时，返回 1
    if (x == T(1.0)) {
        return T(1.0);
    }

    // 当 x 等于 0 时，
    if (x == T(0.0)) {
        // 如果 n 是偶数，返回 n + n + 1
        if (n % 2 == 0) {
            return (n + n + 1);
        }
        // 如果 n 是奇数，返回 -(n + n + 1)
        return -(n + n + 1);
    }
    // 检查条件：n 大于 6 且 abs(x + x - T(1.0)) 小于 T(1.0)
    if ((n > 6) && (std::abs(x + x - T(1.0)) < T(1.0))) {
        // 检查 sin(acos(x + x - T(1.0)) / 2.0) 是否不等于 1.0
        if (std::sin(std::acos(x + x - T(1.0)) / T(2.0)) != T(1.0)) {
            // 返回计算结果：cos(((n) + 0.5) * acos(x + x - T(1.0))) / cos(acos(x + x - T(1.0)) / 2.0)
            return std::cos(((n) + T(0.5)) * std::acos(x + x - T(1.0))) / std::cos(std::acos(x + x - T(1.0)) / T(2.0));
        }

        // 如果 n 是偶数，返回 n + n + 1
        if (n % 2 == 0) {
            return n + n + 1;
        }

        // 如果 n 是奇数，返回 -(n + n + 1)
        return -(n + n + 1);
    }

    // 如果 n 等于 0，返回 1.0
    if (n == 0) {
        return T(1.0);
    }

    // 如果 n 等于 1，返回 x + x - 1.0 + (x + x - 1.0) - 1.0
    if (n == 1) {
        return x + x - T(1.0) + (x + x - T(1.0)) - T(1.0);
    }

    // 初始化变量 p 和 q
    T p = T(1.0);
    T q = x + x - T(1.0) + (x + x - T(1.0)) - T(1.0);
    T r;

    // 计算阶乘项系数，使用循环计算直到 k = n
    for (int64_t k = 2; k <= n; k++) {
        r = (x + x - T(1.0) + (x + x - T(1.0))) * q - p;
        p = q;
        q = r;
    }

    // 返回结果 r
    return r;
} // 结束 shifted_chebyshev_polynomial_v_forward 函数模板的定义

template<typename T, bool is_cuda=false>
inline C10_HOST_DEVICE T shifted_chebyshev_polynomial_v_forward(T x, T n) {
    // 将函数参数 n 转换为 int64_t 类型，然后调用重载的 shifted_chebyshev_polynomial_v_forward 函数
    return shifted_chebyshev_polynomial_v_forward(x, static_cast<int64_t>(n));
} // 结束 shifted_chebyshev_polynomial_v_forward 函数模板的定义

template<typename T>
inline C10_HOST_DEVICE T shifted_chebyshev_polynomial_w_forward(T x, int64_t n) {
    // 如果 n 小于 0，则返回 0.0
    if (n < 0) {
        return T(0.0);
    }

    // 如果 x 等于 1.0，则返回 n + n + 1
    if (x == T(1.0)) {
        return n + n + 1;
    }

    // 如果 x 等于 0.0
    if (x == T(0.0)) {
        // 如果 n 是偶数，则返回 1.0；否则返回 -1.0
        if (n % 2 == 0) {
            return T(1.0);
        }
        return T(-1.0);
    }

    // 如果 n 大于 4 且 abs(x + x - 1.0) 小于 1.0
    if ((n > 4) && (std::abs(x + x - T(1.0)) < T(1.0))) {
        // 如果 cos(acos(x + x - 1.0) / 2.0) 不等于 1.0
        if (std::cos(std::acos(x + x - T(1.0)) / T(2.0)) != T(1.0)) {
            // 返回 sin((n + 0.5) * acos(x + x - 1.0)) / sin(acos(x + x - 1.0) / 2.0)
            return std::sin((n + T(0.5)) * std::acos(x + x - T(1.0))) / std::sin(std::acos(x + x - T(1.0)) / T(2.0));
        }

        // 如果 n 是偶数，则返回 1.0；否则返回 -1.0
        if (n % 2 == 0) {
            return T(1.0);
        }
        return T(-1.0);
    }

    // 如果 n 等于 0，则返回 1.0
    if (n == 0) {
        return T(1.0);
    }

    // 如果 n 等于 1，则返回 x + x - 1.0 + (x + x - 1.0) + 1.0
    if (n == 1) {
        return x + x - T(1.0) + (x + x - T(1.0)) + T(1.0);
    }

    // 初始化变量 p 和 q
    T p = T(1.0);
    T q = x + x - T(1.0) + (x + x - T(1.0)) + T(1.0);
    T r;

    // 计算从 k=2 到 k=n 的 shifted Chebyshev polynomial
    for (int64_t k = 2; k <= n; k++) {
        r = (x + x - T(1.0) + (x + x - T(1.0))) * q - p;
        p = q;
        q = r;
    }

    // 返回结果 r
    return r;
} // 结束 shifted_chebyshev_polynomial_w_forward 函数模板的定义

template<typename T, bool is_cuda=false>
inline C10_HOST_DEVICE T shifted_chebyshev_polynomial_w_forward(T x, T n) {
    // 将函数参数 n 转换为 int64_t 类型，然后调用重载的 shifted_chebyshev_polynomial_w_forward 函数
    return shifted_chebyshev_polynomial_w_forward(x, static_cast<int64_t>(n));
} // 结束 shifted_chebyshev_polynomial_w_forward 函数模板的定义

template<typename T>
inline C10_HOST_DEVICE T spherical_bessel_j0_forward(T x) {
    // 如果 x 是无穷大，则返回 0.0
    if (std::isinf(x)) {
        return T(0.0);
    }

    // 如果 abs(x) 小于 0.5，则计算球贝塞尔函数 J0(x) 的近似值
    if (std::abs(x) < T(0.5)) {
        return T(1.0) + x * x * (T(-1.0) / T(6.0) + x * x * (T(1.0) / T(120.0) + x * x * (T(-1.0) / T(5040.0) + x * x * (T(1.0) / T(362880.0) + x * x * (T(-1.0) / T(39916800.0) + x * x * (T(1.0) / T(6227020800.0)))))));
    }

    // 否则返回 sin(x) / x
    return std::sin(x) / x;
} // 结束 spherical_bessel_j0_forward 函数模板的定义

C10_CLANG_DIAGNOSTIC_POP()
```