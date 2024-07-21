# `.\pytorch\aten\src\ATen\cpu\vml.h`

```
#pragma once
// 只包含一次头文件的指令

#include <ATen/Config.h>
// 引入 ATen 库的配置文件

#include <ATen/Parallel.h>
// 引入 ATen 库的并行处理模块

#include <ATen/OpMathType.h>
// 引入 ATen 库的数学操作类型

#include <ATen/cpu/vec/functional.h>
// 引入 ATen 库的向量化功能函数

#include <ATen/cpu/vec/vec.h>
// 引入 ATen 库的向量化类型

#include <c10/util/complex.h>
// 引入 c10 库的复数处理工具

// This header implements various unary operations using a MKL VML style
// interface.
// 该头文件使用 MKL VML 风格接口实现了各种一元操作

// It implements various functions with a simple interface
// For example it enables the user to call vsin(float* out, const float* in,
// size) This functions takes a pointer to a continuous output array of floats and
// a constant input array. It will then apply sin to each value in the input
// array and write the result into the output array. out and in may point to the
// same memory, i.e. this fully supports in-place operations. These functions
// also implement their own parallelization, so take precautions when calling
// these from threaded functions.
// 该文件实现了多种简单接口的函数。例如，它允许用户调用 vsin(float* out, const float* in,
// size)，此函数接收一个指向连续输出浮点数数组的指针和一个常量输入数组。然后它会对输入数组中的每个值应用 sin 函数，并将结果写入输出数组。out 和 in 可能指向同一内存，因此完全支持原位操作。这些函数也实现了自己的并行化，因此在从线程化函数中调用时需要注意。

// When MKL is available it will call into MKL's VML library similar to NumPy
// If MKL is not available it will use SLEEF.
// 如果 MKL 可用，则会调用 MKL 的 VML 库，类似于 NumPy。如果 MKL 不可用，则会使用 SLEEF。

// This file might be compiled under AVX or AVX2 when called from e.g.
// UnaryOpsKernel.cpp
// 当从 UnaryOpsKernel.cpp 等文件调用时，该文件可能会使用 AVX 或 AVX2 进行编译。

#include <algorithm>
// 引入算法标准库

#include <cstddef>
// 引入标准大小类型

#include <cstdint>
// 引入标准整数类型

#include <cstring>
// 引入 C 字符串操作标准库

#include <type_traits>
// 引入类型特性标准库

#if AT_MKL_ENABLED() && !defined(__APPLE__)
#include <mkl.h>
#endif
// 如果 ATen 使用了 MKL 并且不是在苹果系统下，则引入 MKL 头文件

namespace at {
namespace vml {
inline namespace CPU_CAPABILITY {

using namespace vec;
// 使用 vec 命名空间

template <typename scalar_t>
inline void vrsqrt(scalar_t* out, scalar_t* in, int64_t size) {
  // 定义 vrsqrt 函数，实现对输入数组每个元素的平方根倒数操作并存入输出数组
  parallel_for(0, size, 2048, [out, in](int64_t begin, int64_t end) {
    // 并行化处理范围在 [begin, end) 内的数据
    map(
        [](const Vectorized<scalar_t>& x) {
          return Vectorized<scalar_t>((scalar_t)(1)) / x.sqrt();
          // 对每个向量化元素计算其平方根的倒数
        },
        out + begin,
        in + begin,
        end - begin);
  });
}

// NB: We ignore numerical errors by convention and leave them to the user

#define IMPLEMENT_VML(op)                                               \
  template <typename scalar_t>                                          \
  inline void v##op(scalar_t* out, const scalar_t* in, int64_t size) {  \
    using vec_t = Vectorized<vec_scalar_t<scalar_t>>;                   \
    vec::map([](vec_t x) { return x.op(); }, out, in, size);            \
  }                                                                     \
// 宏定义，生成模板函数 v##op，实现各种操作（如 abs, acos, sin 等）的向量化映射

IMPLEMENT_VML(abs)
IMPLEMENT_VML(acos)
IMPLEMENT_VML(asin)
IMPLEMENT_VML(atan)
IMPLEMENT_VML(atanh)
IMPLEMENT_VML(ceil)
IMPLEMENT_VML(cos)
// IMPLEMENT_VML(cosh)
IMPLEMENT_VML(erf)
IMPLEMENT_VML(erfc)
IMPLEMENT_VML(erfinv)
IMPLEMENT_VML(exp)
IMPLEMENT_VML(expm1)
IMPLEMENT_VML(floor)
IMPLEMENT_VML(i0)
IMPLEMENT_VML(i0e)
IMPLEMENT_VML(digamma)
IMPLEMENT_VML(reciprocal)
IMPLEMENT_VML(log)
IMPLEMENT_VML(log10)
IMPLEMENT_VML(log1p)
IMPLEMENT_VML(log2)
IMPLEMENT_VML(neg)
IMPLEMENT_VML(sin)
// IMPLEMENT_VML(sinh)
IMPLEMENT_VML(sqrt)
IMPLEMENT_VML(round)
IMPLEMENT_VML(rsqrt)
IMPLEMENT_VML(tan)
IMPLEMENT_VML(tanh)
IMPLEMENT_VML(trunc)
IMPLEMENT_VML(lgamma)

#if AT_MKL_ENABLED() && !defined(__APPLE__)

// NB: LP64 MKL is the most commonly used and thus we assume it here. That means
// MKL 使用 LP64 格式，这在常见的情况下是最常用的假设
// 确保 MKL_INT 被假定为 int 类型，通常对应 int32_t 或 int64_t
static_assert(
    std::is_same_v<MKL_INT, int32_t> || std::is_same_v<MKL_INT, int64_t>,
    "MKL_INT is assumed to be int32_t or int64_t");

// 定义一个宏，用于生成特定操作的 MKL 子例程的模板
#define IMPLEMENT_VML_MKL_STUB(op, mklop, type, mkltype)                \
  template <>                                                           \
  inline void v##op(type * out, const type * in, int64_t size) {        \
    // 获取 MKL_INT 的最大值
    int64_t max_mkl_ind = std::numeric_limits<MKL_INT>::max();          \
    // 如果 size 不超过最大索引值，则直接调用单次 MKL 子例程
    if (size <= static_cast<int64_t>(max_mkl_ind)) {                    \
      vm##mkltype##mklop(                                               \
          size, in, out, VML_HA | VML_FTZDAZ_OFF | VML_ERRMODE_IGNORE); \
    } else {                                                            \
      // 否则，分块处理数据
      MKL_INT ind = 0;                                                  \
      int64_t chunks = size / max_mkl_ind;                              \
      int64_t rest = size % max_mkl_ind;                                \
      // 对于每个块，调用相应的 MKL 子例程
      for (; ind < chunks; ind++) {                                     \
        vm##mkltype##mklop(                                             \
            max_mkl_ind,                                                \
            in + ind * max_mkl_ind,                                     \
            out + ind * max_mkl_ind,                                    \
            VML_HA | VML_FTZDAZ_OFF | VML_ERRMODE_IGNORE);              \
      }                                                                 \
      // 处理剩余的数据块
      vm##mkltype##mklop(                                               \
          rest,                                                         \
          in + ind * max_mkl_ind,                                       \
          out + ind * max_mkl_ind,                                      \
          VML_HA | VML_FTZDAZ_OFF | VML_ERRMODE_IGNORE);                \
    }                                                                   \
  }

// 定义宏，用于生成各种类型和操作的 MKL 子例程
#define IMPLEMENT_VML_MKL(op, mklop)          \
  IMPLEMENT_VML_MKL_STUB(op, mklop, float, s) \
  IMPLEMENT_VML_MKL_STUB(op, mklop, double, d)

// 下面一系列宏调用实例化了各种数学函数的 MKL 子例程模板，以供不同类型的浮点数使用

// 注释部分说明了在某些配置中暂时禁用了特定函数，如 abs、cosh、sinh、expm1
IMPLEMENT_VML_MKL(acos, Acos)
IMPLEMENT_VML_MKL(asin, Asin)
IMPLEMENT_VML_MKL(atan, Atan)
IMPLEMENT_VML_MKL(cos, Cos)
// IMPLEMENT_VML_MKL(cosh, Cosh)
IMPLEMENT_VML_MKL(erf, Erf)
IMPLEMENT_VML_MKL(erfc, Erfc)
IMPLEMENT_VML_MKL(erfinv, ErfInv)
IMPLEMENT_VML_MKL(exp, Exp)
// IMPLEMENT_VML_MKL(expm1, Expm1)
IMPLEMENT_VML_MKL(log, Ln)
IMPLEMENT_VML_MKL(log10, Log10)
IMPLEMENT_VML_MKL(sin, Sin)
// IMPLEMENT_VML_MKL(sinh, Sinh)
IMPLEMENT_VML_MKL(sqrt, Sqrt)
IMPLEMENT_VML_MKL(tan, Tan)
IMPLEMENT_VML_MKL(tanh, Tanh)
IMPLEMENT_VML_MKL(trunc, Trunc)
#if INTEL_MKL_VERSION >= 20180406
// 如果 INTEL_MKL_VERSION 大于或等于 20180406，则进行以下条件编译

IMPLEMENT_VML_MKL(log2, Log2)
// 使用 MKL 实现 log2 函数，对应的宏是 Log2

#endif
// 结束条件编译块

#endif
// 结束 ifdef 块

} // namespace
// 结束 vml 命名空间

} // namespace vml
// 结束 vml 命名空间

} // namespace at
// 结束 at 命名空间
```