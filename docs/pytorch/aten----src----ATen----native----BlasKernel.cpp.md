# `.\pytorch\aten\src\ATen\native\BlasKernel.cpp`

```
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/Context.h>
#include <ATen/Config.h>
#include <ATen/OpMathType.h>
#include <ATen/Parallel.h>
#include <c10/core/ScalarType.h>
#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>
#include <c10/util/Unroll.h>
#include <c10/util/complex.h>
#include <c10/util/irange.h>
#include <algorithm>
#include <climits>
#include <limits>

#if defined(__aarch64__) && !defined(C10_MOBILE)
#include <arm_neon.h>
#endif

// 禁止因未使用的函数产生的警告，这些警告在此处被推送并且被忽略
C10_DIAGNOSTIC_PUSH_AND_IGNORED_IF_DEFINED("-Wunused-function")
namespace {

/// Wrapper for const_cast<T*> with type-inference.
///
/// Use this to call into APIs that are not const-correct.
// 模板函数，用于去除 const 修饰符，允许对不是 const-correct 的 API 进行调用
template <typename T>
T* remove_const(const T* x) {
  return const_cast<T*>(x);
}

} // namespace

#if AT_BUILD_WITH_BLAS()
// 使用 extern "C" 关键字声明一些 BLAS（Basic Linear Algebra Subprograms）函数，这些函数是 C 语言接口的实现

extern "C" double ddot_(int *n, double *x, int *incx, double *y, int *incy);
extern "C" void dscal_(int *n, double *a, double *x, int *incx);
extern "C" void sscal_(int *n, float *a, float *x, int *incx);
extern "C" void dgemv_(char *trans, int *m, int *n, double *alpha, double *a, int *lda, double *x, int *incx, double *beta, double *y, int *incy);
extern "C" void sgemv_(char *trans, int *m, int *n, float *alpha, float *a, int *lda, float *x, int *incx, float *beta, float *y, int *incy);

// 如果配置了 AT_BLAS_F2C 宏，则定义 ffloat 为 double 类型，否则定义为 float 类型
#if AT_BLAS_F2C()
# define ffloat double
#else
# define ffloat float
#endif

// 如果配置了 AT_BLAS_USE_CBLAS_DOT 宏，则使用 cblas_sdot 函数实现 sdot_ 函数
// 其余的 cdotu_, zdotu_, cdotc_, zdotc_ 函数也是类似地用 CBLAS 库的函数实现

#if AT_BLAS_USE_CBLAS_DOT()
  extern "C" float cblas_sdot(const int n, const float *x, const int incx, const float *y, const int incy);
  extern "C" void cblas_cdotu_sub(const int n, const void *x, const int incx, const void *y, const int incy, void *dotu);
  extern "C" void cblas_zdotu_sub(const int n, const void *x, const int incx, const void *y, const int incy, void *dotu);
  extern "C" void cblas_cdotc_sub(const int n, const void *x, const int incx, const void *y, const int incy, void *dotc);
  extern "C" void cblas_zdotc_sub(const int n, const void *x, const int incx, const void *y, const int incy, void *dotc);

  // 实现 sdot_ 函数，调用 cblas_sdot 函数
  static inline ffloat sdot_(const int *n, const float *x, const int *incx, const float *y, const int *incy)
  {
    return cblas_sdot(*n, x, *incx, y, *incy);
  }

  // 实现 cdotu_, zdotu_, cdotc_, zdotc_ 函数，分别调用对应的 CBLAS 函数
  static inline void cdotu_(std::complex<float> *res, const int *n, const std::complex<float> *x, const int *incx,
  const std::complex<float> *y, const int *incy) {
    cblas_cdotu_sub(*n, x, *incx, y, *incy, res);
  }
  static inline void zdotu_(std::complex<double> *res, const int *n, const std::complex<double> *x, const int *incx,
  const std::complex<double> *y, const int *incy) {
    cblas_zdotu_sub(*n, x, *incx, y, *incy, res);
  }
  static inline void cdotc_(std::complex<float> *res, const int *n, const std::complex<float> *x, const int *incx,
  const std::complex<float> *y, const int *incy) {
    cblas_cdotc_sub(*n, x, *incx, y, *incy, res);
  }
  static inline void zdotc_(std::complex<double> *res, const int *n, const std::complex<double> *x, const int *incx,
  const std::complex<double> *y, const int *incy) {


这段代码主要是一些 C++ 头文件的包含和一些外部函数的声明，涉及到 BLAS 库的使用和一些模板函数的定义。
    cblas_zdotc_sub(*n, x, *incx, y, *incy, res);



// 调用 cblas_zdotc_sub 函数计算复数向量 x 和 y 的共轭点积，将结果存储在 res 中
cblas_zdotc_sub(*n, x, *incx, y, *incy, res);
#else
  // 声明 C 接口的单精度浮点点积函数 sdot_
  extern "C" ffloat sdot_(int *n, float *x, int *incx, float *y, int *incy);
  // 声明 C 接口的复数单精度浮点点乘法函数 cdotu_
  extern "C" void cdotu_(std::complex<float> *res, int *n, std::complex<float> *x, int *incx, std::complex<float> *y, int *incy);
  // 声明 C 接口的复数双精度浮点点乘法函数 zdotu_
  extern "C" void zdotu_(std::complex<double> *res, int *n, std::complex<double> *x, int *incx, std::complex<double> *y, int *incy);
  // 声明 C 接口的复数单精度浮点点乘法函数 cdotc_
  extern "C" void cdotc_(std::complex<float> *res, int *n, std::complex<float> *x, int *incx, std::complex<float> *y, int *incy);
  // 声明 C 接口的复数双精度浮点点乘法函数 zdotc_
  extern "C" void zdotc_(std::complex<double> *res, int *n, std::complex<double> *x, int *incx, std::complex<double> *y, int *incy);
#endif // AT_BLAS_USE_CBLAS_DOT
#endif // AT_BUILD_WITH_BLAS

namespace at::native {

namespace blas_impl {
#if defined(__aarch64__) && !defined(C10_MOBILE)
// 声明在特定条件下可用的 fp16_gemv_notrans 函数
void fp16_gemv_notrans(
    const int m,
    const int n,
    const float alpha,
    const float16_t* a,
    const int lda,
    const float16_t* x,
    const int incx,
    const float beta,
    float16_t* y,
    const int incy);

// 声明在特定条件下可用的 fp16_gemv_trans 函数
void fp16_gemv_trans(
    const int m,
    const int n,
    const float alpha,
    const float16_t* a,
    const int lda,
    const float16_t* x,
    const int incx,
    const float beta,
    float16_t* y,
    const int incy);

// 声明在特定条件下可用的 fp16_dot_with_fp32_arith 函数
float fp16_dot_with_fp32_arith(
    const float16_t* vec1,
    const float16_t* vec2,
    int64_t len);

// 声明在特定条件下可用的 bf16_gemv_trans 函数
void bf16_gemv_trans(
    const int m,
    const int n,
    const at::BFloat16 alpha,
    const at::BFloat16* a,
    const int lda,
    const at::BFloat16* x,
    const int incx,
    const at::BFloat16 beta,
    at::BFloat16* y,
    const int incy);

// 声明在特定条件下可用的 bf16_dot_with_fp32_arith 函数
float bf16_dot_with_fp32_arith(
    const at::BFloat16* vec1,
    const at::BFloat16* vec2,
    int64_t len);
#endif

// 模板定义：用于确定是否使用快速路径的 scal 函数
template <typename scalar_t>
bool scal_use_fast_path(C10_UNUSED int64_t n, C10_UNUSED int64_t incx) {
  return false;
}

// 模板定义：用于确定是否使用快速路径的 gemv 函数
template <typename scalar_t>
bool gemv_use_fast_path(C10_UNUSED char trans, C10_UNUSED int64_t m,
                        C10_UNUSED int64_t n, C10_UNUSED scalar_t alpha,
                        C10_UNUSED int64_t lda,
                        C10_UNUSED int64_t incx, C10_UNUSED scalar_t beta,
                        C10_UNUSED int64_t incy) {
  return false;
}

// 模板定义：当不应调用 scal_fast_path 时抛出错误
template <typename scalar_t>
void scal_fast_path(C10_UNUSED int *n, C10_UNUSED scalar_t *a, C10_UNUSED scalar_t *x, C10_UNUSED int *incx) {
  TORCH_INTERNAL_ASSERT(false, "scal_fast_path shouldn't be called for this configuration");
}

// 模板定义：当不应调用 gemv_fast_path 时抛出错误
template <typename scalar_t>
void gemv_fast_path(C10_UNUSED const char *trans, C10_UNUSED const int *m, C10_UNUSED const int *n,
                    C10_UNUSED  const scalar_t *alpha, C10_UNUSED const scalar_t *a, C10_UNUSED const int *lda,
                    C10_UNUSED  const scalar_t *x, C10_UNUSED const int *incx, C10_UNUSED const scalar_t *beta,
                    C10_UNUSED  scalar_t *y, C10_UNUSED const int *incy) {
  TORCH_INTERNAL_ASSERT(false, "gemv_fast_path shouldn't be called for this configuration");
}
#define INSTANTIATE(scalar_t)                                                                                                                                                     \
template bool scal_use_fast_path<scalar_t>(int64_t n, int64_t incx);                                                                                                              \  // 实例化模板函数 scal_use_fast_path，用于特定类型 scalar_t
template bool gemv_use_fast_path<scalar_t>(char trans, int64_t m, int64_t n, scalar_t alpha, int64_t lda, int64_t incx, scalar_t beta, int64_t incy); \  // 实例化模板函数 gemv_use_fast_path，用于特定类型 scalar_t
template void gemv_fast_path<scalar_t>(const char *trans, const int *m, const int *n, const scalar_t *alpha, const scalar_t *a, const int *lda, const scalar_t *x, const int *incx, const scalar_t *beta, scalar_t *y, const int *incy);      \  // 实例化模板函数 gemv_fast_path，用于特定类型 scalar_t
template void scal_fast_path<scalar_t>(int *n, scalar_t *a, scalar_t *x, int *incx);  // 实例化模板函数 scal_fast_path，用于特定类型 scalar_t

#if AT_BUILD_WITH_BLAS()
template <>                                                                                                                                                                      \  // 为双精度浮点数特化 scal_use_fast_path，检查 n 和 incx 是否在整型极限范围内
bool scal_use_fast_path<double>(int64_t n, int64_t incx) {
  auto intmax = std::numeric_limits<int>::max();                                                                                                                                  \  // 计算 int 型最大值
  return n <= intmax && incx <= intmax;                                                                                                                                            \  // 检查 n 和 incx 是否小于等于最大值
}

template <>                                                                                                                                                                      \  // 为单精度浮点数特化 scal_use_fast_path，委托给 double 特化版本
bool scal_use_fast_path<float>(int64_t n, int64_t incx) {
  return scal_use_fast_path<double>(n, incx);                                                                                                                                      \  // 委托给 double 特化版本
}

template <>                                                                                                                                                                      \  // 为双精度浮点数特化 scal_fast_path，调用 BLAS 库中的 dscal 函数
void scal_fast_path<double>(int *n, double *a, double *x, int *incx) {
  dscal_(n, a, x, incx);                                                                                                                                                          \  // 调用 BLAS 的 dscal 函数
}

template <>                                                                                                                                                                      \  // 为单精度浮点数特化 scal_fast_path，调用 BLAS 库中的 sscal 函数
void scal_fast_path<float>(int *n, float *a, float *x, int *incx) {
  sscal_(n, a, x, incx);                                                                                                                                                          \  // 调用 BLAS 的 sscal 函数
}

template <>                                                                                                                                                                      \  // 为单精度浮点数特化 gemv_use_fast_path，检查参数是否在整型极限范围内
bool gemv_use_fast_path<float>(C10_UNUSED char trans, int64_t m, int64_t n, C10_UNUSED float alpha, int64_t lda, int64_t incx, C10_UNUSED float beta, int64_t incy) {
  auto intmax = std::numeric_limits<int>::max();                                                                                                                                  \  // 计算 int 型最大值
  return (m <= intmax) && (n <= intmax) && (lda <= intmax) &&                                                                                                                      \  // 检查 m, n, lda, incx, incy 是否在整型极限范围内
         (incx > 0) && (incx <= intmax) && (incy > 0) && (incy <= intmax);
}

template <>                                                                                                                                                                      \  // 为双精度浮点数特化 gemv_use_fast_path，委托给 float 特化版本
bool gemv_use_fast_path<double>(C10_UNUSED char trans, int64_t m, int64_t n, C10_UNUSED double alpha, int64_t lda, int64_t incx, C10_UNUSED double beta, int64_t incy) {
  return gemv_use_fast_path<float>(trans, m, n, (float)alpha, lda, incx, (float)beta, incy);                                                                                      \  // 委托给 float 特化版本
}

template <>                                                                                                                                                                      \  // 为双精度浮点数特化 gemv_fast_path，调用 BLAS 库中的 dgemv 函数
void gemv_fast_path<double>(const char *trans, const int *m, const int *n, const double *alpha, const double *a, const int *lda, const double *x, const int *incx, const double *beta, double *y, const int *incy) {
  dgemv_(remove_const(trans), remove_const(m), remove_const(n), remove_const(alpha), remove_const(a), remove_const(lda), remove_const(x), remove_const(incx), remove_const(beta), y, remove_const(incy));       \  // 调用 BLAS 的 dgemv 函数
}

template <>                                                                                                                                                                      \  // 为单精度浮点数特化 gemv_fast_path，调用 BLAS 库中的 sgemv 函数
void gemv_fast_path<float>(const char *trans, const int *m, const int *n, const float *alpha, const float *a, const int *lda, const float *x, const int *incx, const float *beta, float *y, const int *incy) {
  sgemv_(remove_const(trans), remove_const(m), remove_const(n), remove_const(alpha), remove_const(a), remove_const(lda), remove_const(x), remove_const(incx), remove_const(beta), y, remove_const(incy));          \  // 调用 BLAS 的 sgemv 函数
}
#else
INSTANTIATE(float);                                                                                                                                                              \  // 实例化 float 类型模板
INSTANTIATE(double);                                                                                                                                                             \  // 实例化 double 类型模板
#endif // AT_BUILD_WITH_BLAS

INSTANTIATE(uint8_t);                                                                                                                                                            \  // 实例化 uint8_t 类型模板
INSTANTIATE(int8_t);                                                                                                                                                             \  // 实例化 int8_t 类型模板
// 实例化模板函数，使用 int16_t 类型
INSTANTIATE(int16_t);
// 实例化模板函数，使用 int 类型
INSTANTIATE(int);
// 实例化模板函数，使用 int64_t 类型
INSTANTIATE(int64_t);

// 如果是 ARM 64 位且非移动平台，定义模板函数特化，返回 false
#if defined(__aarch64__) && !defined(C10_MOBILE)
template <>
bool scal_use_fast_path<at::Half>(C10_UNUSED int64_t n, C10_UNUSED int64_t incx) {
  return false;
}

// 模板函数特化，对于 Half 类型的 gemv 函数使用快速路径判断
template <>
bool gemv_use_fast_path<at::Half>(
    C10_UNUSED char trans,
    C10_UNUSED int64_t m,
    C10_UNUSED int64_t n,
    at::Half alpha,
    C10_UNUSED int64_t lda,
    C10_UNUSED int64_t incx,
    at::Half beta,
    C10_UNUSED int64_t incy) {
  return incx == 1 && c10::detail::fp16_from_bits(alpha.x) == 1.0f &&
    c10::detail::fp16_from_bits(beta.x) == 0.0f;
}

// 模板函数特化，对于 BFloat16 类型的 gemv 函数使用快速路径判断
template <>
bool gemv_use_fast_path<at::BFloat16>(
  C10_UNUSED char trans,
  C10_UNUSED int64_t m,
  C10_UNUSED int64_t n,
  at::BFloat16 alpha,
  C10_UNUSED int64_t lda,
  C10_UNUSED int64_t incx,
  at::BFloat16 beta,
  C10_UNUSED int64_t incy) {
  return (trans == 'T' || trans == 't') && incx == 1 && alpha == 1.0 && beta == 0.0;
}

// 如果 ARM 支持 FP16 算术，定义一系列与 FP16 相关的辅助函数和常量

// 将 float16x4_t 向量归约为 float16_t
static inline float16_t reduce(float16x4_t x) {
        auto sum = vpadd_f16(x, x);
        return vget_lane_f16(vpadd_f16(sum, sum), 0);
}

// 将 float16x8_t 向量归约为 float16_t
static inline float16_t reduce(float16x8_t x) {
        return reduce(vadd_f16(vget_low_f16(x), vget_high_f16(x)));
}

/*
 * 注意事项 [ GGML 版权声明 ]
 * 下面的 reduce() 重载和 fp16_dot_with_fp16_arith 函数是从 llama.cpp 中的 ggml_vec_dot_f16
 * 和周围的实用函数调整而来，因此这里需要包含必要的版权声明：
 *
 * MIT 许可证
 *
 * 版权所有（c）2023-2024 ggml 作者
 *
 * 在不受限制的情况下，允许任何获取此软件及其关联文档文件（以下简称“软件”）的人员处理
 * 软件，包括但不限于使用、复制、修改、合并、发布、分发、再许可和/或出售
 * 软件的副本，并允许向获得软件的其他人员提供这样做的许可，前提是：
 *
 * 上述版权声明和本许可声明应包含在所有
 * 软件的副本或重要部分中。
 *
 * 本软件按“原样”提供，不提供任何形式的明示或
 * 默示的保证，包括但不限于适销性保证、
 * 适用于特定目的和非侵权性的保证。在任何情况下也不应
 * 作者或版权持有人对任何索赔、损害或其他责任
 * 无论是在合同行为、侵权行为还是其他行为中发生，与
 * 软件或使用或其他交易中的其他行为相关。
 */
// 为 reduce() 需要的移位操作定义额外的常量
static constexpr auto kF16ElementsPerIterationShift = 7;
static constexpr auto kF16ElementsPerIteration = 1 << kF16ElementsPerIterationShift;
static_assert(kF16ElementsPerIteration == 128);

static constexpr auto kF16ElementsPerRegisterShift = 3;
static constexpr auto kF16ElementsPerRegister = 1 << kF16ElementsPerRegisterShift;
// 计算每个寄存器可以容纳的半精度浮点数元素个数，左移运算确定为2的幂次方

static_assert(kF16ElementsPerRegister == 8);
// 断言：确保每个寄存器能够容纳的半精度浮点数元素个数为8

static constexpr auto kF16RegistersPerIterationShift = kF16ElementsPerIterationShift - kF16ElementsPerRegisterShift;
// 计算每次迭代中使用的寄存器数目的位移量，通过减去每个寄存器能容纳的位移量得到

static constexpr auto kF16RegistersPerIteration = 1 << kF16RegistersPerIterationShift;
// 计算每次迭代中使用的寄存器数目，左移运算确定为2的幂次方

static_assert(kF16RegistersPerIteration == kF16ElementsPerIteration / kF16ElementsPerRegister);
// 断言：确保每次迭代中使用的寄存器数目等于每次迭代中使用的半精度浮点数元素数目除以每个寄存器能够容纳的元素数目

static inline double reduce(float16x8_t x[kF16RegistersPerIteration]) {
  int offset = kF16RegistersPerIteration;
  // 初始化偏移量为每次迭代中使用的寄存器数目
  c10::ForcedUnroll<kF16RegistersPerIterationShift>{}([&offset, &x](auto idx) {
    offset /= 2;
    // 每次迭代减半偏移量
    for (int i = 0; i < offset; ++i) {
      x[i] = vaddq_f16(x[i], x[offset + i]);
      // 使用半精度浮点数向量加法，将当前寄存器的一半元素累加到前半部分
    }
  });
  const float32x4_t t0 = vcvt_f32_f16(vget_low_f16(x[0]));
  // 将第一个寄存器的低位半精度浮点数转换为单精度浮点数向量
  const float32x4_t t1 = vcvt_f32_f16(vget_high_f16(x[0]));
  // 将第一个寄存器的高位半精度浮点数转换为单精度浮点数向量
  return (double)vaddvq_f32(vaddq_f32(t0, t1));
  // 将两个单精度浮点数向量相加并返回其和的双精度浮点数值
}

static inline float16x8_t f16_fma(float16x8_t a, float16x8_t b, float16x8_t c) {
#ifdef __ARM_FEATURE_FMA
  return vfmaq_f16(a, b, c);
  // 使用 FMA 指令进行半精度浮点数向量乘加运算
#else
  return vaddq_f16(a, vmulq_f16(b, c));
  // 否则，使用加法和乘法指令实现半精度浮点数向量乘加运算
#endif
}

static float fp16_dot_with_fp16_arith(const float16_t* x, const float16_t* a, int len) {
  float16x8_t sum[kF16RegistersPerIteration] = {vdupq_n_f16(0)};
  // 初始化用于累加的半精度浮点数向量数组，每个寄存器都初始化为零向量

  const auto len_aligned = len & ~(kF16ElementsPerIteration - 1);
  // 计算对齐长度，以确保每次迭代处理一整数倍的半精度浮点数元素个数

  for (int j = 0; j < len_aligned ; j += kF16ElementsPerIteration) {
    // 迭代处理对齐长度内的每一组数据
    for (int k = 0; k < kF16RegistersPerIteration; ++k) {
      const auto temp_x = vld1q_f16(x + j + k * kF16ElementsPerRegister);
      // 加载半精度浮点数向量 x 的数据到临时向量
      const auto temp_a = vld1q_f16(a + j + k * kF16ElementsPerRegister);
      // 加载半精度浮点数向量 a 的数据到临时向量
      sum[k] = f16_fma(sum[k], temp_x, temp_a);
      // 调用 f16_fma 函数进行半精度浮点数向量乘加运算，将结果累加到 sum 数组的相应位置
    }
  }

  auto reducedSum = reduce(sum);
  // 调用 reduce 函数对累加的结果进行最终的求和操作

  for (int j = len_aligned; j < len; ++j) {
    reducedSum += x[j] * a[j];
    // 处理未对齐部分的数据，使用标量乘法和加法进行累加
  }

  return reducedSum;
  // 返回累加的最终结果
}

// Rather than unrolling to process multiple rows (transposed columns)
// of matrix A at once as done in fp16_gemv_trans_fp16_arith, unroll
// along an individual dot product.
static void fp16_gemv_trans_fp16_arith_by_dot_products(const int m, const int n, const float16_t* a, const int lda, const float16_t *x, float16_t* y, int incy) {
  parallel_for(0, n, 1, [&](int begin, int end) {
    // 并行循环处理每个列向量
    for (int i = begin; i < end; ++i) {
      y[i * incy] = fp16_dot_with_fp16_arith(x, a + lda * i, m);
      // 调用 fp16_dot_with_fp16_arith 函数计算 x 和矩阵 A 的转置列向量之间的点积，并将结果写入 y 数组
    }
  });
}

static inline float reduce(float32x4_t x) {
        auto sum = vpaddq_f32(x, x);
        // 使用 SIMD 指令对单精度浮点数向量 x 中的元素进行两两相加
        return vgetq_lane_f32(vpaddq_f32(sum, sum), 0);
        // 将相加后的结果向量再次两两相加并返回结果向量的第一个元素，即最终的求和结果
}

static inline float32x4_t f32_fma(float32x4_t a, float32x4_t b, float32x4_t c) {
#ifdef __ARM_FEATURE_FMA
  return vfmaq_f32(a, b, c);
  // 使用 FMA 指令进行单精度浮点数向量乘加运算
#else
  return vaddq_f32(a, vmulq_f32(b, c));
  // 否则，使用加法和乘法指令实现单精度浮点数向量乘加运算
#endif
}

static inline float32x4_t f32_fma_low_f16(float32x4_t a, float16x8_t b, float16x8_t c) {
#ifdef __ARM_FEATURE_FP16_FML
  // 如果编译器支持 ARM v8.2 和 v8.3 中的可选指令，则使用 vfmlalq_low_f16 函数
  // 否则，使用 f32_fma 函数来实现浮点数乘加运算
  return vfmlalq_low_f16(a, b, c);
#else
  // 使用 f32_fma 函数来实现浮点数乘加运算，将 float16x8_t 类型的向量 b 和 c
  // 转换为 float32x4_t 类型进行计算
  return f32_fma(a, vcvt_f32_f16(vget_low_f16(b)), vcvt_f32_f16(vget_low_f16(c)));
#endif
}

static inline float32x4_t f32_fma_high_f16(float32x4_t a, float16x8_t b, float16x8_t c) {
#ifdef __ARM_FEATURE_FP16_FML
  // 如果编译器支持 ARM v8.2 和 v8.3 中的可选指令，则使用 vfmlalq_high_f16 函数
  // 否则，使用 f32_fma 函数来实现浮点数乘加运算
  return vfmlalq_high_f16(a, b, c);
#else
  // 使用 f32_fma 函数来实现浮点数乘加运算，将 float16x8_t 类型的向量 b 和 c
  // 转换为 float32x4_t 类型进行计算
  return f32_fma(a, vcvt_f32_f16(vget_high_f16(b)), vcvt_f32_f16(vget_high_f16(c)));
#endif
}

static inline float32x4_t f32_fma_f16(float32x4_t a, float16x4_t b, float16x4_t c) {
  // 将 float16x4_t 类型的向量 b 和 c 扩展为 float16x8_t 类型，然后调用 f32_fma_low_f16 函数
  // 进行浮点数乘加运算
  return f32_fma_low_f16(a, vcombine_f16(b, vdup_n_f16(0)), vcombine_f16(c, vdup_n_f16(0)));
}

// 以下 reduce 函数和 fp16_dot_with_fp32_arith 函数从 llama.cpp 的 ggml_vec_dot_f32 函数
// 和其周围的实用函数进行了适配。
// 请参阅上面的 GGML 版权声明以获取所需的注意事项。

// reduce() 函数需要移位操作，因此需要额外的常量。
static constexpr auto kF32ElementsPerIterationShift = 5;
static constexpr auto kF32ElementsPerIteration = 1 << kF32ElementsPerIterationShift;
static_assert(kF32ElementsPerIteration == 32);

static constexpr auto kF32ElementsPerRegisterShift = 2;
static constexpr auto kF32ElementsPerRegister = 1 << kF32ElementsPerRegisterShift;
static_assert(kF32ElementsPerRegister == 4);

static constexpr auto kF32RegisterPairsPerIteration = 4;
static constexpr auto kF32RegistersPerIteration = kF32RegisterPairsPerIteration * 2;
static constexpr auto kF32RegistersPerIterationShift = 3;
static_assert(kF32RegistersPerIteration == kF32ElementsPerIteration / kF32ElementsPerRegister);
static_assert(kF32RegistersPerIteration == 1 << kF32RegistersPerIterationShift);

static inline double reduce(float32x4_t x[kF32RegistersPerIteration]) {
  int offset = kF32RegistersPerIteration;
  // 使用 C10::ForcedUnroll 模板进行循环展开，以便优化性能
  c10::ForcedUnroll<kF32RegistersPerIterationShift>{}([&offset, &x](auto idx) {
    offset /= 2;
    for (int i = 0; i < offset; ++i) {
      // 对数组 x 中的每对元素进行加法运算
      x[i] = vaddq_f32(x[i], x[offset + i]);
    }
  });
  // 返回数组 x 中第一个元素的累加和
  return vaddvq_f32(x[0]);
}

static C10_ALWAYS_INLINE void dot_with_fp32_arith_main_inner_loop(
  const float16_t* vec1,
  const float16_t* vec2,
  float32x4_t sum[kF32RegistersPerIteration],
  int registerPairIndex) {
  // 一次加载一对 f32 寄存器的数据
  const auto temp_vec1 = vld1q_f16(&vec1[registerPairIndex * 2 * kF32ElementsPerRegister]);
  const auto temp_vec2 = vld1q_f16(&vec2[registerPairIndex * 2 * kF32ElementsPerRegister]);

  // 在 sum 数组中执行低位和高位的 f16 乘加运算
  sum[2 * registerPairIndex] = f32_fma_low_f16(sum[2 * registerPairIndex], temp_vec1, temp_vec2);
  sum[2 * registerPairIndex + 1] = f32_fma_high_f16(sum[2 * registerPairIndex + 1], temp_vec1, temp_vec2);
}
// 对于FP32计算的点积，使用矢量化处理尾部的内部循环
static C10_ALWAYS_INLINE void dot_with_fp32_arith_vectorized_tail_inner_loop(
  const at::BFloat16* vec1,                       // 输入向量1，类型为BFloat16
  const at::BFloat16* vec2,                       // 输入向量2，类型为BFloat16
  float32x4_t* tailSum,                           // 指向尾部和的指针，类型为float32x4_t
  int idx) {                                      // 当前处理的索引位置
  const auto temp_vec1 = vld1_u16(reinterpret_cast<const uint16_t*>(&vec1[idx]));  // 加载向量1的一部分数据，转换为uint16_t类型
  const auto temp_vec2 = vld1_u16(reinterpret_cast<const uint16_t*>(&vec2[idx]));  // 加载向量2的一部分数据，转换为uint16_t类型
  *tailSum = f32_fma_bf16(*tailSum, temp_vec1, temp_vec2);  // 执行BF16的FMA操作，并将结果累加到tailSum中
}

// 将uint16x4_t类型数据转换为bfloat16类型的float32x4_t向量
static C10_ALWAYS_INLINE float32x4_t to_bfloat16(uint16x4_t u16) {
  int32x4_t shift = vdupq_n_s32(16);               // 创建一个全为16的int32x4_t向量，用于后续位移操作
  return vreinterpretq_f32_u32(vshlq_u32(vmovl_u16(u16), shift));  // 执行数据类型转换和位移操作，将uint16x4_t类型转换为bfloat16类型的float32x4_t向量
}

// 执行FP32 FMA操作，其中输入是float32x4_t和两个uint16x4_t向量
static C10_ALWAYS_INLINE float32x4_t f32_fma_bf16(float32x4_t a, uint16x4_t b, uint16x4_t c) {
  return f32_fma(a, to_bfloat16(b), to_bfloat16(c));  // 调用to_bfloat16将b和c转换为bfloat16类型，然后执行FP32的FMA操作
}

// 主循环的内部循环，执行FP32计算的主要部分，矢量化处理
static C10_ALWAYS_INLINE void dot_with_fp32_arith_main_inner_loop(
  const at::BFloat16* vec1,                       // 输入向量1，类型为BFloat16
  const at::BFloat16* vec2,                       // 输入向量2，类型为BFloat16
  float32x4_t sum[kF32RegistersPerIteration],      // 存储部分和的数组，类型为float32x4_t数组
  int registerPairIndex) {                        // 当前处理的寄存器对索引
  // TODO: detect intrinsic availability, use them if they're available. __ARM_FEATURE_BF16
  // 加载一对f32寄存器
  const uint16x8_t temp_vec1 = vld1q_u16(reinterpret_cast<const uint16_t*>(&vec1[registerPairIndex * 2 * kF32ElementsPerRegister]));  // 加载向量1的一对数据，转换为uint16x8_t类型
  const uint16x8_t temp_vec2 = vld1q_u16(reinterpret_cast<const uint16_t*>(&vec2[registerPairIndex * 2 * kF32ElementsPerRegister]));  // 加载向量2的一对数据，转换为uint16x8_t类型

  // 将加载的数据进行拆分，执行FP32 FMA操作，并将结果存储到sum数组中对应的位置
  sum[2 * registerPairIndex] = f32_fma_bf16(sum[2 * registerPairIndex], vget_low_u16(temp_vec1), vget_low_u16(temp_vec2));
  sum[2 * registerPairIndex + 1] = f32_fma_bf16(sum[2 * registerPairIndex + 1], vget_high_u16(temp_vec1), vget_high_u16(temp_vec2));
}

// 对FP32计算的点积进行处理，使用矢量化方法处理尾部数据
template <typename T>
float dot_with_fp32_arith(const T* vec1, const T* vec2, int64_t len) {
  float32x4_t sum[kF32RegistersPerIteration] = {vdupq_n_f32(0)};  // 初始化存储部分和的数组，全部为0
  const auto len_aligned = len & ~(kF32ElementsPerIteration - 1);  // 计算对齐的长度，按照kF32ElementsPerIteration进行对齐
  for (int j = 0; j < len_aligned ; j += kF32ElementsPerIteration) {
    const auto* vec1_ = vec1 + j;                 // 指向向量1的当前处理位置
    const auto* vec2_ = vec2 + j;                 // 指向向量2的当前处理位置
    // 使用模板展开执行循环，调用内部矢量化处理函数
    c10::ForcedUnroll<kF32RegisterPairsPerIteration>{}([vec1_, vec2_, &sum](auto k) {
      dot_with_fp32_arith_main_inner_loop(vec1_, vec2_, sum, k);  // 调用主循环的内部循环处理函数，执行FP32计算的主要部分
    });
  }
  auto reducedSum = reduce(sum);                    // 对部分和数组进行归约，得到最终的和

  // First-tier tail fixup: make sure we handle workloads that can
  // benefit from vectorization, but don't fit into our fully unrolled
  // loop above.
  float32x4_t tailSum = vdupq_n_f32(0);             // 初始化尾部和，全部为0
  const auto len_aligned_4 = len & ~3;              // 计算对齐的长度，按照4进行对齐
  for (int j = len_aligned; j < len_aligned_4; j += 4) {
    // 调用 dot_with_fp32_arith_vectorized_tail_inner_loop 函数处理向量 vec1 和 vec2 的部分计算，将结果存入 tailSum 中，并传递当前循环索引 j。
    dot_with_fp32_arith_vectorized_tail_inner_loop(vec1, vec2, &tailSum, j);
  }
  // 对尾部剩余元素进行累加操作，将累加结果存入 reducedTail
  auto reducedTail = vpaddq_f32(tailSum, tailSum);
  // 将 reducedTail 中的两个元素相加，将结果的第一个元素（即总和）添加到 reducedSum 中
  reducedSum += vgetq_lane_f32(vpaddq_f32(reducedTail, reducedTail), 0);

  // 处理剩余的未对齐的元素，通过简单的循环累加操作计算向量 vec1 和 vec2 的点积
  for (int j = len_aligned_4; j < len; ++j) {
    reducedSum += vec1[j] * vec2[j];
  }
  // 返回计算结果 reducedSum 作为点积的最终值
  return reducedSum;
}

// 使用 float16_t 类型的向量 vec1 和 vec2 进行点积计算，返回结果
float fp16_dot_with_fp32_arith(const float16_t* vec1, const float16_t* vec2, int64_t len) {
  // 调用 dot_with_fp32_arith 函数计算向量 vec1 和 vec2 的点积
  return dot_with_fp32_arith(vec1, vec2, len);
}

// 使用 at::BFloat16 类型的向量 vec1 和 vec2 进行点积计算，返回结果
float bf16_dot_with_fp32_arith(const at::BFloat16* vec1, const at::BFloat16* vec2, int64_t len) {
  // 调用 dot_with_fp32_arith 函数计算向量 vec1 和 vec2 的点积
  return dot_with_fp32_arith(vec1, vec2, len);
}

// 在 Apple M1 MacBook 上，使用 fp16_gemv_trans_fp32_arith_by_dot_products 函数执行 FP16 到 FP32 算术的转置矩阵向量乘法
static void fp16_gemv_trans_fp32_arith_by_dot_products(const int m, const int n, const float16_t* a, const int lda, const float16_t *x, float16_t* y, int incy) {
  // 并行计算 for 循环，遍历从 begin 到 end 的 n 值
  parallel_for(0, n, 1, [&](int begin, int end) {
    for (int i = begin; i < end; ++i) {
      // 调用 fp16_dot_with_fp32_arith 函数计算 x 和 a 的乘积，存储结果到 y[i * incy] 中
      y[i * incy] = fp16_dot_with_fp32_arith(x, a + lda * i, m);
    }
  });
}

// 在 Apple M1 MacBook 上，使用 bf16_gemv_trans_fp32_arith_by_dot_products 函数执行 BF16 到 FP32 算术的转置矩阵向量乘法
static void bf16_gemv_trans_fp32_arith_by_dot_products(const int m, const int n, const at::BFloat16* a, const int lda, const at::BFloat16 *x, at::BFloat16* y, int incy) {
  // 并行计算 for 循环，遍历从 begin 到 end 的 n 值
  parallel_for(0, n, 1, [&](int begin, int end) {
    for (int i = begin; i < end; ++i) {
      // 调用 bf16_dot_with_fp32_arith 函数计算 x 和 a 的乘积，存储结果到 y[i * incy] 中
      y[i * incy] = bf16_dot_with_fp32_arith(x, a + lda * i, m);
    }
  });
}

// 执行 FP16 矩阵向量乘法的函数，要求 alpha 为 1.0，beta 为 0.0，incx 为 1
void fp16_gemv_trans(
    const int m,
    const int n,
    const float alpha,
    const float16_t* a,
    const int lda,
    const float16_t* x,
    const int incx,
    const float beta,
    float16_t* y,
    const int incy) {
  // 断言检查，确保 incx 为 1，alpha 为 1.0，beta 为 0.0
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(incx == 1 && alpha == 1.0 && beta == 0.0);
#ifdef __ARM_FEATURE_FP16_SCALAR_ARITHMETIC
  // 如果硬件支持 FP16 标量运算且允许在 CPU 上进行 FP16 归约
  if (at::globalContext().allowFP16ReductionCPU()) {
    // 调用 fp16_gemv_trans_fp16_arith_by_dot_products 函数执行 FP16 到 FP16 算术的转置矩阵向量乘法
    return fp16_gemv_trans_fp16_arith_by_dot_products(m, n, a, lda, x, y, incy);
  }
#endif
  // 否则调用 fp16_gemv_trans_fp32_arith_by_dot_products 函数执行 FP16 到 FP32 算术的转置矩阵向量乘法
  return fp16_gemv_trans_fp32_arith_by_dot_products(m, n, a, lda, x, y, incy);
}

// 执行 BF16 矩阵向量乘法的函数，要求 alpha 为 1.0，beta 为 0.0，incx 为 1
void bf16_gemv_trans(
  const int m,
  const int n,
  const at::BFloat16 alpha,
  const at::BFloat16* a,
  const int lda,
  const at::BFloat16* x,
  const int incx,
  const at::BFloat16 beta,
  at::BFloat16* y,
  const int incy) {
  // 断言检查，确保 incx 为 1，alpha 为 1.0，beta 为 0.0
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(incx == 1 && alpha == 1.0 && beta == 0.0);
  // 调用 bf16_gemv_trans_fp32_arith_by_dot_products 函数执行 BF16 到 FP32 算术的转置矩阵向量乘法
  return bf16_gemv_trans_fp32_arith_by_dot_products(m, n, a, lda, x, y, incy);
}

#ifdef __ARM_FEATURE_FP16_SCALAR_ARITHMETIC
// 在支持 FP16 标量算术的 ARM 平台上，执行 FP16 矩阵向量乘法的函数
static void fp16_gemv_notrans_fp16_arith(int m, int n, const float16_t* a, const int lda, const float16_t *x, float16_t *y) {
  // 遍历矩阵的列
  for (auto j = 0; j < n; j++) {
    // 将向量 x[j] 的值扩展成 SIMD 向量
    auto vecCol = vdup_n_f16(x[j]);
    // 获取矩阵的第 j 列指针
    const auto* column = a + lda * j;
    // 遍历矩阵的行，每次处理 4 行
    for (auto i = 0; i < m; i += 4) {
      auto yf16 = y + i;
      // 加载矩阵的一行数据
      auto matRow = vld1_f16(column + i);
      // 如果不是第一次迭代，则从 yf16 加载之前的结果向量
      auto resVec = j != 0 ? vld1_f16(yf16) : vdup_n_f16(0);
      // 使用 SIMD 指令计算乘积并累加到 resVec
      resVec = vfma_lane_f16(resVec, matRow, vecCol, 0);
      // 将结果向量存回 yf16
      vst1_f16(yf16, resVec);
    }
  }
}
#endif

// 执行 FP16 矩阵向量乘法的函数，使用 FP32 算术
static void fp16_gemv_notrans_fp32_arith(int m, int n, const float16_t* a, const int lda, const float16_t *x, float16_t *y) {
  // 创建一个大小为 m 的 float 数组来存储累加和
  std::vector<float> sum(m);
  // 遍历矩阵的列
  for (auto j = 0; j < n; j++) {
    // 将向量 x[j] 的值扩展成 SIMD 向量
    auto vecCol = vdup_n_f32(x[j]);
    // 获取矩阵的第 j 列指针
    const auto* column = a + lda * j;
    // 遍历矩阵的行
    for (auto i = 0; i < m; i += 4) {
      auto yf16 = y + i;
      // 加载矩阵的一行数据
      auto matRow = vld1_f16(column + i);
      // 如果不是第一次迭代，则从 yf16 加载之前的结果向量
      auto resVec = j != 0 ? vld1_f16(yf16) : vdup_n_f16(0);
      // 使用 SIMD
    // 对于每个循环步长为4的i，执行以下操作
    for (auto i = 0; i < m; i += 4) {
      // sf32 指向 sum.data() 数组的第 i 个元素
      auto sf32 = sum.data() + i;
      // matRow 是从 column 数组的第 i 个位置开始，转换为单精度浮点数的结果
      auto matRow = vcvt_f32_f16(vld1_f16(column + i));
      // 如果 j 不等于0，resVec 取 sf32 所指向的值；否则，resVec 被初始化为0
      auto resVec = j != 0 ? vld1q_f32(sf32) : vdupq_n_f32(0);
      // 将 matRow 与 vecCol 的第0个元素进行乘法累加到 resVec 中
      resVec = vfmaq_lane_f32(resVec, matRow, vecCol, 0);
      // 将 resVec 中的值存储回 sf32 所指向的位置
      vst1q_f32(sf32, resVec);
    }
  }

  // 将 sum.data() 中每4个元素依次转换为半精度浮点数，存储到 y 数组中的对应位置
  for (auto i = 0; i < m; i += 4) {
    vst1_f16(y + i, vcvt_f16_f32(vld1q_f32(sum.data() + i)));
  }
} // 结束命名空间 blas_impl

// 定义模板函数 scal，用于向量的缩放操作
template <typename scalar_t>
inline void scal(int64_t n, scalar_t a, scalar_t *x, int64_t incx)
{
  // 如果 n 等于 1，则强制设置 incx 为 1，以确保只操作单个元素
  if (n == 1) incx = 1;

  // 如果使用了 BLAS 库且满足条件，调用快速路径优化函数
  if (blas_impl::scal_use_fast_path<scalar_t>(n, incx)) {
    // 将 n 和 incx 转换为整数
    int i_n = (int)n;
    int i_incx = (int)incx;
    # 调用 BLAS 库中的快速缩放函数 `scal_fast_path`
    blas_impl::scal_fast_path<scalar_t>(&i_n, &a, x, &i_incx);
    # 直接返回，结束函数执行
    return;
#endif
  // 对于每个索引 i 在范围 [0, n) 内循环
  for (const auto i : c10::irange(n)) {
    // 如果 alpha 等于 0，则将 x[i * incx] 设置为 0
    if (a == scalar_t(0)) {
      x[i * incx] = 0;
    } else {
      // 否则，将 x[i * incx] 乘以 alpha
      x[i * incx] *= a;
    }
  }
}

template<typename scalar_t>
void gemv(char trans, int64_t m, int64_t n, scalar_t alpha, const scalar_t *a, int64_t lda, const scalar_t *x, int64_t incx, scalar_t beta, scalar_t *y, int64_t incy) {
  // 当 n 等于 1 时，强制将 lda 设为 m
  if(n == 1) lda = m;

#if AT_BUILD_WITH_BLAS()
  // 如果可以使用快速路径优化 gemv 操作
  if (blas_impl::gemv_use_fast_path<scalar_t>(trans, m, n, alpha, lda, incx, beta, incy)) {
    // 检查 lda 是否大于等于 max(1, m)，否则抛出异常
    TORCH_CHECK(lda >= std::max<int64_t>(1L, m), "lda should be at least max(1,", m, "), but have ", lda);
    // 将 m、n、lda、incx、incy 转换为整数类型
    int i_m = (int)m;
    int i_n = (int)n;
    int i_lda = (int)lda;
    int i_incx = (int)incx;
    int i_incy = (int)incy;
    // 调用快速路径的 gemv 实现
    blas_impl::gemv_fast_path<scalar_t>(&trans, &i_m, &i_n, &alpha, a, &i_lda, x, &i_incx, &beta, y, &i_incy);
    // 直接返回
    return;
  }
#endif

  // 定义 opmath_t 类型作为 scalar_t 的运算类型
  using opmath_t = at::opmath_type<scalar_t>;
  // 如果 trans 是 'T' 或 't'，表示需要进行转置操作
  if ((trans == 'T') || (trans == 't')) {
    // 对于每个索引 i 在范围 [0, n) 内循环
    for (const auto i : c10::irange(n)) {
      // 初始化 sum 为 0
      opmath_t sum = 0;
      // 获取 a 的第 i 列起始地址
      const scalar_t *row_ = a + lda * i;
      // 对于每个索引 j 在范围 [0, m) 内循环
      for (const auto j : c10::irange(m)) {
        // 计算 alpha * x[j * incx] * row_[j] 的累加和
        sum += x[j * incx] * row_[j];
      }
      // 根据 beta 的值更新 y[i * incy]
      if (beta == scalar_t(0)) {
        y[i * incy] = alpha * sum;
      } else {
        y[i * incy] = beta * y[i * incy] + alpha * sum;
      }
    }
  } else {
    // 如果 beta 不等于 1 且不等于 0，则调用 scal 函数将 y 向量乘以 beta
    if (beta != scalar_t(1) && beta != scalar_t(0)) scal<scalar_t>(m, beta, y, incy);

    // 检查是否是低精度计算
    constexpr bool is_low_precision = !std::is_same_v<opmath_t, scalar_t>;
    std::vector<opmath_t> sum;
    // 如果是低精度计算，需要初始化 sum 向量
    if constexpr (is_low_precision) {
      sum.resize(m);
    }
    // 对于每个索引 j 在范围 [0, n) 内循环
    for (const auto j : c10::irange(n)) {
      // 获取 a 的第 j 列起始地址
      const scalar_t *column_ = a + lda * j;
      // 计算 alpha * x[j * incx]，并转换为 opmath_t 类型
      opmath_t z = alpha * static_cast<opmath_t>(x[j * incx]);
      // 对于每个索引 i 在范围 [0, m) 内循环
      for (const auto i : c10::irange(m)) {
        // 当 j 为 0 且 beta 为 0 时，将 y[i * incy] 设置为 0
        if (j==0 && beta==scalar_t(0)) {
          if constexpr (!is_low_precision) {
            y[i * incy] = 0;
          }
        }
        // 根据 is_low_precision 的不同情况累加乘积结果到 y[i * incy] 或 sum[i]
        if constexpr (is_low_precision) {
          sum[i] += z * column_[i];
        } else {
          y[i * incy] += z * column_[i];
        }
      }
    }
    // 如果是低精度计算，根据 beta 的值更新 y 向量
    if constexpr (is_low_precision) {
      if (beta == scalar_t(0)) {
        for (const auto i : c10::irange(m)) {
          y[i * incy] = sum[i];
        }
      } else {
        for (const auto i : c10::irange(m)) {
          y[i * incy] += sum[i];
        }
      }
    }
  }
  // 函数返回
  return;
}

// 实例化 gemv 函数模板，针对各种标量类型
#define INSTANTIATE(scalar_t, _) \
template void gemv<scalar_t>(char trans, int64_t m, int64_t n, scalar_t alpha, const scalar_t *a, int64_t lda, const scalar_t *x, int64_t incx, scalar_t beta, scalar_t *y, int64_t incy);
AT_FORALL_SCALAR_TYPES_AND2(BFloat16, Half, INSTANTIATE);
AT_FORALL_COMPLEX_TYPES(INSTANTIATE);
#undef INSTANTIATE

namespace blas_impl {
#if AT_BUILD_WITH_BLAS()
// 定义静态函数 dot_fast_path，计算两个 float 数组的点积，利用 BLAS 库中的 sdot_
static float dot_fast_path(int n, float* x, int incx, float* y, int incy) {
  // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
  return sdot_(&n, x, &incx, y, &incy);
}

// 定义静态函数 dot_fast_path，计算两个 double 数组的点积，利用 BLAS 库中的 ddot_
static double dot_fast_path(int n, double* x, int incx, double* y, int incy) {
  return ddot_(&n, x, &incx, y, &incy);
}

// 定义静态函数 vdot_fast_path，计算两个复数（float 类型）数组的点积，利用 BLAS 库中的 cdotc_
static c10::complex<float> vdot_fast_path(int n, c10::complex<float>* x, int incx, c10::complex<float>* y, int incy) {
  c10::complex<float> result;
  cdotc_(reinterpret_cast<std::complex<float>* >(&result), &n, reinterpret_cast<std::complex<float>*>(x), &incx, reinterpret_cast<std::complex<float>*>(y), &incy);
  return result;
}

// 定义静态函数 vdot_fast_path，计算两个复数（double 类型）数组的点积，利用 BLAS 库中的 zdotc_
static c10::complex<double> vdot_fast_path(int n, c10::complex<double>* x, int incx, c10::complex<double>* y, int incy) {
  c10::complex<double> result;
  zdotc_(reinterpret_cast<std::complex<double>* >(&result), &n, reinterpret_cast<std::complex<double>*>(x), &incx, reinterpret_cast<std::complex<double>*>(y), &incy);
  return result;
}

// 定义静态函数 dot_fast_path，计算两个复数（double 类型）数组的点积，利用 BLAS 库中的 zdotu_
static c10::complex<double> dot_fast_path(int n, c10::complex<double>* x, int incx, c10::complex<double>* y, int incy) {
  c10::complex<double> result;
  zdotu_(reinterpret_cast<std::complex<double>* >(&result), &n, reinterpret_cast<std::complex<double>*>(x), &incx, reinterpret_cast<std::complex<double>*>(y), &incy);
  return result;
}

// 定义静态函数 dot_fast_path，计算两个复数（float 类型）数组的点积，利用 BLAS 库中的 cdotu_
static c10::complex<float> dot_fast_path(int n, c10::complex<float>* x, int incx, c10::complex<float>* y, int incy) {
  c10::complex<float> result;
  cdotu_(reinterpret_cast<std::complex<float>* >(&result), &n, reinterpret_cast<std::complex<float>*>(x), &incx, reinterpret_cast<std::complex<float>*>(y), &incy);
  return result;
}

// 结束 blas_impl 命名空间
} // namespace blas_impl

// 定义 dot_naive 函数模板，计算一般情况下的点积，使用自定义的 Functor
template <typename scalar_t, typename Functor>
scalar_t dot_naive(
    int64_t n,
    scalar_t* x,
    int64_t incx,
    scalar_t* y,
    int64_t incy,
    Functor op) {
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  int64_t i;
  // 定义 opmath_t 为 scalar_t 对应的操作数类型
  using opmath_t = at::opmath_type<scalar_t>;
  // 初始化 sum 为零
  opmath_t sum = 0;
  // 循环计算点积
  for (i = 0; i < n; i++) {
    // 使用 Functor op 计算 x 和 y 数组对应位置元素的运算结果并累加到 sum 中
    sum += op(static_cast<opmath_t>(x[i * incx]), static_cast<opmath_t>(y[i * incy]));
  }
  // 将 sum 转换为 scalar_t 类型并返回
  return static_cast<scalar_t>(sum);
}

// 结束命名空间 blas_impl

// 定义 dot_impl_floating 函数模板，根据模板参数 scalar_t，调用 dot_fast_path 或 dot_naive
template <typename scalar_t>
scalar_t dot_impl_floating(int64_t n, scalar_t* x, int64_t incx, scalar_t* y, int64_t incy)
{
  // 如果 n 等于 1，重置 incx 和 incy 为 1
  if (n == 1) {
    incx = 1;
    incy = 1;
  }
  // 如果编译时开启了 BLAS 支持，且 n、incx、incy 均不超过 INT_MAX
  if ((n <= INT_MAX) && (incx <= INT_MAX) && (incy <= INT_MAX)) {
    // 调用 blas_impl 命名空间中的 dot_fast_path 函数，计算点积并返回结果
    return blas_impl::dot_fast_path(n, x, incx, y, incy);
  } else {
    // 否则调用 dot_naive 函数，使用 std::multiplies<scalar_t> 作为 Functor，计算点积并返回结果
    return blas_impl::dot_naive(n, x, incx, y, incy, std::multiplies<scalar_t>{});
  }
}

// 如果未开启 BLAS 支持，直接调用 dot_naive 函数，使用 std::multiplies<scalar_t> 作为 Functor，计算点积并返回结果
template <typename scalar_t>
scalar_t dot_impl(int64_t n, scalar_t* x, int64_t incx, scalar_t* y, int64_t incy) {
  // 如果 n 等于 1，重置 incx 和 incy 为 1
  if (n == 1) {
    incx = 1;
    incy = 1;
  }
  return blas_impl::dot_naive(n, x, incx, y, incy, std::multiplies<scalar_t>{});
}

// 结束 dot_impl 命名空间
template <>
// 调用 dot_impl_floating 函数计算浮点数类型的向量点积并返回结果
float dot_impl(int64_t n, float* x, int64_t incx, float* y, int64_t incy) {
  return dot_impl_floating(n, x, incx, y, incy);
}

// 使用 dot_impl_floating 函数计算双精度浮点数类型的向量点积并返回结果
template <>
double dot_impl(int64_t n, double* x, int64_t incx, double* y, int64_t incy) {
  return dot_impl_floating(n, x, incx, y, incy);
}

// 使用 dot_impl_floating 函数计算双精度复数类型的向量点积并返回结果
template <>
c10::complex<double> dot_impl(int64_t n, c10::complex<double>* x, int64_t incx, c10::complex<double>* y, int64_t incy) {
  return dot_impl_floating(n, x, incx, y, incy);
}

// 使用 dot_impl_floating 函数计算单精度复数类型的向量点积并返回结果
template <>
c10::complex<float> dot_impl(int64_t n, c10::complex<float>* x, int64_t incx, c10::complex<float>* y, int64_t incy) {
  return dot_impl_floating(n, x, incx, y, incy);
}

namespace {
// 定义结构体 vdot_op，用于计算标量 x 和 y 的共轭乘积
template <typename scalar_t>
struct vdot_op {
  scalar_t operator()(scalar_t x, scalar_t y) {
    return std::conj(x) * y;
  }
};
} // anonymous namespace

// 使用 blas_impl 命名空间中的优化 dot 函数或者基本 dot 函数计算向量 x 和 y 的点积并返回结果
template <typename scalar_t>
scalar_t vdot_impl(int64_t n, scalar_t* x, int64_t incx, scalar_t* y, int64_t incy) {
  // 如果 n 等于 1，设置 incx 和 incy 为 1
  if (n == 1) {
    incx = 1;
    incy = 1;
  }
  // 如果编译配置支持 BLAS，且 n、incx、incy 均不超过 INT_MAX，则调用 blas_impl 命名空间中的 vdot_fast_path 函数
  // 否则调用 dot_naive 函数计算点积，并使用 vdot_op<scalar_t> 结构体计算复数的共轭乘积
#if AT_BUILD_WITH_BLAS()
        if ((n <= INT_MAX) && (incx <= INT_MAX) && (incy <= INT_MAX)) {
          return blas_impl::vdot_fast_path(n, x, incx, y, incy);
        } else {
          return blas_impl::dot_naive(n, x, incx, y, incy, vdot_op<scalar_t>{});
        }
#else
        { return blas_impl::dot_naive(n, x, incx, y, incy, vdot_op<scalar_t>{}); }
#endif
}

// 实例化模板函数 dot_impl，用于指定不同类型的标量
#define INSTANTIATE_DOT_IMPL(scalar_t)  \
  template scalar_t dot_impl<scalar_t>( \
      int64_t n, scalar_t * x, int64_t incx, scalar_t * y, int64_t incy);
INSTANTIATE_DOT_IMPL(uint8_t);
INSTANTIATE_DOT_IMPL(int8_t);
INSTANTIATE_DOT_IMPL(int16_t);
INSTANTIATE_DOT_IMPL(int);
INSTANTIATE_DOT_IMPL(int64_t);
INSTANTIATE_DOT_IMPL(c10::Half);
INSTANTIATE_DOT_IMPL(c10::BFloat16);

// 实例化模板函数 vdot_impl，用于指定不同类型的标量
#define INSTANTIATE_VDOT_IMPL(scalar_t)  \
  template scalar_t vdot_impl<scalar_t>( \
      int64_t n, scalar_t * x, int64_t incx, scalar_t * y, int64_t incy);
INSTANTIATE_VDOT_IMPL(c10::complex<float>);
INSTANTIATE_VDOT_IMPL(c10::complex<double>);

// 取消定义 INSTANTIATE_DOT_IMPL 宏
#undef INSTANTIATE_DOT_IMPL

} // namespace at::native

// 恢复先前的诊断设置
C10_DIAGNOSTIC_POP()
```