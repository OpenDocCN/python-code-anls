# `.\pytorch\aten\src\ATen\native\cpu\BlasKernel.cpp`

```py
// 定义 TORCH_ASSERT_NO_OPERATORS 以避免引入操作符
#define TORCH_ASSERT_NO_OPERATORS
// 包含 ATen 库的分发和并行处理相关头文件
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
// 包含 ATen 库中与 CPU BLAS 相关的头文件
#include <ATen/native/CPUBlas.h>
#include <ATen/native/cpu/zmath.h>
// 包含 c10 实用工具中的 irange 和 Unroll 相关头文件
#include <c10/util/irange.h>
#include <c10/util/Unroll.h>

// 如果目标架构是 ARM 64 且不是移动设备上，包含 NEON SIMD 指令集的头文件
#if defined(__aarch64__) && !defined(C10_MOBILE)
#include <arm_neon.h>

namespace at::native::blas_impl {
// 声明用于半精度矩阵向量乘法的函数原型
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

// 声明混合精度计算中的半精度浮点数与单精度浮点数乘法的函数原型
float fp16_dot_with_fp32_arith(
  const float16_t* x,
  const float16_t* a,
  int64_t len);

// 声明混合精度计算中的 BF16 与单精度浮点数乘法的函数原型
float bf16_dot_with_fp32_arith(
  const at::BFloat16* x,
  const at::BFloat16* a,
  int64_t len);
}
#endif

// 定义 ATen 库中的 CPU BLAS 命名空间
namespace at::native::cpublas {
namespace {

// 模板函数，用于对矩阵进行缩放操作
template <typename scalar_t, typename opmath_t>
void scale_(int64_t m, int64_t n, opmath_t alpha, scalar_t *a, int64_t lda) {
  // 如果 alpha 等于 1，返回，表示进行了单位操作
  if (alpha == opmath_t(1)) {
    return;  // identity
  }

  // 如果 alpha 等于 0，将矩阵 a 中的所有元素置为 0
  if (alpha == opmath_t(0)) {
    for (const auto j : c10::irange(n)) {
      for (const auto i : c10::irange(m)) {
        a[j * lda + i] = scalar_t(0);
      }
    }
    return;
  }

  // 对矩阵 a 中的每个元素乘以 alpha
  for (const auto j : c10::irange(n)) {
    for (const auto i : c10::irange(m)) {
      a[j * lda + i] *= alpha;
    }
  }
}

// 模板函数，计算给定函数在 [0, N) 范围内的和
template <typename Func>
auto sum(int64_t N, Func f) {
  constexpr int ilp_factor = 4;
  using acc_t = decltype(f(0));

  // 计算独立的部分和，然后在最后将它们相加
  std::array<acc_t, ilp_factor> partial_sums{};

  int64_t i = 0;
  for (; i + ilp_factor <= N; i += ilp_factor) {
    c10::ForcedUnroll<ilp_factor>{}([&](int k) {
      partial_sums[k] += f(i + k);
    });
  }
  for (; i < N; ++i) {
    partial_sums[0] += f(i);
  }
  for (int k = 1; k < ilp_factor; ++k) {
    partial_sums[0] += partial_sums[k];
  }
  return partial_sums[0];
}

// 模板函数，用于执行未转置矩阵乘法操作
template <typename scalar_t, typename opmath_t>
typename std::enable_if<std::is_same<scalar_t, opmath_t>::value, void>::type
gemm_notrans_(
    int64_t m,
    int64_t n,
    int64_t k,
    opmath_t alpha,
    const scalar_t* a,
    int64_t lda,
    const scalar_t* b,
    int64_t ldb,
    opmath_t beta,
    scalar_t* c,
    int64_t ldc) {
  // 将矩阵 c 中的元素乘以 beta
  scale_(m, n, beta, c, ldc);

  // 计算 c += alpha * (a @ b)，其中 @ 表示矩阵乘法
  for (const auto l : c10::irange(k)) {
    // 对于给定的范围 n，遍历 j 的所有值
    for (const auto j : c10::irange(n)) {
      // 计算矩阵乘法中的一个元素，使用 b[l + j * ldb] 乘以 alpha
      opmath_t val = b[l + j * ldb] * alpha;
      // 计算每个 j 对应的子矩阵中 i 的范围，每个子矩阵大小为 4 行
      int64_t i_m = m / 4;
      // 对于每个子矩阵，遍历 i_i 的所有值
      for (const auto i_i : c10::irange(i_m)) {
        // 更新 C 矩阵中的元素，根据 A 和 B 矩阵的乘积和乘以 val
        c[j * ldc + i_i * 4 + 0] += a[i_i * 4 + 0 + l * lda] * val;
        c[j * ldc + i_i * 4 + 1] += a[i_i * 4 + 1 + l * lda] * val;
        c[j * ldc + i_i * 4 + 2] += a[i_i * 4 + 2 + l * lda] * val;
        c[j * ldc + i_i * 4 + 3] += a[i_i * 4 + 3 + l * lda] * val;
      }
      // 处理剩余的不完整子矩阵部分，每个子矩阵大小为 4 行
      int64_t i = i_m * 4;
      for (; i < m; i++)
        // 更新 C 矩阵中的元素，根据 A 和 B 矩阵的乘积和乘以 val
        c[j * ldc + i] += a[i + l * lda] * val;
    }
}

// 检查是否是标量类型为 at::BFloat16 或 at::Half，如果不是则执行矩阵乘法运算
template <typename scalar_t, typename opmath_t>
typename std::enable_if<!std::is_same<scalar_t, opmath_t>::value, void>::type
gemm_notrans_(
    int64_t m,                             // 矩阵 C 的行数
    int64_t n,                             // 矩阵 C 的列数
    int64_t k,                             // 矩阵 A 的列数（同时也是矩阵 B 的行数）
    opmath_t alpha,                        // 乘法操作的标量因子
    const scalar_t* a,                     // 矩阵 A 的数据指针
    int64_t lda,                           // 矩阵 A 的列跨度
    const scalar_t* b,                     // 矩阵 B 的数据指针
    int64_t ldb,                           // 矩阵 B 的列跨度
    opmath_t beta,                         // 乘法结果与矩阵 C 的缩放因子
    scalar_t* c,                           // 矩阵 C 的数据指针
    int64_t ldc) {                         // 矩阵 C 的列跨度
  // 对矩阵 C 进行乘法操作：c += alpha * (a @ b)
  for (const auto i : c10::irange(m)) {    // 遍历矩阵 C 的行索引
    for (const auto j : c10::irange(n)) {  // 遍历矩阵 C 的列索引
      const auto dot = sum(k, [&](int64_t l) -> opmath_t {  // 计算矩阵乘法的内积 dot product
        return static_cast<opmath_t>(a[l * lda + i]) *
            static_cast<opmath_t>(b[j * ldb + l]);
      });
      if (beta == opmath_t(0)) {           // 如果 beta 等于 0，则直接赋值
        c[j * ldc + i] = alpha * dot;
      } else {                             // 否则，进行缩放并加到现有值上
        c[j * ldc + i] = beta * c[j * ldc + i] + alpha * dot;
      }
    }
  }
}

template <typename scalar_t, typename opmath_t>
void gemm_transa_(
    TransposeType transa,                  // 矩阵 A 的转置类型
    int64_t m, int64_t n, int64_t k,       // 矩阵 C 的行数、列数以及矩阵 A 和 B 的维度
    opmath_t alpha,                        // 乘法操作的标量因子
    const scalar_t *a, int64_t lda,        // 矩阵 A 的数据指针和列跨度
    const scalar_t *b, int64_t ldb,        // 矩阵 B 的数据指针和列跨度
    opmath_t beta,                         // 乘法结果与矩阵 C 的缩放因子
    scalar_t *c,                           // 矩阵 C 的数据指针
    int64_t ldc) {                         // 矩阵 C 的列跨度
  // 执行乘法操作：c = alpha * (a.T @ b) + beta * c
  const scalar_t *a_ = a;                  // 指向矩阵 A 数据的指针
  for (const auto i : c10::irange(m)) {    // 遍历矩阵 C 的行索引
    const scalar_t *b_ = b;                // 指向矩阵 B 数据的指针
    for (const auto j : c10::irange(n)) {  // 遍历矩阵 C 的列索引
      const auto dot = sum(k, [&](int64_t l) -> opmath_t {  // 计算矩阵乘法的内积 dot product
        return static_cast<opmath_t>(transa == TransposeType::ConjTranspose ? conj_impl(a_[l]) : a_[l]) * static_cast<opmath_t>(b_[l]);
      });
      b_ += ldb;                           // 更新矩阵 B 数据的指针
      if (beta == opmath_t(0)) {           // 如果 beta 等于 0，则直接赋值
        c[j * ldc + i] = alpha * dot;
      } else {                             // 否则，进行缩放并加到现有值上
        c[j * ldc + i] = beta * c[j * ldc + i] + alpha * dot;
      }
    }
    a_ += lda;                             // 更新矩阵 A 数据的指针
  }
}

template <typename scalar_t, typename opmath_t>
void gemm_transb_impl(
    TransposeType transb,                  // 矩阵 B 的转置类型
    int64_t m, int64_t n, int64_t k,       // 矩阵 C 的行数、列数以及矩阵 A 和 B 的维度
    opmath_t alpha,                        // 乘法操作的标量因子
    const scalar_t* a, int64_t lda,        // 矩阵 A 的数据指针和列跨度
    const scalar_t* b, int64_t ldb,        // 矩阵 B 的数据指针和列跨度
    /* we expect pre-applied beta */
    opmath_t* c,                           // 矩阵 C 的数据指针
    int64_t ldc) {                         // 矩阵 C 的列跨度
  // 执行乘法操作：c += alpha * (a @ b.T)
  for (const auto l : c10::irange(k)) {    // 遍历矩阵 A 和 B 的维度
    for (const auto j : c10::irange(n)) {  // 遍历矩阵 C 的列索引
      opmath_t val = (transb == TransposeType::ConjTranspose ? conj_impl(b[j + l * ldb]) : b[j + l * ldb]) * alpha;  // 计算乘法操作的值
      int64_t i_m = m / 4;                 // 计算矩阵 C 行数的四分之一
      for (const auto i_i : c10::irange(i_m)) {  // 遍历矩阵 C 行的索引
        c[j * ldc + i_i * 4 + 0] += a[i_i * 4 + 0 + l * lda] * val;  // 更新矩阵 C 的值
        c[j * ldc + i_i * 4 + 1] += a[i_i * 4 + 1 + l * lda] * val;  // 更新矩阵 C 的值
        c[j * ldc + i_i * 4 + 2] += a[i_i * 4 + 2 + l * lda] * val;  // 更新矩阵 C 的值
        c[j * ldc + i_i * 4 + 3] += a[i_i * 4 + 3 + l * lda] * val;  // 更新矩阵 C 的值
      }
      int64_t i = i_m * 4;                 // 计算未处理的行索引
      for (; i < m; i++)                   // 遍历矩阵 C 的行索引
        c[j * ldc + i] += a[i + l * lda] * val;  // 更新矩阵 C 的值
    }
  }
}

template <typename scalar_t, typename opmath_t>
typename std::enable_if<std::is_same<scalar_t, opmath_t>::value, void>::type
gemm_transb_(
    TransposeType transb,                  // 矩阵 B 的转置类型
    int64_t m, int64_t n, int64_t k,       // 矩阵 C 的行数、列数以及矩阵 A 和 B 的维度
    opmath_t alpha,                        // 乘法操作的标量因子
    const scalar_t* a, int64_t lda,        // 矩阵 A 的数据指针和列跨度
    const scalar_t* b,
    int64_t ldb,
    opmath_t beta,
    scalar_t* c,
    int64_t ldc) {
  // 接收指向矩阵 b 的常量指针
  // ldb 是矩阵 b 的列数
  // beta 是一个数学运算类型的参数，用于乘法运算
  // c 是指向结果矩阵的指针
  // ldc 是结果矩阵 c 的列数

  // 调用 scale_ 函数对结果矩阵 c 进行缩放，乘以 beta
  scale_(m, n, beta, c, ldc);

  // 调用 gemm_transb_impl 函数执行矩阵乘法运算，其中 transb 参数指示是否对矩阵 b 进行转置
  gemm_transb_impl(transb, m, n, k, alpha, a, lda, b, ldb, c, ldc);
// 结束前一个模板函数定义的大括号

// 如果 scalar_t 不是 opmath_t 类型，那么启用此模板函数
template <typename scalar_t, typename opmath_t>
typename std::enable_if<!std::is_same<scalar_t, opmath_t>::value, void>::type
gemm_transb_(
    TransposeType transb,        // 表示矩阵 B 是否需要转置的枚举值
    int64_t m,                   // 矩阵 C 的行数
    int64_t n,                   // 矩阵 C 的列数
    int64_t k,                   // 矩阵 A 的列数（如果不转置）或行数（如果转置）
    opmath_t alpha,              // 乘法运算的 alpha 系数
    const scalar_t* a,           // 矩阵 A 的数据指针
    int64_t lda,                 // 矩阵 A 的 leading dimension
    const scalar_t* b,           // 矩阵 B 的数据指针
    int64_t ldb,                 // 矩阵 B 的 leading dimension
    opmath_t beta,               // 乘法运算的 beta 系数
    scalar_t* c,                 // 矩阵 C 的数据指针
    int64_t ldc) {               // 矩阵 C 的 leading dimension

  // 为了保证正确性，我们需要使用全精度的点积运算；
  // 用户注意到使用降低宽度类型（例如 float16/bfloat16）可能导致误差累积，
  // 这会影响性能。因此，我们选择分配临时空间，以高精度保存结果，
  // 然后使用缓存友好的算法进行累加。
  const auto c_size = m * n;
  auto c_accum = std::make_unique<opmath_t[]>(c_size);

  // 根据 beta 的值初始化 c_accum
  if (beta == 1) {
    for (const auto j : c10::irange(n)) {
      for (const auto i : c10::irange(m)) {
        c_accum[j * m + i] = c[j * ldc + i];
      }
    }
  } else if (beta == 0) {
    for (const auto j : c10::irange(n)) {
      for (const auto i : c10::irange(m)) {
        c_accum[j * m + i] = 0;
      }
    }
  } else {
    for (const auto j : c10::irange(n)) {
      for (const auto i : c10::irange(m)) {
        c_accum[j * m + i] = beta * c[j * ldc + i];
      }
    }
  }

  // 调用 gemm_transb_impl 函数进行矩阵乘法运算
  gemm_transb_impl(transb, m, n, k, alpha, a, lda, b, ldb, c_accum.get(), m);

  // 将累加结果从 c_accum 复制回矩阵 C 中
  for (const auto j : c10::irange(n)) {
    for (const auto i : c10::irange(m)) {
      c[j * ldc + i] = c_accum[j * m + i];
    }
  }
}

// 矩阵乘法函数模板，支持不同的转置方式
template <typename scalar_t, typename opmath_t>
void gemm_transab_(
    TransposeType transa,         // 表示矩阵 A 是否需要转置的枚举值
    TransposeType transb,         // 表示矩阵 B 是否需要转置的枚举值
    int64_t m,                    // 矩阵 C 的行数
    int64_t n,                    // 矩阵 C 的列数
    int64_t k,                    // 矩阵 A 的列数（如果不转置）或行数（如果转置）
    opmath_t alpha,               // 乘法运算的 alpha 系数
    const scalar_t *a,            // 矩阵 A 的数据指针
    int64_t lda,                  // 矩阵 A 的 leading dimension
    const scalar_t *b,            // 矩阵 B 的数据指针
    int64_t ldb,                  // 矩阵 B 的 leading dimension
    opmath_t beta,                // 乘法运算的 beta 系数
    scalar_t *c,                  // 矩阵 C 的数据指针
    int64_t ldc) {                // 矩阵 C 的 leading dimension

  // 计算 c = beta * c + alpha * (a^T @ b^T) 的矩阵乘法运算
  for (const auto i : c10::irange(m)) {
    for (const auto j : c10::irange(n)) {
      // 计算点积 dot
      const auto dot = sum(k, [&](int64_t l) -> opmath_t {
        // 根据 transa 和 transb 的值选择是否使用共轭转置
        return static_cast<opmath_t>(transa == TransposeType::ConjTranspose ? conj_impl(a[i * lda + l]) : a[i * lda + l]) *
            static_cast<opmath_t>(transb == TransposeType::ConjTranspose ? conj_impl(b[l * ldb + j]) : b[l * ldb + j]);
      });

      // 根据 beta 的值更新矩阵 C 的元素
      if (beta == opmath_t(0)) {
        c[j * ldc + i] = alpha * dot;
      } else {
        c[j * ldc + i] = beta * c[j * ldc + i] + alpha * dot;
      }
    }
  }
}
#if defined(__aarch64__) && !defined(C10_MOBILE)
// 如果目标架构是 __aarch64__，且不是 C10_MOBILE 编译时
template <>
void gemm_notrans_(
    int64_t m,
    int64_t n,
    int64_t k,
    float alpha,
    const at::Half* a,
    int64_t lda,
    const at::Half* b,
    int64_t ldb,
    float beta,
    at::Half* c,
    int64_t ldc) {
  // c += alpha * (a @ b)
  // 如果 n 等于 1，且 beta 等于 0.0，使用 gemv_notrans 执行矩阵向量乘法
  if (n == 1 && beta == 0.0) {
    at::native::blas_impl::fp16_gemv_notrans(m, k, alpha, reinterpret_cast<const float16_t*>(a), lda, reinterpret_cast<const float16_t*>(b), 1, beta, reinterpret_cast<float16_t*>(c), 1);
    return;
  }
  // 循环遍历矩阵 c
  for (const auto i : c10::irange(m)) {
    for (const auto j : c10::irange(n)) {
      // 计算点乘 dot
      const auto dot = sum(k, [&](int64_t l) -> float {
        return float(c10::detail::fp16_from_bits(a[l * lda + i].x)) *
            float(c10::detail::fp16_from_bits(b[j * ldb + l].x));
      });
      // 根据 beta 的值更新 c 的元素
      if (beta == 0) {
        c[j * ldc + i] = alpha * dot;
      } else {
        c[j * ldc + i] = beta * c[j * ldc + i] + alpha * dot;
      }
    }
  }
}

// 加载 BFloat16 指针内容并转换为 float32x4_t
inline float32x4_t load_as_float32x4(const BFloat16* ptr) {
  int32x4_t shift = vdupq_n_s32(16);
  uint32x4_t as_int = vmovl_u16(vld1_u16(reinterpret_cast<const uint16_t *>(ptr)));
  return vreinterpretq_f32_u32(vshlq_u32(as_int, shift));
}

// 计算两个 Half 类型数组的点乘结果
static float compute_dot(const at::Half* a, const at::Half* b, int64_t len) {
  return at::native::blas_impl::fp16_dot_with_fp32_arith(
    reinterpret_cast<const float16_t*>(a),
    reinterpret_cast<const float16_t*>(b),
    len);
}

// 计算两个 BFloat16 类型数组的点乘结果
static float compute_dot(const at::BFloat16* a, const at::BFloat16* b, int64_t len) {
  return at::native::blas_impl::bf16_dot_with_fp32_arith(a, b, len);
}

// 如果 transa 是 TransposeType::NoTranspose，则执行矩阵乘法 gemm 操作
template <>
void gemm_transa_(
    TransposeType transa,
    int64_t m, int64_t n, int64_t k,
    float alpha,
    const at::Half *a, int64_t lda,
    const at::Half *b, int64_t ldb,
    float beta,
    at::Half *c, int64_t ldc) {
  // c = alpha * (a.T @ b) + beta * c
  // 如果 n 等于 1，且 beta 等于 0.0，使用 gemv_trans 执行矩阵向量乘法
  if (n == 1 && beta == 0.0) {
    at::native::blas_impl::fp16_gemv_trans(k, m, alpha, reinterpret_cast<const float16_t*>(a), lda, reinterpret_cast<const float16_t*>(b), 1, beta, reinterpret_cast<float16_t*>(c), 1);
    return;
  }
  // 使用并行计算进行矩阵乘法 gemm 操作
  parallel_for(0, m, 1, [&](int64_t begin, int64_t end) {
    const auto *a_ = a + begin * lda;
    for (const auto i : c10::irange(begin, end)) {
      const auto *b_ = b;
      for (const auto j : c10::irange(n)) {
        // 计算点乘 dot
        const auto dot = compute_dot(a_, b_, k);
        b_ += ldb;
        // 根据 beta 的值更新 c 的元素
        if (beta == 0) {
          c[j * ldc + i] = alpha * dot;
        } else {
          c[j * ldc + i] = beta * c[j * ldc + i] + alpha * dot;
        }
      }
      a_ += lda;
    }
  });
}

// 如果 transa 是 TransposeType::NoTranspose，则执行矩阵乘法 gemm 操作
template <>
void gemm_transa_(
    TransposeType transa,
    int64_t m, int64_t n, int64_t k,
    float alpha,
    const at::BFloat16 *a, int64_t lda,
    const at::BFloat16 *b, int64_t ldb,
    float beta,
    at::BFloat16 *c, int64_t ldc) {
  // c = alpha * (a.T @ b) + beta * c
  // 使用并行计算进行矩阵乘法 gemm 操作
  parallel_for(0, m, 1, [&](int64_t begin, int64_t end) {
    const auto *a_ = a + begin * lda;
    // 使用范围遍历 `begin` 到 `end` 的索引，`i` 是当前的索引值
    for (const auto i : c10::irange(begin, end)) {
        // 指针 `b_` 指向数组 `b` 的起始位置
        const auto *b_ = b;
        // 使用范围遍历 `n` 的索引，`j` 是当前的索引值
        for (const auto j : c10::irange(n)) {
            // 计算向量 `a_` 和 `b_` 的点积，使用函数 `compute_dot`
            const auto dot = compute_dot(a_, b_, k);
            // 将指针 `b_` 向后移动 `ldb` 步长
            b_ += ldb;
            // 如果 `beta` 等于 0，则将 `alpha*dot` 存储在 `c` 数组中的对应位置
            if (beta == 0) {
                c[j*ldc+i] = alpha*dot;
            } else {
                // 否则，将 `alpha*dot` 与 `beta*c[j*ldc+i]` 相加后存储在 `c` 数组中的对应位置
                c[j*ldc+i] = beta*c[j*ldc+i]+alpha*dot;
            }
        }
        // 将指针 `a_` 向后移动 `lda` 步长
        a_ += lda;
    }
});
#endif

// 定义模板函数 gemm_core_，实现通用矩阵乘法操作
template <typename scalar_t, typename opmath_t>
void gemm_core_(
    TransposeType transa, TransposeType transb,
    int64_t m, int64_t n, int64_t k,
    opmath_t alpha,
    const scalar_t *a, int64_t lda,
    const scalar_t *b, int64_t ldb,
    opmath_t beta,
    scalar_t *c, int64_t ldc) {
  
  // 如果 transa 和 transb 都是 NoTranspose，则调用 gemm_notrans_ 函数
  if (transa == TransposeType::NoTranspose &&
      transb == TransposeType::NoTranspose) {
    return gemm_notrans_(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
  } else if (
      // 如果 transa 不是 NoTranspose 而 transb 是 NoTranspose，则调用 gemm_transa_ 函数
      transa != TransposeType::NoTranspose &&
      transb == TransposeType::NoTranspose) {
    gemm_transa_(transa, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
  } else if (
      // 如果 transa 是 NoTranspose 而 transb 不是 NoTranspose，则调用 gemm_transb_ 函数
      transa == TransposeType::NoTranspose &&
      transb != TransposeType::NoTranspose) {
    gemm_transb_(transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
  } else {
    // 如果 transa 和 transb 都不是 NoTranspose，则调用 gemm_transab_ 函数
    gemm_transab_(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
  }
}

// 定义模板宏 _AT_DISPATCH_GEMM_TYPES，根据不同条件分派不同的模板类型
#if !defined(C10_MOBILE)
#define _AT_DISPATCH_GEMM_TYPES(TYPE, NAME, ...)                                                \
        AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND6(                                                 \
            kHalf, kBFloat16, kFloat8_e5m2, kFloat8_e4m3fn, kFloat8_e5m2fnuz, kFloat8_e4m3fnuz, \
            TYPE, NAME, __VA_ARGS__)
#else
#define _AT_DISPATCH_GEMM_TYPES(TYPE, NAME, ...)         \
        AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(          \
            kHalf, kBFloat16,                            \
            TYPE, NAME, __VA_ARGS__)
#endif

// 定义函数 cpublas_gemm_impl，根据类型和转置类型调用 gemm_core_ 函数
void cpublas_gemm_impl(
    at::ScalarType type,
    TransposeType transa, TransposeType transb,
    int64_t m, int64_t n, int64_t k,
    const Scalar& alpha,
    const void *a, int64_t lda,
    const void *b, int64_t ldb,
    const Scalar& beta,
    void *c, int64_t ldc) {
  
  // 使用宏 _AT_DISPATCH_GEMM_TYPES 分派模板类型
  _AT_DISPATCH_GEMM_TYPES(type, "cpublas_gemm_impl", [&]{
        using opmath_t = at::opmath_type<scalar_t>;
        // 调用 gemm_core_ 函数进行矩阵乘法操作
        gemm_core_(
            transa, transb, m, n, k,
            alpha.to<opmath_t>(),
            static_cast<const scalar_t *>(a), lda,
            static_cast<const scalar_t *>(b), ldb,
            beta.to<opmath_t>(),
            static_cast<scalar_t *>(c), ldc);
      });
}

// 定义函数 cpublas_axpy_impl，根据类型和操作类型执行向量加法操作
void cpublas_axpy_impl(at::ScalarType type, int64_t n, const Scalar& _a, const void *_x, int64_t incx, void *_y, int64_t incy){
  
  // 如果类型是 bool，则执行按位或操作
  if (type == at::kBool) {
      auto a = _a.to<bool>();
      auto x = static_cast<const bool *>(_x);
      auto y = static_cast<bool *>(_y);
      int64_t i;
      for(i = 0; i < n; i++)
        y[i*incy] |= a & x[i*incx];
  } else {
    // 对于其他类型，使用宏 AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2 分派模板类型
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(at::kHalf, at::kBFloat16, type, "cpublas_axpy_impl",
      [&] {
        using opmath_t = at::opmath_type<scalar_t>;
        auto a = _a.to<opmath_t>();
        auto x = static_cast<const scalar_t *>(_x);
        auto y = static_cast<scalar_t *>(_y);
        int64_t i;
        for(i = 0; i < n; i++)
          y[i*incy] += a*x[i*incx];
      });
  }
}
void cpublas_copy_impl(at::ScalarType type, int64_t n, const void *_x, int64_t incx, void *_y, int64_t incy){
  // 使用 AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND4 宏，根据给定的类型进行类型分发
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND4(at::kComplexHalf, at::kHalf, at::kBFloat16, at::kBool, type, "cpublas_copy_impl",
    [&] {
      // 将输入指针 _x 和 _y 转换为对应的标量类型 scalar_t
      auto x = static_cast<const scalar_t *>(_x);
      auto y = static_cast<scalar_t *>(_y);
      int64_t i;
      // 遍历元素进行复制操作
      for(i = 0; i < n; i++)
        // 将 x 中的元素按照增量 incx 复制到 y 中的对应位置，增量为 incy
        y[i*incy] = x[i*incx];
    });
}

}}  // namespace cpublas::(anonymous)


// 将 cpublas::cpublas_gemm_impl 函数注册到 cpublas::gemm_stub 分发器
REGISTER_DISPATCH(cpublas::gemm_stub, &cpublas::cpublas_gemm_impl);
// 将 cpublas::cpublas_axpy_impl 函数注册到 cpublas::axpy_stub 分发器
REGISTER_DISPATCH(cpublas::axpy_stub, &cpublas::cpublas_axpy_impl);
// 将 cpublas_copy_impl 函数注册到 cpublas::copy_stub 分发器
REGISTER_DISPATCH(cpublas::copy_stub, &cpublas::cpublas_copy_impl);

}  // namespace at::native
```