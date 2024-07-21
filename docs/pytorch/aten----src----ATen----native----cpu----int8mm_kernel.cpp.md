# `.\pytorch\aten\src\ATen\native\cpu\int8mm_kernel.cpp`

```
// 定义宏，用于指定仅使用方法操作符
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
// 引入 ATen 库的 Tensor 类
#include <ATen/core/Tensor.h>

// 引入 ATen 库的调度功能
#include <ATen/Dispatch.h>
// 引入 ATen 库的并行功能
#include <ATen/Parallel.h>
// 引入 ATen 库的向量化函数
#include <ATen/cpu/vec/functional.h>
// 引入 ATen 库的向量化支持
#include <ATen/cpu/vec/vec.h>
// 引入 ATen 库的 CPU 下整数矩阵乘法内核
#include <ATen/native/cpu/int_mm_kernel.h>
// 引入 ATen 库的 CPU 工具函数
#include <ATen/native/cpu/utils.h>
// 引入 C10 库的整数范围函数
#include <c10/util/irange.h>
// 引入 C10 库的强制展开模板支持
#include <c10/util/Unroll.h>

// 如果在 Windows 平台下，定义 RESTRICT 宏为 __restrict
// 否则定义为 __restrict__
#if (defined(_WIN32) || defined(_WIN64))
#define RESTRICT __restrict
#else
#define RESTRICT __restrict__
#endif

namespace at::native {

// 命名空间内部的匿名命名空间，用于局部定义的函数和变量
namespace {

// 当 CPU 支持 AVX512 指令集且不是在 MSVC 编译器下时
template <int BLOCK_M, int BLOCK_N>
inline void tinygemm_kernel(
    const BFloat16* RESTRICT A,  // 输入矩阵 A，限制指针不会重叠
    const int8_t* RESTRICT B,    // 输入矩阵 B，限制指针不会重叠
    const BFloat16* RESTRICT scales,  // 每列的缩放因子，限制指针不会重叠
    BFloat16* RESTRICT C,        // 输出矩阵 C，限制指针不会重叠
    int lda,  // 矩阵 A 的列跨度
    int ldb,  // 矩阵 B 的列跨度
    int ldc,  // 矩阵 C 的列跨度
    int K) {  // 矩阵乘法的维度 K

  constexpr int ROWS = BLOCK_M;  // 定义行数为 BLOCK_M
  constexpr int COLS = BLOCK_N;  // 定义列数为 BLOCK_N

  const int PREFETCH_SIZE_K = 16 * 4;  // 预取大小为 64 字节

  __m512 va;  // AVX512 的 512 位寄存器，用于存储 A 的向量
  __m512 vb[COLS];  // AVX512 的 512 位寄存器数组，用于存储 B 的向量
  __m512 vc[ROWS * COLS];  // AVX512 的 512 位寄存器数组，用于存储 C 的向量
  __m512 scale[COLS];  // AVX512 的 512 位寄存器数组，用于存储缩放因子的向量

  // 定义加载缩放因子的 Lambda 函数
  auto load_scale = [&](int i) {
    float ss = static_cast<float>(scales[i]);
    scale[i] = _mm512_set1_ps(ss);  // 使用缩放因子初始化 AVX512 向量
  };
  c10::ForcedUnroll<COLS>{}(load_scale);  // 强制展开加载缩放因子的循环

  // 初始化 C 矩阵向量为零
  auto loadc = [&](auto i) {
    vc[i] = _mm512_setzero_ps();  // 将 C 的 AVX512 向量置零
  };
  c10::ForcedUnroll<ROWS * COLS>{}(loadc);  // 强制展开初始化 C 的循环

  // 定义计算 Lambda 函数
  auto compute = [&](auto i, int k) {
    constexpr int row = i / COLS;  // 计算当前元素所在的行
    constexpr int col = i % COLS;  // 计算当前元素所在的列

    if constexpr (col == 0) {
      // 如果是第一列，则加载 A 的向量
      __m256i a16 = _mm256_load_si256((__m256i*)(A + row * lda + k));
      if (k + PREFETCH_SIZE_K < K) {
        _mm_prefetch(A + row * lda + k + PREFETCH_SIZE_K, _MM_HINT_T0);  // 预取下一段 A 数据
      }
      vec::cvtbf16_fp32(a16, va);  // 将加载的 BFloat16 转换为单精度浮点数存入 va
    }

    if constexpr (row == 0) {
      // 如果是第一行，则加载 B 的向量
      __m128i b8 = _mm_load_si128((__m128i*)(B + col * ldb + k));
      if (k + PREFETCH_SIZE_K < K) {
        _mm_prefetch(B + col * ldb + k + PREFETCH_SIZE_K, _MM_HINT_T0);  // 预取下一段 B 数据
      }
      __m512i b32 = _mm512_cvtepi8_epi32(b8);  // 将加载的 int8 转换为 AVX512 的 32 位整数
      vb[col] = _mm512_cvtepi32_ps(b32);  // 将 AVX512 的 32 位整数转换为单精度浮点数
      vb[col] = _mm512_mul_ps(vb[col], scale[col]);  // 对 B 的向量乘以缩放因子
    }

    constexpr int idx = row * COLS + col;  // 计算在 C 中的索引位置
    vc[idx] = _mm512_fmadd_ps(va, vb[col], vc[idx]);  // 执行 FMA 操作将结果累加到 C 中
  };

  // 执行主循环，每次处理 16 列数据
  for (int k = 0; k < K; k += 16) {
      c10::ForcedUnroll<ROWS * COLS>{}(compute, k);  // 强制展开计算的循环
  }

  // 定义存储 Lambda 函数
  auto storec = [&](auto i) {
    constexpr int row = i / COLS;  // 计算当前元素所在的行
    constexpr int col = i % COLS;  // 计算当前元素所在的列
    C[row * ldc + col] = static_cast<BFloat16>(_mm512_reduce_add_ps(vc[i]));  // 将结果存入 C 矩阵
  };
  c10::ForcedUnroll<ROWS * COLS>{}(storec);  // 强制展开存储结果的循环
}

// 如果 CPU 支持 AVX2 指令集且不是在 MSVC 编译器下时
template <int BLOCK_M, int BLOCK_N>
inline void tinygemm_kernel(
    const BFloat16* RESTRICT A,


以上是对给定代码的详细注释，按照要求将每行代码功能进行解释和说明。
    const`
// 定义一个函数，计算矩阵乘法 C = A * B，其中 A 是大小为 ROWS x lda 的矩阵，B 是大小为 COLS x ldb 的矩阵，C 是大小为 ROWS x COLS 的矩阵
void gemm_kernel(
    const int8_t* RESTRICT A,
    const BFloat16* RESTRICT B,
    BFloat16* RESTRICT C,
    int lda,
    int ldb,
    int ldc,
    int K) {

  // 定义常量，表示块的大小
  constexpr int ROWS = BLOCK_M;
  constexpr int COLS = BLOCK_N;

  // 设置预取的步长
  const int PREFETCH_SIZE_K = 16 * 4;

  // 定义向量寄存器
  __m256 va;
  __m256 vb[COLS];
  __m256 vc[ROWS * COLS];
  __m256 scale[COLS];

  // 定义加载 scale 的 Lambda 函数
  auto load_scale = [&](int i) {
    float ss = static_cast<float>(scales[i]);
    scale[i] = _mm256_set1_ps(ss);
  };
  c10::ForcedUnroll<COLS>{}(load_scale);

  // 初始化 vc 数组为零向量
  auto loadc = [&](auto i) {
    vc[i] = _mm256_setzero_ps();
  };
  c10::ForcedUnroll<ROWS * COLS>{}(loadc);

  // 计算矩阵乘法的 Lambda 函数
  auto compute = [&](auto i, int k) {
    // 计算行和列的索引
    constexpr int row = i / COLS;
    constexpr int col = i % COLS;

    // 如果列是 0，则加载 A 的数据
    if constexpr (col == 0) {
      __m128i a16 = _mm_load_si128((__m128i*)(A + row * lda + k));
      // 预取下一批数据
      if (k + PREFETCH_SIZE_K < K) {
        _mm_prefetch(A + row * lda + k + PREFETCH_SIZE_K, _MM_HINT_T0);
      }
      vec::cvtbf16_fp32(a16, va);
    }

    // 如果行是 0，则加载 B 的数据并乘以 scale
    if constexpr (row == 0) {
       __m128i b8 = _mm_loadu_si64((__m128i*)(B + col * ldb + k));
       // 预取下一批数据
       if (k + PREFETCH_SIZE_K < K) {
         _mm_prefetch(B + col * ldb + k + PREFETCH_SIZE_K, _MM_HINT_T0);
       }
       __m256i b32 = _mm256_cvtepi8_epi32(b8);
       vb[col] = _mm256_cvtepi32_ps(b32);
       vb[col] = _mm256_mul_ps(vb[col], scale[col]);
     }

     // 计算乘法累加到 vc 数组
     constexpr int idx = row * COLS + col;
     vc[idx] = _mm256_fmadd_ps(va, vb[col], vc[idx]);
  };

  // 循环执行矩阵乘法的计算
  for (int k = 0; k < K; k += 8) {
    c10::ForcedUnroll<ROWS * COLS>{}(compute, k);
  }

  // 将结果存储到矩阵 C 中
  auto storec = [&](auto i) {
    constexpr int row = i / COLS;
    constexpr int col = i % COLS;
    C[row * ldc + col] = static_cast<BFloat16>(_mm256_reduce_add_ps(vc[i]));
  };
  c10::ForcedUnroll<ROWS * COLS>{}(storec);
}
#endif
// 结束条件：如果已定义 C10_MOBILE 且同时定义了 __aarch64__，则结束当前代码段

#if !defined(C10_MOBILE) && defined(__aarch64__)
// 如果未定义 C10_MOBILE 并且定义了 __aarch64__，则包含 ARM NEON 头文件
#include <arm_neon.h>

// 定义一个内联函数 reduce，用于计算四个单精度浮点数向量的和
inline float reduce(float32x4_t x) {
    auto sum = vpaddq_f32(x, x);  // 向量加法，结果存储在 sum 中
    return vgetq_lane_f32(vpaddq_f32(sum, sum), 0);  // 提取 sum 向量的第一个元素并返回
}

// 定义一个内联函数 load_as_float32x4x2，从半精度浮点数指针加载数据并转换为两个 float32x4x2_t 向量
inline float32x4x2_t load_as_float32x4x2(const Half* ptr) {
  float16x8_t f16_val = vld1q_f16(reinterpret_cast<const float16_t *>(ptr));  // 从 ptr 加载 8 个半精度浮点数
  auto val_low = vcvt_f32_f16(vget_low_f16(f16_val));  // 转换得到低 4 个元素的 float32x4_t 向量
  auto val_high = vcvt_f32_f16(vget_high_f16(f16_val));  // 转换得到高 4 个元素的 float32x4_t 向量
  return {val_low, val_high};  // 返回由这两个向量组成的结构体
}

// 定义一个内联函数 load_as_float32x4，从半精度浮点数指针加载数据并转换为 float32x4_t 向量
inline float32x4_t load_as_float32x4(const Half* ptr) {
    return vcvt_f32_f16(vld1_f16(reinterpret_cast<const float16_t *>(ptr)));  // 加载 4 个半精度浮点数并转换为单精度浮点数向量
}

// 定义一个内联函数 load_as_float32x4x2，从 BFloat16 指针加载数据并转换为两个 float32x4x2_t 向量
inline float32x4x2_t load_as_float32x4x2(const BFloat16* ptr) {
  int32x4_t shift = vdupq_n_s32(16);  // 创建一个所有元素为 16 的 int32x4_t 向量
  uint16x8_t u16_val = vld1q_u16(reinterpret_cast<const uint16_t *>(ptr));  // 从 ptr 加载 8 个 BFloat16 并转换为 uint16x8_t 向量
  uint32x4_t int_low = vmovl_u16(vget_low_u16(u16_val));  // 将低 4 个元素扩展为 uint32x4_t 向量
  uint32x4_t int_high = vmovl_u16(vget_high_u16(u16_val));  // 将高 4 个元素扩展为 uint32x4_t 向量
  return {vreinterpretq_f32_u32(vshlq_u32(int_low, shift)), vreinterpretq_f32_u32(vshlq_u32(int_high, shift))};  // 左移操作后转换为两个 float32x4_t 向量
}

// 定义一个内联函数 load_as_float32x4，从 BFloat16 指针加载数据并转换为 float32x4_t 向量
inline float32x4_t load_as_float32x4(const BFloat16* ptr) {
  int32x4_t shift = vdupq_n_s32(16);  // 创建一个所有元素为 16 的 int32x4_t 向量
  uint32x4_t as_int = vmovl_u16(vld1_u16(reinterpret_cast<const uint16_t *>(ptr)));  // 从 ptr 加载 4 个 BFloat16 并转换为 uint32x4_t 向量
  return vreinterpretq_f32_u32(vshlq_u32(as_int, shift));  // 左移操作后转换为 float32x4_t 向量
}

// 定义一个内联函数 load_as_float32x4，从 float 指针加载数据并转换为 float32x4_t 向量
inline float32x4_t load_as_float32x4(const float* ptr) {
  return vld1q_f32(ptr);  // 直接加载 4 个单精度浮点数
}

// 定义一个内联函数 load_as_float32x4x2，从 float 指针加载数据并转换为两个 float32x4x2_t 向量
inline float32x4x2_t load_as_float32x4x2(const float* ptr) {
  return {vld1q_f32(ptr), vld1q_f32(ptr + 4)};  // 分别加载前 4 个和后 4 个单精度浮点数
}

// 定义模板函数 tinygemm_kernel_，执行矩阵乘法的一部分计算，处理输入为 T 类型，块大小为 BLOCK_M 和 BLOCK_N
template <int BLOCK_M, int BLOCK_N, typename T>
inline void tinygemm_kernel_(
    const T* RESTRICT A,
    const int8_t* RESTRICT B,
    const T* RESTRICT scales,
    T* RESTRICT C,
    int lda,
    int ldb,
    int ldc,
    int K) {

  for (const auto m : c10::irange(BLOCK_M)) {
    float32x4_t c_val[BLOCK_N];  // 声明长度为 BLOCK_N 的 c_val 数组，存储结果向量
    c10::ForcedUnroll<BLOCK_N>{}([&](auto i) {
        c_val[i] = vdupq_n_f32(0.0);  // 初始化 c_val[i] 为全 0 的 float32x4_t 向量
    });
    for (int k = 0; k < K; k += 8) {
      auto a_val = load_as_float32x4x2(A + m * lda + k);  // 从 A 加载两个 float32x4_t 向量
      c10::ForcedUnroll<BLOCK_N>{}([&](auto i) {
        int16x8_t b_val = vmovl_s8(vld1_s8(B + i * ldb + k));  // 从 B 加载并扩展为 int16x8_t 向量
        auto b_val_low = vcvtq_f32_s32(vmovl_s16(vget_low_s16(b_val)));  // 转换为 float32x4_t 向量
        auto b_val_high = vcvtq_f32_s32(vmovl_s16(vget_high_s16(b_val)));  // 转换为 float32x4_t 向量
        c_val[i] = vfmaq_f32(c_val[i], a_val.val[1], b_val_high);  // 使用 vfmaq_f32 计算乘加
        c_val[i] = vfmaq_f32(c_val[i], a_val.val[0], b_val_low);  // 使用 vfmaq_f32 计算乘加
      });
    }

#if __OPTIMIZE__
    float32x4_t scale_val = load_as_float32x4(scales);  // 加载 scales 并转换为 float32x4_t 向量
    c10::ForcedUnroll<BLOCK_N>{}([&](auto i) {
      C[m * ldc + i] = reduce(c_val[i]) * vgetq_lane_f32(scale_val, i);  // 计算最终结果并写入 C
    });
#else
    // GCC 编译器无法在编译时推断索引的能力，使用运行时转换 scales[i]
    c10::ForcedUnroll<BLOCK_N>{}([&](auto i) {
      C[m * ldc + i] = reduce(c_val[i]) * float(scales[i]);  // 计算最终结果并写入 C
    });
#endif
  }
}

// 定义模板函数 tinygemm_kernel，执行矩阵乘法的一部分计算，处理输入为 Half 类型的 A 和 scales，以及 int8_t 类型的 B
template <int BLOCK_M, int BLOCK_N>
inline void tinygemm_kernel(
    const Half* RESTRICT A,
    const int8_t* REST
    # 声明一个指向 Half 类型的指针 C，用于存储矩阵乘积的结果
    Half* RESTRICT C,
    # 矩阵 A 的列数或者矩阵 C 的行数
    int lda,
    # 矩阵 B 的列数或者矩阵 C 的列数
    int ldb,
    # 矩阵 C 的列数
    int ldc,
    # 矩阵乘法中的维度 K，即矩阵 A 的行数和矩阵 B 的行数
    int K) {
  # 调用 tinygemm_kernel_<BLOCK_M, BLOCK_N> 函数来执行矩阵乘法的计算
  tinygemm_kernel_<BLOCK_M, BLOCK_N>(A, B, scales, C, lda, ldb, ldc, K);
// 定义模板函数，用于调用特定尺寸的 tinygemm_kernel 函数处理 BFloat16 类型数据
template <int BLOCK_M, int BLOCK_N>
inline void tinygemm_kernel(
    const BFloat16* RESTRICT A,
    const int8_t* RESTRICT B,
    const BFloat16* RESTRICT scales,
    BFloat16* RESTRICT C,
    int lda,
    int ldb,
    int ldc,
    int K) {
  // 调用具体的内部实现函数 tinygemm_kernel_，处理 BFloat16 数据
  tinygemm_kernel_<BLOCK_M, BLOCK_N>(A, B, scales, C, lda, ldb, ldc, K);
}

// 定义模板函数，用于调用特定尺寸的 tinygemm_kernel 函数处理 float 类型数据
template <int BLOCK_M, int BLOCK_N>
inline void tinygemm_kernel(
    const float* RESTRICT A,
    const int8_t* RESTRICT B,
    const float* RESTRICT scales,
    float* RESTRICT C,
    int lda,
    int ldb,
    int ldc,
    int K) {
  // 调用具体的内部实现函数 tinygemm_kernel_，处理 float 数据
  tinygemm_kernel_<BLOCK_M, BLOCK_N>(A, B, scales, C, lda, ldb, ldc, K);
}
#endif

// 非向量化版本的 tinygemm_kernel 函数模板
template <int BLOCK_M, int BLOCK_N, typename T>
inline void tinygemm_kernel(
    const T* RESTRICT A,
    const int8_t* RESTRICT B,
    const T* RESTRICT scales,
    T* RESTRICT C,
    int lda,
    int ldb,
    int ldc,
    int K) {
  // 使用双重循环遍历 BLOCK_M x BLOCK_N 大小的矩阵块
  for (const auto m : c10::irange(BLOCK_M)) {
    for (const auto n : c10::irange(BLOCK_N)) {
      float c_val = 0; // 初始化结果值为 0
      float scale_val = static_cast<float>(scales[n]); // 提取当前列的缩放因子
      // 使用双重循环遍历矩阵块中的每一列
      for (const auto k : c10::irange(K)) {
        float a_val = static_cast<float>(A[m * lda + k]); // 提取 A 矩阵元素
        float b_val = static_cast<float>(B[n * ldb + k]); // 提取 B 矩阵元素
        c_val += a_val * (b_val * scale_val); // 执行矩阵乘法并加到结果中
      }
      C[m * ldc + n] = c_val; // 将计算结果存入 C 矩阵对应位置
    }
  }
}

// 宏定义，用于根据不同的 NB_SIZE 启动特定尺寸的 tinygemm_kernel 函数
#define LAUNCH_TINYGEMM_KERNEL(MB_SIZE, NB_SIZE)                 \
  tinygemm_kernel<MB_SIZE, NB_SIZE>(                             \
      A_ptr, B_ptr, S_ptr, C_ptr,                                \
      K, K, N, K);

// 宏定义，根据 nb_size 的值选择合适的 tinygemm_kernel 函数启动
#define LAUNCH_TINYGEMM_NB_SIZE(MB_SIZE)                         \
  switch (nb_size) {                                             \
    case 1:                                                      \
      LAUNCH_TINYGEMM_KERNEL(MB_SIZE, 1);                        \
      break;                                                     \
    case 2:                                                      \
      LAUNCH_TINYGEMM_KERNEL(MB_SIZE, 2);                        \
      break;                                                     \
    case 3:                                                      \
      LAUNCH_TINYGEMM_KERNEL(MB_SIZE, 3);                        \
      break;                                                     \
    case 4:                                                      \
      LAUNCH_TINYGEMM_KERNEL(MB_SIZE, 4);                        \
      break;                                                     \
    default:                                                     \
      TORCH_CHECK(false, "Unsupported n block size: ", nb_size); \
      break;                                                     \
  }

// 定义模板函数，用于执行 int8 数据类型的矩阵乘法
template<typename T>
void int8pack_mm_kernel_(
    const Tensor& C,
    const Tensor& A,
    const Tensor& B,
    // 定义函数，执行矩阵乘法操作，使用张量 A, B, C 和缩放因子 scales
    const Tensor& scales) {
    
      // 获取张量 A 的常量数据指针
      const auto* A_data = A.const_data_ptr<T>();
      // 获取张量 B 的常量数据指针，数据类型为 int8_t
      const auto* B_data = B.const_data_ptr<int8_t>();
      // 获取张量 C 的可修改数据指针
      auto* C_data = C.data_ptr<T>();
      // 获取缩放因子 scales 的常量数据指针
      const auto* S_data = scales.const_data_ptr<T>();
    
      // 获取矩阵 A 的行数 M，矩阵 B 的行数 N，矩阵 A 的列数 K
      int M = A.size(0);
      int N = B.size(0);
      int K = A.size(1);
    
      // 定义块的大小
      constexpr int BLOCK_M = 4;
      constexpr int BLOCK_N = 4;
    
      // 计算需要的块数目
      const int MB = (M + BLOCK_M - 1) / BLOCK_M;
      const int NB = (N + BLOCK_N - 1) / BLOCK_N;
    
      // 并行执行循环，分割为 MB * NB 块
      at::parallel_for(0, MB * NB, 0, [&](int begin, int end) {
        // 初始化 mb 和 nb 的索引
        int mb{0}, nb{0};
        data_index_init(begin, mb, MB, nb, NB);
    
        // 对每个块进行循环迭代
        for (const auto i : c10::irange(begin, end)) {
          (void)i;
    
          // 计算当前块的起始行和列，以及块的大小
          int mb_start = mb * BLOCK_M;
          int mb_size = std::min(BLOCK_M, M - mb_start);
          int nb_start = nb * BLOCK_N;
          int nb_size = std::min(BLOCK_N, N - nb_start);
    
          // 获取当前块的 A, B, S, C 的指针
          const auto* A_ptr = A_data + mb_start * K;
          const auto* B_ptr = B_data + nb_start * K;
          const auto* S_ptr = S_data + nb_start;
          auto* C_ptr = C_data + mb_start * N + nb_start;
    
          // 根据当前块大小选择合适的小型矩阵乘法函数进行计算
          switch (mb_size) {
            case 1:
              LAUNCH_TINYGEMM_NB_SIZE(1);
              break;
            case 2:
              LAUNCH_TINYGEMM_NB_SIZE(2);
              break;
            case 3:
              LAUNCH_TINYGEMM_NB_SIZE(3);
              break;
            case 4:
              LAUNCH_TINYGEMM_NB_SIZE(4);
              break;
            default:
              // 如果块大小不支持，抛出异常
              TORCH_CHECK(false, "Unsupported m block size: ", mb_size);
          }
    
          // 移动到下一个索引
          data_index_step(mb, MB, nb, NB);
        }
      });
    }
}

// 定义一个函数 int8pack_mm_kernel，用于执行 int8 矩阵乘法操作
void int8pack_mm_kernel(
    const Tensor& C,         // 输出张量 C
    const Tensor& A,         // 输入张量 A
    const Tensor& B,         // 输入张量 B
    const Tensor& scales) {  // 输入张量 scales，用于缩放操作

  // 如果输出张量 C 的数据类型为 kHalf
  if (C.dtype() == kHalf) {
    // 调用具体实现模板函数 int8pack_mm_kernel_<Half> 处理
    int8pack_mm_kernel_<Half>(C, A, B, scales);
  }
  // 如果输出张量 C 的数据类型为 kBFloat16
  else if (C.dtype() == kBFloat16) {
    // 调用具体实现模板函数 int8pack_mm_kernel_<BFloat16> 处理
    int8pack_mm_kernel_<BFloat16>(C, A, B, scales);
  }
  // 如果输出张量 C 的数据类型为其它（假定为 float）
  else {
    // 调用具体实现模板函数 int8pack_mm_kernel_<float> 处理
    int8pack_mm_kernel_<float>(C, A, B, scales);
  }
}

// 匿名命名空间结束

} // anonymous namespace

// 使用宏 ALSO_REGISTER_AVX512_DISPATCH 注册 AVX512 分发函数 int8pack_mm_stub，并指定处理函数为 int8pack_mm_kernel
ALSO_REGISTER_AVX512_DISPATCH(int8pack_mm_stub, &int8pack_mm_kernel);

// 结束 at::native 命名空间

} // at::native
```