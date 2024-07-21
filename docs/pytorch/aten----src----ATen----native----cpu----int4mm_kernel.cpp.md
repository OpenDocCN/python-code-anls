# `.\pytorch\aten\src\ATen\native\cpu\int4mm_kernel.cpp`

```
    // 包含头文件 <type_traits>，用于类型特性的支持
#include <type_traits>
    // 定义宏 TORCH_ASSERT_ONLY_METHOD_OPERATORS，用于在包含 <ATen/core/Tensor.h> 时控制断言的行为
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
    // 包含 ATen 库的 Tensor 类的头文件
#include <ATen/core/Tensor.h>

    // 包含 ATen 库的 Dispatch.h 头文件，用于分发机制
#include <ATen/Dispatch.h>
    // 包含 ATen 库的 Parallel.h 头文件，用于并行处理支持
#include <ATen/Parallel.h>
    // 包含 ATen 库的向量功能头文件，位于 cpu/vec 目录下
#include <ATen/cpu/vec/functional.h>
    // 包含 ATen 库的向量化支持头文件，位于 cpu/vec 目录下
#include <ATen/cpu/vec/vec.h>
    // 包含 ATen 库的整数矩阵乘法的 CPU 实现的头文件，位于 native/cpu 目录下
#include <ATen/native/cpu/int_mm_kernel.h>
    // 包含 ATen 库的 CPU 实用工具函数头文件，位于 native/cpu 目录下
#include <ATen/native/cpu/utils.h>
    // 包含 c10 库的 irange.h 头文件，提供整数范围的支持
#include <c10/util/irange.h>
    // 包含 c10 库的 Unroll.h 头文件，提供展开循环的支持
#include <c10/util/Unroll.h>

    // 如果操作系统为 Windows，定义 RESTRICT 为 __restrict；否则定义为 __restrict__
#if (defined(_WIN32) || defined(_WIN64))
#define RESTRICT __restrict
#else
#define RESTRICT __restrict__
#endif

    // ATen 库的 native 命名空间
namespace at::native {

    // 匿名命名空间，定义一些内部函数和常量

    // 内联函数，判断索引是否是块的起始位置
inline bool is_block_start(int index, int BLOCK_SIZE) {
        // 使用位运算判断是否是块的起始位置
  return !(index & (BLOCK_SIZE -1));
}

    // 如果 CPU 支持 AVX512 或 AVX2，且不是在 MSVC 编译环境下
#if (defined(CPU_CAPABILITY_AVX512) || defined(CPU_CAPABILITY_AVX2)) && !defined(_MSC_VER)
    // 将 16x int4 转换为 int8，处理每次 64 位
    // 在 AVX2 和 AVX512 中使用
inline __m128i conver_int4_to_int8(const uint8_t* data) {
        // 加载 64 位数据作为 __m128i 类型
  __m128i tmp = _mm_loadu_si64((const __m128i*)data);
        // 将 uint8 转换为 uint16
  __m128i bytes = _mm_cvtepu8_epi16(tmp);
        // 创建低位掩码
  const __m128i lowMask = _mm_set1_epi8(0xF);
        // 取反低位掩码，得到高位
  __m128i high = _mm_andnot_si128(lowMask, bytes);
        // 取低位
  __m128i low = _mm_and_si128(lowMask, bytes);
        // 高位左移 4 位
  high = _mm_slli_epi16(high, 4);
        // 合并高位和低位
  bytes = _mm_or_si128(low, high);
  return bytes;
}
#endif

    // 如果 CPU 支持 AVX512，且不是在 MSVC 编译环境下
#if defined(CPU_CAPABILITY_AVX512) && !defined(_MSC_VER)

    // A block : {BLOCK_M, BLOCK_K}, lda = K
    // B block : {BLOCK_K, BLOCK_N / 2}, ldb = BLOCK_N / 2
    // C block : {BLOCK_M, BLOCK_N}, ldc = N
    //
    // ScaleAndZeros block : {1, BLOCK_N, 2}
    //
template <int BLOCK_M, int BLOCK_N>
inline void tinygemm_kernel(
        // A 矩阵，存储在 BFloat16 类型的数组中，限定指针只能指向指定的内存
    const BFloat16* RESTRICT A,
        // B 矩阵，存储在 uint8_t 类型的数组中，限定指针只能指向指定的内存
    const uint8_t* RESTRICT B,
        // 存储 Scale 和 Zeros 的矩阵，存储在 BFloat16 类型的数组中，限定指针只能指向指定的内存
    const BFloat16* RESTRICT ScaleAndZeros,
        // 结果矩阵 C，存储在 BFloat16 类型的数组中，限定指针只能指向指定的内存
    BFloat16* RESTRICT C,
        // A 矩阵的列数
    int lda,
        // B 矩阵的列数
    int ldb,
        // C 矩阵的列数
    int ldc,
        // A 和 B 矩阵的内部维度
    int K,
        // K 块的大小
    int BLOCK_K) {

        // 块的行数
  constexpr int ROWS = BLOCK_M;
        // 块的列数，每列有 16 个元素
  constexpr int COLS = BLOCK_N / 16;

        // 预取的 K 的大小，每次预取 16 * 4 个元素
  const int PREFETCH_SIZE_K = 16 * 4;
        // 计算 PREFETCH_SIZE_K 对应的块数
  const int PREFETCH_SIZE_KB = (PREFETCH_SIZE_K + BLOCK_K - 1) / BLOCK_K;

        // K 的块数
  const int KB = K / BLOCK_K;

        // 各种向量类型的声明
  __m512 va;
  __m512 vb[COLS];
  __m512 vc[ROWS * COLS];
  __m512 scale[COLS];
  __m512 zero[COLS];

        // 查找表，用于将 int4 值反量化为 bf16
        // 值在真实的 int4 [-8, 7] 范围内
        //
        // dequant = (bf16(int4_value) * bf16_scale) + bf16_zero
        //
  static const __m512 lut = _mm512_set_ps(
      7.0f, 6.0f, 5.0f, 4.0f,
      3.0f, 2.0f, 1.0f, 0.0f,
      -1.0f, -2.0f, -3.0f, -4.0f,
      -5.0f, -6.0f, -7.0f, -8.0f);

        // 转置的索引表
  static const __m512i idx1 = _mm512_set_epi32(
      30, 28, 26, 24, 22, 20, 18, 16,
      14, 12, 10, 8, 6, 4, 2, 0);
  static const __m512i idx2 = _mm512_set_epi32(
      31, 29, 27, 25, 23, 21, 19, 17,
      15, 13, 11, 9, 7, 5, 3, 1);

        // 加载 Scale 和 Zero 点
  auto load_scale_and_zeros = [&](int i, int _kb) {
        // 加载 2x bfloat16 向量
    __m512i t = _mm512_loadu_si512((__m512i*)(ScaleAndZeros + _kb * ldc * 2 + 32 * i));
        // 如果 _kb + PREFETCH_SIZE_KB 小于 KB，预取 ScaleAndZeros 数据
    if (_kb + PREFETCH_SIZE_KB < KB) {
      _mm_prefetch(ScaleAndZeros + (_kb + PREFETCH_SIZE_KB) * ldc * 2 + 32 * i, _MM_HINT_T0);
    }


This completes the annotated code snippet according to the guidelines provided.
    // 将输入转换为两个 2x8 的 f32 向量
    __m512 a, b;
    vec::cvtbf16_fp32(t, a, b);

    // 将 scale_and_zero 从 {16, 2} 转置为 {2, 16}
    // 输入：
    //   a: {s0, z0, s1, z1, ..., s7, z7}
    //   b: {s8, z8, s9, z9, ..., s15, z15}
    // 输出：
    //   scale: {s0, s1, s2, ..., s15}
    //   zero:  {z0, z1, z2, ..., z15}
    scale[i] = _mm512_mask_permutex2var_ps(a, 0xffff, idx1, b);
    zero[i] = _mm512_mask_permutex2var_ps(a, 0xffff, idx2, b);
  };

  // 初始化 vc[i] 为零向量
  auto loadc = [&](auto i) {
    vc[i] = _mm512_setzero_ps();
  };
  c10::ForcedUnroll<ROWS * COLS>{}(loadc);

  // 计算函数，使用延展量 COLS
  auto compute = [&, COLS](auto i, int k) {
    constexpr int row = i / COLS;
    constexpr int col = i % COLS;

    // 如果 col 是 0，则进行初始化操作
    if constexpr (col == 0) {
      float aa = static_cast<float>(A[row * lda + k]);
      // 预取 A 的下一个数据块，以提高性能
      if (k + PREFETCH_SIZE_K < K) {
        _mm_prefetch(A + row * lda + k + PREFETCH_SIZE_K, _MM_HINT_T0);
      }
      // 设置 va 为 aa 的广播值
      va = _mm512_set1_ps(aa);
    }

    // 如果 row 是 0
    if constexpr (row == 0) {
      // 当 COLS 等于 4 时，处理每行的数据以减少去量化的开销
      if constexpr (COLS == 4) {
        // 当 BLOCK_N = 64 时，每次处理一行，以减少去量化的开销
        if constexpr (col == 0) {
          // 加载 B 的数据，并转换为 32 位整数
          __m256i b4 = _mm256_loadu_si256((__m256i*)(B + k * ldb));
          // 预取 B 的下一个数据块，以提高性能
          if (k + PREFETCH_SIZE_K < K) {
            _mm_prefetch(B + (k + PREFETCH_SIZE_K) * ldb, _MM_HINT_T0);
          }

          // 将 b4 转换为 32 位整数，并通过 lut 执行 permute 操作
          __m512i b32 = _mm512_cvtepu8_epi32(_mm256_castsi256_si128(b4));
          vb[0] = _mm512_permutexvar_ps(b32, lut);
          vb[0] = _mm512_fmadd_ps(vb[0], scale[0], zero[0]);
          vb[2] = _mm512_permutexvar_ps(_mm512_srli_epi32(b32, 4), lut);
          vb[2] = _mm512_fmadd_ps(vb[2], scale[2], zero[2]);

          // 继续处理 b4 的第二部分数据
          b32 = _mm512_cvtepu8_epi32(_mm256_extracti128_si256(b4, 1));
          vb[1] = _mm512_permutexvar_ps(b32, lut);
          vb[1] = _mm512_fmadd_ps(vb[1], scale[1], zero[1]);
          vb[3] = _mm512_permutexvar_ps(_mm512_srli_epi32(b32, 4), lut);
          vb[3] = _mm512_fmadd_ps(vb[3], scale[3], zero[3]);
        }
      } else {
        // 否则，处理单个列的情况，将 B 转换为 32 位整数，并通过 lut 执行 permute 操作
        __m128i b8 = conver_int4_to_int8(B + k * ldb + col * 8);
        __m512i b32 = _mm512_cvtepu8_epi32(b8);
        vb[col] = _mm512_permutexvar_ps(b32, lut);
        vb[col] = _mm512_fmadd_ps(vb[col], scale[col], zero[col]);
      }
    }

    // 计算索引 idx，然后通过 va、vb[col] 和 vc[idx] 执行 FMA 操作
    constexpr int idx = row * COLS + col;
    vc[idx] = _mm512_fmadd_ps(va, vb[col], vc[idx]);
  };

  // 对于每个 k，循环执行计算操作
  for (int k = 0, kb = 0; k < K; ++k) {
    // 如果 k 是块的起始点，执行加载 scale 和 zero 的操作
    if (is_block_start(k, BLOCK_K)) {
      c10::ForcedUnroll<COLS>{}(load_scale_and_zeros, kb++);
    }
    // 执行计算函数
    c10::ForcedUnroll<ROWS * COLS>{}(compute, k);
  }

  // 存储结果到 C
  auto storec = [&, COLS](auto i) {
    constexpr int row = i / COLS;
    constexpr int col = i % COLS;
    if constexpr (COLS == 4) {
      // 如果 COLS 等于 4，进入此条件分支
      // 当 BLOCK_N = 64 时，逐行处理以减少 `cvtfp32_bf16` 的开销
      if constexpr (col == 0) {
        // 如果 col 等于 0，则处理每一行以降低 `cvtfp32_bf16` 的开销
        // 将第一行的前两列数据转换为 __m512i 类型
        __m512i c01 = vec::cvtfp32_bf16(vc[row * 4 + 0], vc[row * 4 + 1]);
        // 将第一行的后两列数据转换为 __m512i 类型
        __m512i c23 = vec::cvtfp32_bf16(vc[row * 4 + 2], vc[row * 4 + 3]);
        // 将转换后的数据存储到 C 矩阵中对应位置
        _mm512_storeu_si512((__m512i*)(C + row * ldc + 0 * 32), c01);
        _mm512_storeu_si512((__m512i*)(C + row * ldc + 1 * 32), c23);
      }
    } else {
      // 如果 COLS 不等于 4，进入此分支
      // 将当前元素 vc[i] 转换为 __m256i 类型
      __m256i ci = vec::cvtfp32_bf16(vc[i]);
      // 将转换后的数据存储到 C 矩阵中对应位置
      _mm256_storeu_si256((__m256i*)(C + row * ldc + col * 16), ci);
    }
  };
  // 使用 c10 库的 ForcedUnroll 执行循环展开，展开次数为 ROWS * COLS
  c10::ForcedUnroll<ROWS * COLS>{}(storec);
}

#elif defined(CPU_CAPABILITY_AVX2) && !defined(_MSC_VER)
// AVX2 指令集和非 Microsoft 编译器的条件编译分支

template <int BLOCK_M, int BLOCK_N>
inline void tinygemm_kernel(
    const BFloat16* RESTRICT A,
    // A 数组，存储限制在 BFloat16 类型的指针，用于存储矩阵 A 的数据
    const uint8_t* RESTRICT B,
    // B 数组，存储限制在 uint8_t 类型的指针，用于存储矩阵 B 的数据
    const BFloat16* RESTRICT ScaleAndZeros,
    // ScaleAndZeros 数组，存储限制在 BFloat16 类型的指针，用于存储缩放和零点的数据
    BFloat16* RESTRICT C,
    // C 数组，存储限制在 BFloat16 类型的指针，用于存储矩阵 C 的数据
    int lda,
    // lda，整数，表示矩阵 A 的列数
    int ldb,
    // ldb，整数，表示矩阵 B 的列数
    int ldc,
    // ldc，整数，表示矩阵 C 的列数
    int K,
    // K，整数，表示矩阵的维度大小
    int BLOCK_K) {
    // BLOCK_K，整数，表示块的大小

  constexpr int ROWS = BLOCK_M;
  // ROWS，常量整数，表示块的行数
  constexpr int COLS = BLOCK_N / 8;
  // COLS，常量整数，表示块的列数除以 8

  const int PREFETCH_SIZE_K = 16 * 4;
  // PREFETCH_SIZE_K，常量整数，表示预取数据的大小
  const int PREFETCH_SIZE_KB = (PREFETCH_SIZE_K + BLOCK_K - 1) / BLOCK_K;
  // PREFETCH_SIZE_KB，常量整数，表示块 K 的预取大小

  // number of blocks on K
  const int KB = K / BLOCK_K;
  // KB，整数，表示块 K 的数量

  __m256 va;
  // va，AVX 寄存器类型，用于存储 8 个单精度浮点数
  __m256 vb[COLS];
  // vb 数组，AVX 寄存器类型数组，用于存储 8 个单精度浮点数
  __m256 vc[ROWS * COLS];
  // vc 数组，AVX 寄存器类型数组，用于存储 8 个单精度浮点数的块
  __m256 scale[COLS];
  // scale 数组，AVX 寄存器类型数组，用于存储 8 个单精度浮点数的缩放因子
  __m256 zero[COLS];
  // zero 数组，AVX 寄存器类型数组，用于存储 8 个单精度浮点数的零点

  static const __m256i idx1 = _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7);
  // idx1，AVX 寄存器类型，用于控制转置操作的索引顺序

  // offset to shift from range [0, 15] to [-8, 7]
  const __m256 offset = _mm256_set1_ps(-8.0f);
  // offset，AVX 寄存器类型，用于偏移转置操作的范围

  // load scale and zero point
  auto load_scale_and_zeros = [&](int i, int _kb) {
    // load_scale_and_zeros，lambda 函数，加载缩放和零点数据
    // i，整数，表示索引
    // _kb，整数，表示块 K 的索引

    // load 2x bfloat16 vector
    __m256i t = _mm256_loadu_si256((__m256i*)(ScaleAndZeros + _kb * ldc * 2 + 16 * i));
    // t，AVX 寄存器类型，加载 BFloat16 向量数据

    if (_kb + PREFETCH_SIZE_KB < KB) {
      _mm_prefetch(ScaleAndZeros + (_kb + PREFETCH_SIZE_KB) * ldc * 2 + 16 * i, _MM_HINT_T0);
    }
    // 预取下一个缩放和零点数据

    // convert to 2x f32 vector
    __m256 a, b;
    // a、b，AVX 寄存器类型，用于存储转换后的单精度浮点数向量
    vec::cvtbf16_fp32(t, a, b);
    // 转换 BFloat16 到单精度浮点数

    // transpose scale_and_zero from {8, 2} to {2, 8}
    // inputs:
    //   a: {s0, z0, s1, z1, s2, z2, s3, z3}
    //   b: {s4, z4, s5, z5, s6, z6, s7, z7}
    // output:
    //   scale: {s0, s1, s2, s3, s4, s5, s6, s7}
    //   zero:  {z0, z1, z2, z3, z4, z5, z6, z7}
    a = _mm256_permutevar8x32_ps(a, idx1);
    b = _mm256_permutevar8x32_ps(b, idx1);
    scale[i] = _mm256_permute2f128_ps(a, b, 0b0100000);
    zero[i] = _mm256_permute2f128_ps(a, b, 0b0110001);

    // zero = -8 * scale + zero
    zero[i] = _mm256_fmadd_ps(scale[i], offset, zero[i]);
    // 计算零点数据
  };

  auto loadc = [&](auto i) {
    vc[i] = _mm256_setzero_ps();
    // 设置 VC 寄存器为零
  };
  c10::ForcedUnroll<ROWS * COLS>{}(loadc);
  // 循环展开加载 C 矩阵

  auto compute = [&, COLS](auto i, int k) {
    // compute，lambda 函数，计算操作
    // i，整数，表示索引
    // k，整数，表示块 K 的索引

    constexpr int row = i / COLS;
    // row，常量整数，表示行索引
    constexpr int col = i % COLS;
    // col，常量整数，表示列索引

    if constexpr (col == 0) {
      float aa = static_cast<float>(A[row * lda + k]);
      // aa，单精度浮点数，从 A 矩阵加载数据
      if (k + PREFETCH_SIZE_K < K) {
        _mm_prefetch(A + row * lda + k + PREFETCH_SIZE_K, _MM_HINT_T0);
      }
      // 预取下一个 A 矩阵的数据
      va = _mm256_set1_ps(aa);
      // 设置 VA 寄存器为 AA 的值
    }
    // 如果当前行数为0
    if constexpr (row == 0) {
      // 如果列数为4
      if constexpr (COLS == 4) {
        // 当 BLOCK_N = 32 时，处理每一行的数据
        // 如果当前列数为0
        if constexpr (col == 0) {
          // 创建一个掩码，用于提取低4位的值
          __m256i mask = _mm256_set1_epi32(0xF);
          // 加载 B 矩阵中第 k 行的数据
          __m128i b4 = _mm_loadu_si128((__m128i*)(B + k * ldb));
          // 如果 k + PREFETCH_SIZE_K 小于 K，则预取下一行数据
          if (k + PREFETCH_SIZE_K < K) {
            _mm_prefetch(B + (k + PREFETCH_SIZE_K) * ldb, _MM_HINT_T0);
          }

          // 将每个8位整数扩展为32位整数，并进行掩码操作，同时乘以比例系数并加上偏置
          __m256i b32 = _mm256_cvtepu8_epi32(b4);
          vb[0] = _mm256_cvtepi32_ps(_mm256_and_si256(b32, mask));
          vb[0] = _mm256_fmadd_ps(vb[0], scale[0], zero[0]);
          vb[2] = _mm256_cvtepi32_ps(_mm256_srli_epi32(b32, 4));
          vb[2] = _mm256_fmadd_ps(vb[2], scale[2], zero[2]);

          // 对另一组数据进行类似操作，通过字节交换得到
          b32 = _mm256_cvtepu8_epi32(_mm_shuffle_epi32(b4, _MM_SHUFFLE(3, 2, 3, 2)));
          vb[1] = _mm256_cvtepi32_ps(_mm256_and_si256(b32, mask));
          vb[1] = _mm256_fmadd_ps(vb[1], scale[1], zero[1]);
          vb[3] = _mm256_cvtepi32_ps(_mm256_srli_epi32(b32, 4));
          vb[3] = _mm256_fmadd_ps(vb[3], scale[3], zero[3]);
        }
      } else {
        // 如果 COLS 不等于 4，且当前列数是偶数
        if constexpr (col % 2 == 0) {
          // 将每个64位整数（16个 int4）进行去量化
          __m128i b8 = conver_int4_to_int8(B + k * ldb + col * 4);
          // 分别设置64位整数的两个值
          __m128i b8_val0 = _mm_set1_epi64x(_mm_extract_epi64(b8, 0));
          __m128i b8_val1 = _mm_set1_epi64x(_mm_extract_epi64(b8, 1));
          // 如果 k + PREFETCH_SIZE_K 小于 K，则预取下一行数据
          if (k + PREFETCH_SIZE_K < K) {
            _mm_prefetch(B + (k + PREFETCH_SIZE_K) * ldb + col * 4, _MM_HINT_T0);
          }

          // 将每个8位整数扩展为32位整数，并乘以比例系数并加上偏置
          vb[col] = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(b8_val0));
          vb[col] = _mm256_fmadd_ps(vb[col], scale[col], zero[col]);
          vb[col + 1] = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(b8_val1));
          vb[col + 1] = _mm256_fmadd_ps(vb[col + 1], scale[col + 1], zero[col + 1]);
        }
      }
    }

    // 计算在 C 矩阵中的索引
    constexpr int idx = row * COLS + col;
    // 将计算得到的结果累加到 C 矩阵中的对应位置
    vc[idx] = _mm256_fmadd_ps(va, vb[col], vc[idx]);
  };
  // 对每个 k 进行循环计算
  for (int k = 0, kb = 0; k < K; ++k) {
    // 如果 k 是块的起始位置，根据 BLOCK_K 强制展开
    if (is_block_start(k, BLOCK_K)) {
        c10::ForcedUnroll<COLS>{}(load_scale_and_zeros, kb++);
    }
    // 根据 ROWS 和 COLS 强制展开计算
    c10::ForcedUnroll<ROWS * COLS>{}(compute, k);
  }

  // 将结果存储到 C 矩阵中
  auto storec = [&](auto i) {
    // 计算当前索引对应的行号和列号
    constexpr int row = i / COLS;
    constexpr int col = i % COLS;
    // 如果当前列数是偶数
    if constexpr (col % 2 == 0) {
      // 将浮点数向量 vc 转换为 bf16 类型的整数向量，然后存储到 C 矩阵中
      __m256i ci = vec::cvtfp32_bf16(vc[row * COLS + col], vc[row * COLS + col + 1]);
      _mm256_storeu_si256((__m256i*)(C + row * ldc + col * 8), ci);
    }
  };
  // 根据 ROWS * COLS 强制展开存储过程
  c10::ForcedUnroll<ROWS * COLS>{}(storec);
#endif

#if !defined(C10_MOBILE) && defined(__aarch64__)
#include <arm_neon.h>

// 将指针指向的内存解释为 float32x4x2_t 类型的数据，并返回
inline float32x4x2_t load_as_float32x4x2(const Half* ptr) {
  // 从内存中加载两组 float16x4x2_t 类型数据，转换为 float32x4_t 类型
  float16x4x2_t f16_val = vld2_f16(reinterpret_cast<const float16_t *>(ptr));
  auto val_low = vcvt_f32_f16(f16_val.val[0]);  // 将第一组数据转换为 float32x4_t 类型
  auto val_high = vcvt_f32_f16(f16_val.val[1]);  // 将第二组数据转换为 float32x4_t 类型
  return {val_low, val_high};  // 返回转换后的结果作为 float32x4x2_t 类型
}

// 将 float32x4_t 类型数据存储到指针指向的内存中
inline void store_float32x4(Half* ptr, float32x4_t val) {
    // 将 float32x4_t 类型数据转换为 float16x4_t 类型，然后存储到内存中
    vst1_f16(reinterpret_cast<float16_t*>(ptr), vcvt_f16_f32(val));
}

// 将指针指向的内存解释为 float32x4x2_t 类型的数据，并返回
inline float32x4x2_t load_as_float32x4x2(const BFloat16* ptr) {
  int32x4_t shift = vdupq_n_s32(16);  // 创建所有元素为 16 的 int32x4_t 类型向量
  uint16x4x2_t u16_val = vld2_u16(reinterpret_cast<const uint16_t *>(ptr));  // 从内存中加载两组 uint16x4x2_t 类型数据
  uint32x4_t int_low = vmovl_u16(u16_val.val[0]);  // 将第一组 uint16x4_t 类型数据转换为 uint32x4_t 类型
  uint32x4_t int_high = vmovl_u16(u16_val.val[1]);  // 将第二组 uint16x4_t 类型数据转换为 uint32x4_t 类型
  // 对转换后的 uint32x4_t 类型数据进行移位和重新解释为 float32x4x2_t 类型，然后返回
  return {vreinterpretq_f32_u32(vshlq_u32(int_low, shift)), vreinterpretq_f32_u32(vshlq_u32(int_high, shift))};
}

// 将 float32x4_t 类型数据存储到指针指向的内存中
inline void store_float32x4(BFloat16* ptr, float32x4_t val) {
    int32x4_t shift = vdupq_n_s32(-16);  // 创建所有元素为 -16 的 int32x4_t 类型向量
    uint32x4_t uint32_val = vshlq_u32(vreinterpretq_u32_f32(val), shift);  // 将 float32x4_t 类型数据转换为 uint32x4_t 类型并进行左移位
    // 将左移位后的 uint32x4_t 类型数据转换为 uint16x4_t 类型并存储到内存中
    vst1_u16(reinterpret_cast<uint16_t*>(ptr), vmovn_u32(uint32_val));
}

// 将指针指向的内存解释为 float32x4x2_t 类型的数据，并返回
inline float32x4x2_t load_as_float32x4x2(const float* ptr) {
  return vld2q_f32(ptr);  // 从内存中加载两组 float32x4_t 类型数据，返回为 float32x4x2_t 类型
}

// 将 float32x4_t 类型数据存储到指针指向的内存中
inline void store_float32x4(float* ptr, float32x4_t val) {
    vst1q_f32(ptr, val);  // 将 float32x4_t 类型数据存储到内存中
}

// 模板函数，实现小矩阵乘法的核心计算
template <int BLOCK_M, int BLOCK_N, typename T>
inline void tinygemm_kernel_(
    const T* RESTRICT A,
    const uint8_t* RESTRICT B,
    const T* RESTRICT ScaleAndZeros,
    T* RESTRICT C,
    int lda,
    int ldb,
    int ldc,
    int K,
    int BLOCK_K) {
  int16_t shift_vals[4] = {0, -4, -8, -12};  // 定义包含四个元素的 int16_t 类型数组
  int16x4_t shifts = vld1_s16(shift_vals);  // 将 shift_vals 数组加载到 int16x4_t 类型向量
  int16x4_t offs = vdup_n_s16(8);  // 创建所有元素为 8 的 int16x4_t 类型向量
  uint16x4_t mask = vdup_n_u16(0x0F);  // 创建所有元素为 0x0F 的 uint16x4_t 类型向量
  for (const auto m : c10::irange(BLOCK_M)) {  // 对 BLOCK_M 进行迭代
    // 对每个块循环处理，每次增加16个元素
    for (int n = 0; n < BLOCK_N; n+= 16) {
      // 初始化存储每列计算结果的向量
      float32x4_t c_val[4];
      // 初始化缩放因子和零向量的向量数组
      float32x4_t scales[4], zeros[4];
      // 对四个元素进行强制展开，将每个元素初始化为0.0
      c10::ForcedUnroll<4>{}([&](auto i) {
          c_val[i] = vdupq_n_f32(0.0);
      });
      // 遍历K的范围
      for (const auto k : c10::irange(K)) {
        // 获取A矩阵中的值，并转换为float32x4_t类型
        const auto a_val = vdupq_n_f32(static_cast<float>(A[m * lda + k]));
        // 如果k是块的起始位置，执行以下操作
        if (is_block_start(k, BLOCK_K)) {
          // 计算块索引
          int kb = k / BLOCK_K;
          // 对四个元素进行强制展开，加载缩放因子和零向量
          c10::ForcedUnroll<4>{}([&](auto i) {
            // 加载ScaleAndZeros数组中的数据作为float32x4x2类型
            auto scales_and_zeros = load_as_float32x4x2(ScaleAndZeros + kb * ldc * 2 + n * 2 + i * 8);
            // 将加载的数据分别赋值给scales和zeros向量数组
            scales[i] = scales_and_zeros.val[0];
            zeros[i] = scales_and_zeros.val[1];
          });
        }
        // 对四个元素进行强制展开，处理B矩阵中的数据
        c10::ForcedUnroll<4>{}([&](auto i) {
          // 从B矩阵中加载一个16位整数，并将其转换为uint16_t类型
          uint16_t b_pack = reinterpret_cast<const uint16_t*>(B + k * ldb + n / 2)[i];
          // 对加载的数据应用位移和掩码操作，得到一个16位整数向量
          uint16x4_t b_masked = vand_u16(vshl_u16(vdup_n_u16(b_pack), shifts), mask);
          // 将16位整数向量转换为int16x4_t类型，并进行偏移调整
          int16x4_t b_ints = vsub_s16(vreinterpret_s16_u16(b_masked), offs);
          // 将int16x4_t类型的整数向量转换为float32x4_t类型的浮点数向量
          float32x4_t b_vals = vcvtq_f32_s32(vmovl_s16(b_ints));
          // 应用缩放因子和零向量，将其与A矩阵值相乘并累加到c_val中
          b_vals = vaddq_f32(zeros[i], vmulq_f32(scales[i], b_vals));
          c_val[i] = vfmaq_f32(c_val[i], b_vals, a_val);
        });
      }
      // 对四个元素进行强制展开，将计算结果存储到C矩阵中
      c10::ForcedUnroll<4>{}([&](auto i) {
        store_float32x4(C + m * ldc + n + i * 4, c_val[i]);
      });
    }
  }
// 根据模板参数 BLOCK_N，选择不同的转换方法将 uint8_t 类型的输入数据转换为 float 类型
// 如果支持 AVX512 并且 BLOCK_N 等于 64，则使用 AVX512 指令集进行转换
// 否则，如果支持 AVX2 并且 BLOCK_N 等于 32，则使用 AVX2 指令集进行转换
// 如果以上条件都不满足，则使用非向量化的方法进行转换
template<int BLOCK_N>
inline float convert_int4_to_float(const uint8_t* b, int n) {
  // 定义一个固定的查找表，将 uint8_t 类型的输入转换为 float 类型
  static constexpr float lut[16] = {
    -8.0f, -7.0f, -6.0f, -5.0f,
    -4.0f, -3.0f, -2.0f, -1.0f,
    0.0f, 1.0f, 2.0f, 3.0f,
    4.0f, 5.0f, 6.0f, 7.0f
  };
  int index;
  
  // 根据不同的条件选择不同的转换方法
#if defined(CPU_CAPABILITY_AVX512) && !defined(_MSC_VER)
  if constexpr (BLOCK_N == 64) {
    // 计算当前块的索引并确定使用的具体值进行转换
    const int nb = n / BLOCK_N;
    n -= nb * BLOCK_N;
    if (n < 32) {
      auto val = b[nb * BLOCK_N / 2 + n];
      index = val & 0x0f;  // 取低 4 位作为索引
    } else {
      auto val = b[nb * BLOCK_N / 2 + (n - 32)];
      index = val >> 4;    // 取高 4 位作为索引
    }
  } else
#elif defined(CPU_CAPABILITY_AVX2) && !defined(_MSC_VER)
  if constexpr (BLOCK_N == 32) {
    // 计算当前块的索引并确定使用的具体值进行转换
    const int nb = n / BLOCK_N;
    n -= nb * BLOCK_N;
    if (n < 16) {
      auto val = b[nb * BLOCK_N / 2 + n];
      index = val & 0x0f;  // 取低 4 位作为索引
    } else {
      auto val = b[nb * BLOCK_N / 2 + (n - 16)];
      index = val >> 4;    // 取高 4 位作为索引
    }
  } else
#endif
  {
    // 如果不满足前两种情况，使用非向量化的方式进行转换
    const auto is_even = (n & 1) == 0;
    auto val = b[n / 2];
    index = is_even ? (val & 0x0F) : (val >> 4);  // 根据 n 的奇偶性确定使用的位数并作为索引
  }
  
  // 返回查找表中对应索引的值，完成数据类型的转换
  return lut[index];
}
    // 遍历BLOCK_N范围内的索引n，计算矩阵乘法中C矩阵的每列
    for (const auto n : c10::irange(BLOCK_N)) {
      // 初始化C矩阵中的当前列的值为0
      float c_val = 0;
      // 遍历K范围内的索引k，执行矩阵乘法的内积计算
      for (const auto k : c10::irange(K)) {
        // 根据索引k计算出块内索引kb，获取比例因子和零值
        int kb = k / BLOCK_K;
        const auto scale = static_cast<float>(ScaleAndZeros[kb * ldc * 2 + n * 2]);
        const auto zero = static_cast<float>(ScaleAndZeros[kb * ldc * 2 + n * 2 + 1]);
        // 从矩阵A中读取当前元素a_val
        const auto a_val = static_cast<float>(A[m * lda + k]);
        // 从矩阵B中读取并转换为float类型的当前元素b_val
        float b_val = convert_int4_to_float<BLOCK_N>(B + k * ldb, n);
        // 根据比例因子和零值对b_val进行缩放和平移
        b_val = b_val * scale + zero;

        // 计算C矩阵中当前元素的值，累加乘积结果
        c_val += a_val * b_val;
      }
      // 将计算得到的c_val赋值给C矩阵中的当前元素
      C[m * ldc + n] = c_val;
    }
  }
// 定义宏 LAUNCH_TINYGEMM_KERNEL，用于启动 tinygemm_kernel 函数的模板化实例化
#define LAUNCH_TINYGEMM_KERNEL(MB_SIZE, NB_SIZE)                 \
  tinygemm_kernel<MB_SIZE, NB_SIZE>(                             \
      A_ptr, B_ptr, S_ptr, C_ptr,                                \
      K, NB_SIZE / 2, N, K, BLOCK_K);

// 定义宏 LAUNCH_TINYGEMM_NB_SIZE，根据 nb_size 的值选择不同的 NB_SIZE 来调用 LAUNCH_TINYGEMM_KERNEL 宏
#define LAUNCH_TINYGEMM_NB_SIZE(MB_SIZE)                         \
  switch (nb_size) {                                             \
    case 16:                                                     \
      LAUNCH_TINYGEMM_KERNEL(MB_SIZE, 16);                       \
      break;                                                     \
    case 32:                                                     \
      LAUNCH_TINYGEMM_KERNEL(MB_SIZE, 32);                       \
      break;                                                     \
    case 48:                                                     \
      LAUNCH_TINYGEMM_KERNEL(MB_SIZE, 48);                       \
      break;                                                     \
    case 64:                                                     \
      LAUNCH_TINYGEMM_KERNEL(MB_SIZE, 64);                       \
      break;                                                     \
    default:                                                     \
      TORCH_CHECK(false, "Unsupported n block size: ", nb_size); \
      break;                                                     \
  }

// 函数 weight_to_int4pack_kernel 的注释：
//
// NB: int4 weight pack (with BLOCK_N 64)
//   weight (int32): {N/64, 64, K}
//   packed (uint8): {N/64, K, 32}
//
// 1. avx512 packed format:
//   当 N 为 64 时，每次进行 256 位解包，将 Lane0 与 Lane2、Lane1 与 Lane3 组合，
//   因为只能在 128 位基础上进行位移。
//
//   weight:
//     [Lane0] N0...15:  {a00, a01, a02, ...}
//     [Lane1] N16...31: {a10, a11, a12, ...}
//     [Lane2] N32...47: {a20, a21, a22, ...}
//     [Lane3] N48...63: {a30, a31, a32, ...}
//
//  packed:
//     [Lane02] N0...31:  {a20|a00, a21|a01, a22|a02, ...}
//     [Lane13] N32...63: {a30|a10, a31|a11, a32|a12, ...}
//
//  注意：当 N 为 16、32 或 48 时，使用 64 位格式进行打包。
//
// 2. avx2 packed format:
//   当 N 为 32 时，每次进行 128 位解包。
//
//   weight:
//     [Lane0] N0...15:  { a0,  a1,  a2, ...}
//     [Lane1] N16...32: {a16, a17, a18, ...}
//
//  packed:
//    [Lane01] N0...32: {a16|a0, a17|a1, a18|a2, ...}
//
//  注意：当 N 为 16 时，使用 64 位格式进行打包。
//
// 3 non-vectorized packed format:
//   每次进行 64 位解包。
//
//   weight: {a0, a1, a2, a3, ..., a14, a15}
//   packed: {a1|a0, a3, a2, ..., a15|a14}
//
void weight_to_int4pack_kernel(
    const Tensor& weight_packed,
    const Tensor& weight,
    int N, int K) {


// 定义函数参数 N 和 K，分别表示矩阵的行数和列数
auto weight_packed_data = reinterpret_cast<uint8_t*>(weight_packed.data_ptr());
// 获取压缩后权重数据的指针，并转换为 uint8_t* 类型
const auto weight_data = weight.data_ptr<int32_t>();
// 获取原始权重数据的指针，并转换为 int32_t* 类型

// 64 for avx512 and 32 for avx2/non-vectorized
// 根据硬件支持的向量化指令集不同，设置不同的块大小
constexpr int BLOCK_N = vec::Vectorized<float>::size() * 4;
// BLOCK_N 表示每个处理块的大小，这里乘以 4 是因为 AVX512 指令集的块大小是 AVX 指令集的四倍
const int NB =  (N + BLOCK_N - 1) / BLOCK_N;
// 将矩阵行数 N 分割成 NB 个块，每个块的大小为 BLOCK_N

// parallel on NB blocks
// 在 NB 个块上并行执行
at::parallel_for(0, NB, 0, [&](int begin, int end) {
// 使用 ATen 库提供的并行函数 parallel_for 迭代处理从 begin 到 end 的块
for (const auto i : c10::irange(begin, end)) {
// 循环迭代处理每个块的索引 i

int nb_size = std::min(BLOCK_N, N - i * BLOCK_N);
// 计算当前块的实际大小，不能超过矩阵剩余的行数

const int32_t* src = weight_data + i * BLOCK_N * K;
// 计算当前块在原始权重数据中的起始位置
uint8_t* dst = weight_packed_data + i * K * BLOCK_N / 2;
// 计算当前块在压缩后数据中的起始位置

for (const auto k : c10::irange(K)) {
// 循环处理每列数据
#if defined(CPU_CAPABILITY_AVX512) && !defined(_MSC_VER)
        // 如果编译器支持 AVX512 并且不是 MSVC 编译器
        if (nb_size == BLOCK_N) {
          // 对于 nb_size 等于 BLOCK_N（64），执行以下循环
          for (const auto d : c10::irange(16)) {
            // 按照 AVX512 的格式打包数据
            int32_t val0 = src[(d +  0) * K + k];
            int32_t val1 = src[(d + 16) * K + k];
            int32_t val2 = src[(d + 32) * K + k];
            int32_t val3 = src[(d + 48) * K + k];

            uint8_t packed02 = (((uint8_t)(val2) << 4)) | ((uint8_t)(val0));
            uint8_t packed13 = (((uint8_t)(val3) << 4)) | ((uint8_t)(val1));

            // 将打包后的数据存入目标数组 dst
            dst[k * 32 + d] = packed02;
            dst[k * 32 + 16 + d] = packed13;
          }
        } else {
          // 对于 nb_size 为 16, 32, 48 的情况
          // 使用 AVX512 打包数据
          for (int n = 0; n < nb_size; n += 2) {
            int32_t val0 = src[n * K + k];
            int32_t val1 = src[n * K + K + k];

            uint8_t packed = (((uint8_t)(val1) << 4)) | ((uint8_t)(val0));
            // 将打包后的数据存入目标数组 dst
            dst[k * nb_size / 2 + n / 2] = packed;
          }
        }
#elif defined(CPU_CAPABILITY_AVX2) && !defined(_MSC_VER)
        // 如果编译器支持 AVX2 并且不是 MSVC 编译器
        if (nb_size == BLOCK_N) {
          // 对于 nb_size 等于 32 的情况
          // 使用 AVX2 打包数据
          for (const auto d : c10::irange(16)) {
            int32_t val0 = src[(d + 0) * K + k];
            int32_t val1 = src[(d + 16) * K + k];

            uint8_t packed01 = (((uint8_t)(val1) << 4)) | ((uint8_t)(val0));
            // 将打包后的数据存入目标数组 dst
            dst[k * 16 + d] = packed01;
          }
        } else {
          // 对于 nb_size 等于 16 的情况
          // 使用 AVX2 打包数据
          for (int n = 0; n < nb_size; n += 2) {
            int32_t val0 = src[n * K + k];
            int32_t val1 = src[n * K + K + k];

            uint8_t packed = (((uint8_t)(val1) << 4)) | ((uint8_t)(val0));
            // 将打包后的数据存入目标数组 dst
            dst[k * nb_size / 2 + n / 2] = packed;
          }
        }
#else
        // 如果没有特定的 CPU 支持，或者是 MSVC 编译器
        // 使用通用的打包方式
        for (int n = 0; n < nb_size; n += 2) {
          int32_t val0 = src[n * K + k];
          int32_t val1 = src[n * K + K + k];

          uint8_t packed = (((uint8_t)(val1) << 4)) | ((uint8_t)(val0));
          // 将打包后的数据存入目标数组 dst
          dst[k * nb_size / 2 + n / 2] = packed;
        }
#endif
      }
    }
  });
}

template<typename T>
void int4pack_mm_kernel_(
    const Tensor& C,
    const Tensor& A,
    const Tensor& B,
    int qGroupSize,
    const Tensor& qScaleAndZeros,
    int N, int K) {

  const auto* A_data = A.const_data_ptr<T>();
  const auto* B_data = reinterpret_cast<const uint8_t*>(B.const_data_ptr());
  auto* C_data = C.data_ptr<T>();
  const auto* S_data = qScaleAndZeros.const_data_ptr<T>();

  int M = A.size(0);

  constexpr int BLOCK_M = 4;
  // 根据 AVX512 和 AVX2 的支持情况确定 BLOCK_N 的大小
  constexpr int BLOCK_N = vec::Vectorized<float>::size() * 4;
  // 根据 qGroupSize 确定 BLOCK_K 的大小
  const int BLOCK_K = qGroupSize;

  // 计算块的数量
  const int MB = (M + BLOCK_M - 1) / BLOCK_M;
  const int NB = (N + BLOCK_N - 1) / BLOCK_N;

  // 并行处理每个块
  at::parallel_for(0, MB * NB, 0, [&](int begin, int end) {
    int mb{0}, nb{0};
    // 初始化数据索引
    data_index_init(begin, mb, MB, nb, NB);
    // 使用范围循环迭代整数 i，范围是 [begin, end)
    for (C10_UNUSED const auto i : c10::irange(begin, end)) {
      // 计算当前块的起始行号
      int mb_start = mb * BLOCK_M;
      // 计算当前块的行数，最多为 BLOCK_M，但不能超过剩余的行数 M - mb_start
      int mb_size = std::min(BLOCK_M, M - mb_start);
      // 计算当前块的起始列号
      int nb_start = nb * BLOCK_N;
      // 计算当前块的列数，最多为 BLOCK_N，但不能超过剩余的列数 N - nb_start
      int nb_size = std::min(BLOCK_N, N - nb_start);

      // 获取矩阵 A 的指针，指向当前块的数据起始位置
      const auto* A_ptr = A_data + mb_start * K;
      // 获取矩阵 B 的指针，指向当前块的数据起始位置，每个元素占据 K/2 个字节
      const auto* B_ptr = B_data + nb_start * K / 2;
      // 获取矩阵 S 的指针，指向当前块的数据起始位置，每个元素占据 2 个字节
      const auto* S_ptr = S_data + nb_start * 2;
      // 获取矩阵 C 的指针，指向当前块的结果数据起始位置
      auto* C_ptr = C_data + mb_start * N + nb_start;

      // 根据当前块的行数选择合适的内核启动函数 LAUNCH_TINYGEMM_NB_SIZE
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
          // 如果当前块行数不支持，则抛出错误信息
          TORCH_CHECK(false, "Unsupported m block size: ", mb_size);
      }

      // 移动到下一个索引位置
      data_index_step(mb, MB, nb, NB);
    }
  });
} // 匿名命名空间的结束

void int4pack_mm_kernel(
    const Tensor& C,  // 第一个参数，表示结果张量 C
    const Tensor& A,  // 第二个参数，表示输入张量 A
    const Tensor& B,  // 第三个参数，表示输入张量 B
    int qGroupSize,   // 第四个参数，表示量化组大小
    const Tensor& qScaleAndZeros,  // 第五个参数，包含量化参数和零点
    int N, int K) {   // 最后两个参数，表示矩阵乘法的维度 N 和 K
  if (C.scalar_type() == kBFloat16) {  // 如果 C 的数据类型是 kBFloat16
    调用具体的模板函数 int4pack_mm_kernel_<BFloat16> 处理
    int4pack_mm_kernel_<BFloat16>(C, A, B, qGroupSize, qScaleAndZeros, N, K);
  } else if (C.scalar_type() == kHalf) {  // 如果 C 的数据类型是 kHalf
    调用具体的模板函数 int4pack_mm_kernel_<Half> 处理
    int4pack_mm_kernel_<Half>(C, A, B, qGroupSize, qScaleAndZeros, N, K);
  } else {  // 否则
    调用具体的模板函数 int4pack_mm_kernel_<float> 处理
    int4pack_mm_kernel_<float>(C, A, B, qGroupSize, qScaleAndZeros, N, K);
  }
}

} // 匿名命名空间的结束

ALSO_REGISTER_AVX512_DISPATCH(weight_to_int4pack_stub, &weight_to_int4pack_kernel);  // 注册 AVX512 的调度函数 weight_to_int4pack_kernel
ALSO_REGISTER_AVX512_DISPATCH(int4pack_mm_stub, &int4pack_mm_kernel);  // 注册 AVX512 的调度函数 int4pack_mm_kernel

} // at::native 命名空间的结束
```