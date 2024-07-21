# `.\pytorch\aten\src\ATen\native\quantized\cpu\qnnpack\src\requantization\gemmlowp-sse4.c`

```
/*
 * 包含必要的头文件，这些文件提供了程序运行所需的函数和类型声明
 */
#include <assert.h>
#include <stdint.h>

/*
 * 包含 smmintrin.h，该文件提供了使用 SSE 指令集的 SIMD 操作函数的声明
 */
#include <smmintrin.h>

/*
 * 包含特定于项目的自定义头文件和函数声明
 */
#include <fp16/bitcasts.h>
#include <qnnpack/requantization-stubs.h>

/*
 * 包含 gemmlowp-sse.h 头文件，该文件提供了 gemmlowp 库在 SSE 下的实现
 */
#include "gemmlowp-sse.h"

/*
 * 函数定义开始，该函数实现了通过 gemmlowp 库进行 SSE4 下的量化和反量化操作
 */
void pytorch_qnnp_requantize_gemmlowp__sse4(
    size_t n,                              // 输入数据的大小，必须是 16 的倍数
    const int32_t* input,                  // 输入数据的指针，指向待量化的整型数据数组
    float scale,                           // 量化的缩放因子
    uint8_t zero_point,                    // 量化的零点
    uint8_t qmin,                          // 输出量化值的最小限制
    uint8_t qmax,                          // 输出量化值的最大限制
    uint8_t* output) {                     // 输出量化后的数据指针，指向待填充的无符号字节数组
  assert(n % 16 == 0);                     // 断言输入数据大小是 16 的倍数
  assert(scale < 1.0f);                    // 断言缩放因子小于 1.0
  assert(scale >= 0x1.0p-32f);             // 断言缩放因子大于或等于 2^-32

  const uint32_t scale_bits = fp32_to_bits(scale);  // 将浮点数缩放因子转换为其二进制表示形式

  /*
   * 计算量化参数
   */
  const uint32_t multiplier =
      ((scale_bits & UINT32_C(0x007FFFFF)) | UINT32_C(0x00800000)) << 7;  // 计算乘法器
  const int32_t exponent = (fp32_to_bits(scale) >> 23) - 127 - 23 - 7;    // 计算指数
  const int32_t shift =
      -(32 /* using high 32 bits in VQRDMUL */ - 1 /* doubling in VQRDMUL */ +
        exponent);                          // 计算移位量

  /*
   * 创建 SIMD 寄存器常量，加载和设置量化所需的参数
   */
  const __m128i vmultiplier = _mm_set1_epi32(multiplier);  // 创建包含乘法器的 SIMD 寄存器
  const __m128i vzero_point = _mm_set1_epi16((short)(uint16_t)zero_point);  // 创建包含零点的 SIMD 寄存器
  const __m128i vqmin = _mm_set1_epi8((char)qmin);          // 创建包含 qmin 的 SIMD 寄存器
  const __m128i vqmax = _mm_set1_epi8((char)qmax);          // 创建包含 qmax 的 SIMD 寄存器

  /*
   * 循环处理输入数据，每次处理 16 个元素
   */
  for (; n != 0; n -= 16) {
    /*
     * 加载输入数据到 SIMD 寄存器
     */
    const __m128i x = _mm_loadu_si128((const __m128i*)input);
    const __m128i y = _mm_loadu_si128((const __m128i*)(input + 4));
    const __m128i z = _mm_loadu_si128((const __m128i*)(input + 8));
    const __m128i w = _mm_loadu_si128((const __m128i*)(input + 12));
    input += 16;

    /*
     * 使用 gemmlowp 库进行量化操作
     */
    const __m128i x_product = gemmlowp_sse_vqrdmulh_s32(x, vmultiplier);
    const __m128i y_product = gemmlowp_sse_vqrdmulh_s32(y, vmultiplier);
    const __m128i z_product = gemmlowp_sse_vqrdmulh_s32(z, vmultiplier);
    const __m128i w_product = gemmlowp_sse_vqrdmulh_s32(w, vmultiplier);

    /*
     * 使用 gemmlowp 库进行反量化操作
     */
    const __m128i x_scaled = gemmlowp_sse_rdivbypo2_s32(x_product, shift);
    const __m128i y_scaled = gemmlowp_sse_rdivbypo2_s32(y_product, shift);
    const __m128i z_scaled = gemmlowp_sse_rdivbypo2_s32(z_product, shift);
    const __m128i w_scaled = gemmlowp_sse_rdivbypo2_s32(w_product, shift);

    /*
     * 将量化后的结果打包成 8 位整数，并进行饱和操作
     */
    const __m128i xy_packed =
        _mm_adds_epi16(_mm_packs_epi32(x_scaled, y_scaled), vzero_point);
    const __m128i zw_packed =
        _mm_adds_epi16(_mm_packs_epi32(z_scaled, w_scaled), vzero_point);
    const __m128i xyzw_packed = _mm_packus_epi16(xy_packed, zw_packed);
    const __m128i xyzw_clamped =
        _mm_max_epu8(_mm_min_epu8(xyzw_packed, vqmax), vqmin);

    /*
     * 将结果存储回输出数组
     */
    _mm_storeu_si128((__m128i*)output, xyzw_clamped);
    output += 16;
  }
}
/*
 * 函数定义结束
 */
```