# `.\pytorch\aten\src\ATen\native\quantized\cpu\qnnpack\src\q8avgpool\up8xm-sse2.c`

```
/*
 * 包含必要的头文件，用于声明和定义所需的函数和数据类型
 */
#include <assert.h>

/*
 * 包含 SSE2 指令集的头文件，提供了 SIMD 操作的函数和宏定义
 */
#include <emmintrin.h>

/*
 * 包含 QNNPACK 中定义的 Q8 平均池化函数的头文件
 */
#include <qnnpack/q8avgpool.h>

/*
 * 定义了一个名为 pytorch_q8avgpool_ukernel_up8xm__sse2 的函数，
 * 用于处理 Q8 数据的平均池化操作，利用 SSE2 指令集加速计算
 */
void pytorch_q8avgpool_ukernel_up8xm__sse2(
    size_t n,  // 待处理数据的数量
    size_t ks, // 卷积核大小
    size_t kc, // 通道数（小于8）
    const uint8_t** input,  // 输入数据的指针数组
    const uint8_t* zero,    // 零值，用于补齐
    uint8_t* output,        // 输出数据的指针
    size_t input_increment, // 输入数据指针增量
    size_t output_increment, // 输出数据指针增量
    const union pytorch_qnnp_avgpool_quantization_params
        quantization_params[RESTRICT_STATIC 1]) {  // 量化参数结构体数组
  assert(n != 0);  // 断言：数据数量不为零
  assert(ks != 0); // 断言：卷积核大小不为零
  assert(kc < 8);  // 断言：通道数小于8

  /*
   * 加载偏置值到 __m128i 类型的向量 vbias 中
   */
  const __m128i vbias =
      _mm_load_si128((const __m128i*)&quantization_params->sse2.bias);
  
  /*
   * 将零值加载到 __m128i 类型的向量 vzero 中
   */
  const __m128i vzero = _mm_setzero_si128();
  
  /*
   * 加载缩放因子到 __m128 类型的向量 vscale 中
   */
  const __m128 vscale = _mm_loadu_ps(quantization_params->sse2.scale);

  /*
   * 开始处理数据循环
   */
  do {
    /*
     * 计算下一行输入数据的指针
     */
    const uint8_t** next_input =
        (const uint8_t**)((uintptr_t)input + input_increment);
    
    /*
     * 初始化累加器向量
     */
    __m128i vacc_lo = vbias;
    __m128i vacc_hi = vbias;

    /*
     * 处理卷积核大小的数据
     */
    size_t m = ks;
    do {
      /*
       * 加载输入数据，并向前偏移 kc 个字节
       */
      const uint8_t* i = *input++;
      i += kc;

      /*
       * 初始化输入数据向量 vi
       */
      __m128i vi = _mm_setzero_si128();

      /*
       * 根据 kc 的奇偶性逐步加载输入数据到 vi 中
       */
      if (kc & 1) {
        i -= 1;
        vi = _mm_cvtsi32_si128((int)(uint32_t)*i);
      }
      if (kc & 2) {
        vi = _mm_slli_epi32(vi, 16);
        i -= 2;
        vi = _mm_insert_epi16(vi, *((const uint16_t*)i), 0);
      }
      if (kc & 4) {
        i -= 4;
        vi = _mm_unpacklo_epi32(
            _mm_cvtsi32_si128((int)*((const uint32_t*)i)), vi);
      }

      /*
       * 将输入数据向量 vi 拆分成字节，并与 vzero 向量进行无符号拆分
       * 然后与累加器向量相加
       */
      const __m128i vxi = _mm_unpacklo_epi8(vi, vzero);
      vacc_lo = _mm_add_epi32(vacc_lo, _mm_unpacklo_epi16(vxi, vzero));
      vacc_hi = _mm_add_epi32(vacc_hi, _mm_unpackhi_epi16(vxi, vzero));
    } while (--m != 0);
    input = next_input;

    /*
     * 对累加器中的结果进行缩放操作，并转换为整数类型
     */
    const __m128 vacc_lo_f = _mm_mul_ps(_mm_cvtepi32_ps(vacc_lo), vscale);
    const __m128 vacc_hi_f = _mm_mul_ps(_mm_cvtepi32_ps(vacc_hi), vscale);

    const __m128i vscaled_lo = _mm_cvtps_epi32(vacc_lo_f);
    const __m128i vscaled_hi = _mm_cvtps_epi32(vacc_hi_f);

    /*
     * 将结果打包成 16 位整数，并加上输出的零点偏移量
     */
    __m128i vout = _mm_packs_epi32(vscaled_lo, vscaled_hi);
    vout = _mm_adds_epi16(
        vout,
        _mm_load_si128(
            (const __m128i*)quantization_params->sse2.output_zero_point));
    
    /*
     * 将结果向量 vout 打包成无符号 8 位整数，并进行饱和处理
     */
    vout = _mm_packus_epi16(vout, vout);
    vout = _mm_min_epu8(
        vout,
        _mm_load_si128((const __m128i*)quantization_params->sse2.output_max));
    vout = _mm_max_epu8(
        vout,
        _mm_load_si128((const __m128i*)quantization_params->sse2.output_min));

    /*
     * 根据 kc 的奇偶性将结果写入输出指针中，并更新输出指针
     */
    if (kc & 4) {
      *((uint32_t*)output) = (uint32_t)_mm_cvtsi128_si32(vout);
      output += 4;
      vout = _mm_srli_epi64(vout, 32);
    }
    if (kc & 2) {
      *((uint16_t*)output) = (uint16_t)_mm_extract_epi16(vout, 0);
      output += 2;
      vout = _mm_srli_epi32(vout, 16);
    }
    if (kc & 1) {
      *((uint8_t*)output) = (uint8_t)_mm_cvtsi128_si32(vout);
      output += 1;
    }
    output = (uint8_t*)((uintptr_t)output + output_increment);
  // 将 output 指针转换为 uintptr_t 类型，然后加上 output_increment 的值，
  // 最后将结果强制转换回 uint8_t* 类型，更新 output 指针的位置
  } while (--n != 0);
  // 循环，直到 n 减少为 0
}



# 这行代码是一个单独的右括号 '}'，用于闭合一个代码块或者表达式。
# 在大多数编程语言中，右括号用于结束代码块、函数定义、循环或条件语句等。
```