# `.\pytorch\aten\src\ATen\native\quantized\cpu\qnnpack\src\q8conv\4x4c2-sse2.c`

```
/*
 * 版权所有（c）Facebook公司及其关联公司。
 * 保留所有权利。
 *
 * 此源代码在源目录中的LICENSE文件中使用BSD风格许可证许可。
 */

#include <immintrin.h>

#include <qnnpack/q8conv.h>
#include <requantization/runtime-sse2.h>

// 定义一个 SSE2 架构下的卷积函数，处理4x4块，每通道2个输入通道
void pytorch_q8conv_ukernel_4x4c2__sse2(
    size_t mr,
    size_t nr,
    size_t kc,
    size_t ks,
    const uint8_t** restrict a,           // 输入数据指针的数组
    const void* restrict w,               // 权重数据的指针
    uint8_t* restrict c,                  // 输出数据的指针
    size_t c_stride,
    size_t output_channel_index,
    const union pytorch_qnnp_conv_quantization_params
        quantization_params[RESTRICT_STATIC 1]) {  // 量化参数结构体数组
  // 加载权重的第一个128位数据到寄存器
  __m128i vacc0x0123 = _mm_loadu_si128((const __m128i*)w);
  // 复制第一个128位数据到其他寄存器
  __m128i vacc1x0123 = vacc0x0123;
  __m128i vacc2x0123 = vacc0x0123;
  __m128i vacc3x0123 = vacc0x0123;
  // 更新权重指针，使其指向下一个128位数据
  w = (const void*)((uintptr_t)w + 16);

  // 加载输入的零点偏移量到寄存器
  const __m128i va_zero_point = _mm_load_si128(
      (const __m128i*)quantization_params->sse2.input_zero_point);
  // 加载输出通道对应的4个卷积核零点偏移量到变量
  const int16_t vb_zero_point_0 =
    quantization_params->sse2.kernel_zero_points[output_channel_index];
  const int16_t vb_zero_point_1 =
      quantization_params->sse2.kernel_zero_points[output_channel_index + 1];
  const int16_t vb_zero_point_2 =
      quantization_params->sse2.kernel_zero_points[output_channel_index + 2];
  const int16_t vb_zero_point_3 =
      quantization_params->sse2.kernel_zero_points[output_channel_index + 3];
  // 组装卷积核零点偏移量为一个128位寄存器
  const __m128i vb_zero_point = _mm_set_epi16(vb_zero_point_3,
                                              vb_zero_point_3,
                                              vb_zero_point_2,
                                              vb_zero_point_2,
                                              vb_zero_point_1,
                                              vb_zero_point_1,
                                              vb_zero_point_0,
                                              vb_zero_point_0
                                              );
  // 设置一个全为零的128位寄存器
  const __m128i vzero = _mm_setzero_si128();
  do {
    // 分别从输入数据数组中加载4个指针，每个指针指向一个输入通道的数据
    const uint8_t* restrict a0 = *a++;
    const uint8_t* restrict a1 = *a++;
    const uint8_t* restrict a2 = *a++;
    const uint8_t* restrict a3 = *a++;

    // 设置一个循环计数器，用于处理卷积核
    size_t k = kc;
    }
  }
  } while (--ks != 0);

// 以 do-while 循环计算的反向递减。ks 是计数器。


  const __m128 vmultiplier =
      _mm_loadu_ps(&quantization_params->sse2.requantization_scales
          [output_channel_index]);

// 加载输出通道索引处的 SSE2 重新量化比例的四个单精度浮点数。


  vacc0x0123 = _mm_cvtps_epi32(
                _mm_mul_ps(
                  _mm_cvtepi32_ps(vacc0x0123),
                  vmultiplier
                  )
                );

// 使用 vmultiplier 对 vacc0x0123 中的四个单精度浮点数执行乘法和转换，然后将结果转换为整数型。


  vacc1x0123 = _mm_cvtps_epi32(
                _mm_mul_ps(
                  _mm_cvtepi32_ps(vacc1x0123),
                  vmultiplier
                  )
                );

// 对 vacc1x0123 中的四个单精度浮点数执行类似的乘法、转换和整数转换操作。


  vacc2x0123 = _mm_cvtps_epi32(
                _mm_mul_ps(
                  _mm_cvtepi32_ps(vacc2x0123),
                  vmultiplier
                  )
                );

// 对 vacc2x0123 中的四个单精度浮点数执行类似的乘法、转换和整数转换操作。


  vacc3x0123 = _mm_cvtps_epi32(
                _mm_mul_ps(
                  _mm_cvtepi32_ps(vacc3x0123),
                  vmultiplier
                  )
                );

// 对 vacc3x0123 中的四个单精度浮点数执行类似的乘法、转换和整数转换操作。


  const __m128i voutput_zero_point = _mm_load_si128(
      (const __m128i*)quantization_params->sse2.output_zero_point);

// 加载 SSE2 输出零点的值，作为整型的 128 位数据。


  const __m128i vacc01x0123 = _mm_adds_epi16(
      _mm_packs_epi32(vacc0x0123, vacc1x0123), voutput_zero_point);

// 将 vacc0x0123 和 vacc1x0123 中的整数转换成 16 位有符号整数，然后将其与 voutput_zero_point 相加。


  const __m128i vacc23x0123 = _mm_adds_epi16(
      _mm_packs_epi32(vacc2x0123, vacc3x0123), voutput_zero_point);

// 将 vacc2x0123 和 vacc3x0123 中的整数转换成 16 位有符号整数，然后将其与 voutput_zero_point 相加。


  __m128i vout = _mm_packus_epi16(vacc01x0123, vacc23x0123);

// 将 vacc01x0123 和 vacc23x0123 合并成一个 8 位无符号整数的向量。


  vout = _mm_min_epu8(
      vout,
      _mm_load_si128((const __m128i*)quantization_params->sse2.output_max));

// 将 vout 向量中的每个元素与 SSE2 输出最大值比较，取较小值。


  vout = _mm_max_epu8(
      vout,
      _mm_load_si128((const __m128i*)quantization_params->sse2.output_min));

// 将 vout 向量中的每个元素与 SSE2 输出最小值比较，取较大值。


  uint8_t* c0 = c;
  uint8_t* c1 = (uint8_t*)((uintptr_t)c0 + c_stride);
  if (mr < 2) {
    c1 = c0;
  }
  uint8_t* c2 = (uint8_t*)((uintptr_t)c1 + c_stride);
  if (mr <= 2) {
    c2 = c1;
  }
  uint8_t* c3 = (uint8_t*)((uintptr_t)c2 + c_stride);
  if (mr != 4) {
    c3 = c2;
  }

// 根据 mr 和 c_stride 设置四个指向输出缓冲区的指针 c0、c1、c2、c3，以处理输出的布局。


  if (nr == 4) {
    *((uint32_t*)c0) = (uint32_t)_mm_cvtsi128_si32(vout);
    *((uint32_t*)c1) = (uint32_t)_mm_cvtsi128_si32(_mm_srli_epi64(vout, 32));
    *((uint32_t*)c2) =
        (uint32_t)_mm_cvtsi128_si32(_mm_unpackhi_epi32(vout, vout));
    *((uint32_t*)c3) = (uint32_t)_mm_cvtsi128_si32(_mm_srli_si128(vout, 12));
  } else {
    typedef PYTORCH_QNNP_UNALIGNED uint16_t unaligned_uint16_t;
    if (nr >= 2) {
      *((unaligned_uint16_t*)c0) = (uint16_t)_mm_extract_epi16(vout, 0);
      c0 += 2;
      *((unaligned_uint16_t*)c1) = (uint16_t)_mm_extract_epi16(vout, 2);
      c1 += 2;
      *((unaligned_uint16_t*)c2) = (uint16_t)_mm_extract_epi16(vout, 4);
      c2 += 2;
      *((unaligned_uint16_t*)c3) = (uint16_t)_mm_extract_epi16(vout, 6);
      c3 += 2;
      vout = _mm_srli_epi32(vout, 16);
      nr -= 2;
    }
    if (nr != 0) {
      *((uint8_t*)c0) = (uint8_t)_mm_cvtsi128_si32(vout);
      *((uint8_t*)c1) = (uint8_t)_mm_extract_epi16(vout, 2);
      *((uint8_t*)c2) = (uint8_t)_mm_extract_epi16(vout, 4);
      *((uint8_t*)c3) = (uint8_t)_mm_extract_epi16(vout, 6);
    }
  }

// 根据 nr 的值，将 vout 中的数据打包成 32 位或 16 位无符号整数，并将它们存储到输出缓冲区 c0、c1、c2、c3 中，以完成数据的格式化输出。
}


注释：

这行代码是一个单独的右大括号 '}'，用于闭合之前的代码块或语句。在大多数编程语言中，大括号通常用于定义函数、循环、条件语句等的开始和结束位置。在这里，它表示一个代码块的结束。
```