# `.\pytorch\aten\src\ATen\native\quantized\cpu\qnnpack\src\q8gemm\2x4c8-sse2.c`

```py
/*
 * 版权所有（c）Facebook公司及其关联公司。
 * 保留所有权利。
 *
 * 此源代码根据根目录中的LICENSE文件中的BSD样式许可证进行许可。
 */

#include <immintrin.h>  // 包含了使用SSE指令集的头文件

#include <qnnpack/q8gemm.h>  // 包含了QNNPACK库中的q8gemm函数声明
#include <requantization/runtime-sse2.h>  // 包含了SSE2运行时函数的声明

// 定义了一个内联函数，使用SSE指令集进行整数运算，将四个__m128i类型参数进行加和
static inline __m128i pytorch_sse_reduce4_i32(
    __m128i x,
    __m128i y,
    __m128i z,
    __m128i w) {
#if defined(__SSSE3__) && !defined(__ANDROID__)
  /* xxyy = ( y2 + y3, y0 + y1, x2 + x3, x0 + x1 ) */
  const __m128i xxyy = _mm_hadd_epi32(x, y);  // 对x和y进行水平加和操作
  /* zzww = ( w2 + w3, w0 + w1, z2 + z3, z0 + z1 ) */
  const __m128i zzww = _mm_hadd_epi32(z, w);  // 对z和w进行水平加和操作
  /* xyzw = ( w0 + w1 + w2 + w3, y0 + y1 + y2 + y3, z0 + z1 + z2 + z3, x0 + x1 + x2 + x3 ) */
  return _mm_hadd_epi32(xxyy, zzww);  // 对xxyy和zzww进行水平加和操作，返回最终结果
#else
  /* xzxz = ( z1 + z3, x1 + x3, z0 + z2, x0 + x2 ) */
  const __m128i xzxz =
      _mm_add_epi32(_mm_unpacklo_epi32(x, z), _mm_unpackhi_epi32(x, z));  // 对x和z进行解包和加和操作
  /* ywyw = ( w1 + w3, y1 + y3, w0 + w2, y0 + y2 ) */
  const __m128i ywyw =
      _mm_add_epi32(_mm_unpacklo_epi32(y, w), _mm_unpackhi_epi32(y, w));  // 对y和w进行解包和加和操作
  /* xyzw = ( w0 + w2 + w1 + w3, y0 + y2 + y1 + y3, z0 + z2 + z1 + z3, x0 + x2 + x1 + x3 ) */
  return _mm_add_epi32(
      _mm_unpacklo_epi32(xzxz, ywyw), _mm_unpackhi_epi32(xzxz, ywyw));  // 对xzxz和ywyw进行解包和加和操作，返回最终结果
#endif
}

// 定义了一个使用SSE2指令集的矩阵乘法函数，专门用于8位量化数据
void pytorch_q8gemm_ukernel_2x4c8__sse2(
    size_t mr,
    size_t nr,
    size_t k,
    const uint8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    uint8_t* restrict c,
    size_t c_stride,
    size_t output_channel_index,
    const union pytorch_qnnp_conv_quantization_params
        quantization_params[RESTRICT_STATIC 1]) {
  // 将权重w的前四个元素分别加载到__m128i类型的寄存器中
  __m128i vacc00 = _mm_cvtsi32_si128((int)((const int32_t*)w)[0]);
  __m128i vacc01 = _mm_cvtsi32_si128((int)((const int32_t*)w)[1]);
  __m128i vacc02 = _mm_cvtsi32_si128((int)((const int32_t*)w)[2]);
  __m128i vacc03 = _mm_cvtsi32_si128((int)((const int32_t*)w)[3]);
  // 复制前面加载的四个元素到另外四个寄存器中
  __m128i vacc10 = vacc00;
  __m128i vacc11 = vacc01;
  __m128i vacc12 = vacc02;
  __m128i vacc13 = vacc03;
  // 将指针w向后移动16字节，指向下一个权重数据
  w = (const void*)((uintptr_t)w + 16);

  // 加载矩阵a的第一行和第二行数据
  const uint8_t* a0 = a;
  const uint8_t* a1 = (const uint8_t*)((uintptr_t)a0 + a_stride);
  if (mr != 2) {
    // 如果mr不等于2，将第二行数据指针指向第一行数据，实现数据复用
    a1 = a0;
  }

  // 加载权重矩阵的列数据，根据nr的不同情况加载不同数量的列
  const uint8_t* b0 = w;
  const uint8_t* b1 = b0 + 8;
  if (nr < 2) {
    // 如果nr小于2，将第二列数据指针指向第一列数据，实现数据复用
    b1 = b0;
  }
  const uint8_t* b2 = b1 + 8;
  if (nr <= 2) {
    // 如果nr小于等于2，将第三列数据指针指向第二列数据，实现数据复用
    b2 = b1;
  }
  const uint8_t* b3 = b2 + 8;
  if (nr != 4) {
    // 如果nr不等于4，将第四列数据指针指向第三列数据，实现数据复用
    b3 = b2;
  }
}
    b3 = b2;
  }
  // 计算每个输出通道的步长，每个通道占8个元素的空间
  const size_t b_stride = nr * 8;

  // 加载输入的零点偏移量，作为 VA 的零点
  const __m128i va_zero_point = _mm_load_si128(
      (const __m128i*)quantization_params->sse2.input_zero_point);
  
  // 加载第一个输出通道的卷积核零点偏移量作为 VB 的零点
  const __m128i vb_zero_point_0 = _mm_set1_epi16(
      (int16_t)(uint16_t)quantization_params->sse2.kernel_zero_points[
        output_channel_index]);
  
  // 加载第二个输出通道的卷积核零点偏移量作为 VB 的零点
  const __m128i vb_zero_point_1 = _mm_set1_epi16(
      (int16_t)(uint16_t)quantization_params->sse2.kernel_zero_points[
        output_channel_index + 1]);
  
  // 加载第三个输出通道的卷积核零点偏移量作为 VB 的零点
  const __m128i vb_zero_point_2 = _mm_set1_epi16(
      (int16_t)(uint16_t)quantization_params->sse2.kernel_zero_points[
        output_channel_index + 2]);
  
  // 加载第四个输出通道的卷积核零点偏移量作为 VB 的零点
  const __m128i vb_zero_point_3 = _mm_set1_epi16(
      (int16_t)(uint16_t)quantization_params->sse2.kernel_zero_points[
        output_channel_index + 3]);
  
  // 设置一个全零的 128 位整数向量
  const __m128i vzero = _mm_setzero_si128();

  // 循环处理每8个元素，直到 k 小于 8
  for (; k >= 8; k -= 8) {
    // 加载并处理 a0 的低64位为 va0
    const __m128i va0 = _mm_loadl_epi64((const __m128i*)a0);
    // 对 va0 进行零点处理，得到 vxa0
    const __m128i vxa0 =
        sub_zero_point(_mm_unpacklo_epi8(va0, vzero), va_zero_point);
    a0 += 8;

    // 加载并处理 a1 的低64位为 va1
    const __m128i va1 = _mm_loadl_epi64((const __m128i*)a1);
    // 对 va1 进行零点处理，得到 vxa1
    const __m128i vxa1 =
        sub_zero_point(_mm_unpacklo_epi8(va1, vzero), va_zero_point);
    a1 += 8;

    // 加载并处理 b0 的低64位为 vb0
    const __m128i vb0 = _mm_loadl_epi64((const __m128i*)b0);
    // 对 vb0 进行卷积核零点处理，得到 vxb0
    const __m128i vxb0 =
        _mm_sub_epi16(_mm_unpacklo_epi8(vb0, vzero), vb_zero_point_0);
    b0 += b_stride;

    // 加载并处理 b1 的低64位为 vb1
    const __m128i vb1 = _mm_loadl_epi64((const __m128i*)b1);
    // 对 vb1 进行卷积核零点处理，得到 vxb1
    const __m128i vxb1 =
        _mm_sub_epi16(_mm_unpacklo_epi8(vb1, vzero), vb_zero_point_1);
    b1 += b_stride;

    // 加载并处理 b2 的低64位为 vb2
    const __m128i vb2 = _mm_loadl_epi64((const __m128i*)b2);
    // 对 vb2 进行卷积核零点处理，得到 vxb2
    const __m128i vxb2 =
        _mm_sub_epi16(_mm_unpacklo_epi8(vb2, vzero), vb_zero_point_2);
    b2 += b_stride;

    // 加载并处理 b3 的低64位为 vb3
    const __m128i vb3 = _mm_loadl_epi64((const __m128i*)b3);
    // 对 vb3 进行卷积核零点处理，得到 vxb3
    const __m128i vxb3 =
        _mm_sub_epi16(_mm_unpacklo_epi8(vb3, vzero), vb_zero_point_3);
    b3 += b_stride;

    // 累加 va0 和 vb0 的乘积到对应的累加器
    vacc00 = _mm_add_epi32(vacc00, _mm_madd_epi16(vxa0, vxb0));
    vacc01 = _mm_add_epi32(vacc01, _mm_madd_epi16(vxa0, vxb1));
    vacc02 = _mm_add_epi32(vacc02, _mm_madd_epi16(vxa0, vxb2));
    vacc03 = _mm_add_epi32(vacc03, _mm_madd_epi16(vxa0, vxb3));
    vacc10 = _mm_add_epi32(vacc10, _mm_madd_epi16(vxa1, vxb0));
    vacc11 = _mm_add_epi32(vacc11, _mm_madd_epi16(vxa1, vxb1));
    vacc12 = _mm_add_epi32(vacc12, _mm_madd_epi16(vxa1, vxb2));
    vacc13 = _mm_add_epi32(vacc13, _mm_madd_epi16(vxa1, vxb3));
  }

  // 处理剩余不足8个元素的情况
  if (k != 0) {
    // 计算需要预减的数量
    const size_t a_predecrement = 8 - k;
    // 构建位移量 va_shift
    const __m128i va_shift = _mm_cvtsi32_si128(8 * a_predecrement);

    // 部分加载 VA 的零点偏移量，处理零点
    const __m128i va_zero_point_partial = _mm_unpacklo_epi8(
        _mm_srl_epi64(_mm_packus_epi16(va_zero_point, va_zero_point), va_shift),
        vzero);

    // 加载部分 a0 并右移 va_shift 位
    const __m128i va0 = _mm_srl_epi64(
        _mm_loadl_epi64((const __m128i*)(a0 - a_predecrement)), va_shift);


这段代码是一个 SIMD 向量化计算的例子，用于处理卷积操作中的乘积累加，其中包含了对输入和卷积核的零点处理。
    const __m128i vxa0 =
        sub_zero_point(_mm_unpacklo_epi8(va0, vzero), va_zero_point_partial);
    // 计算 va0 的低位 8 个字节的解压缩，并减去零点偏移，结果保存在 vxa0 中

    const __m128i va1 = _mm_srl_epi64(
        _mm_loadl_epi64((const __m128i*)(a1 - a_predecrement)), va_shift);
    // 从地址 a1 - a_predecrement 处加载 8 字节数据到 va1，然后对其进行右移操作，结果保存在 va1 中

    const __m128i vxa1 =
        sub_zero_point(_mm_unpacklo_epi8(va1, vzero), va_zero_point_partial);
    // 计算 va1 的低位 8 个字节的解压缩，并减去零点偏移，结果保存在 vxa1 中

    const __m128i vb0 = _mm_loadl_epi64((const __m128i*)b0);
    // 从地址 b0 处加载 8 字节数据到 vb0

    const __m128i vxb0 =
        _mm_sub_epi16(_mm_unpacklo_epi8(vb0, vzero), vb_zero_point_0);
    // 计算 vb0 的低位 8 个字节的解压缩，并减去零点偏移，结果保存在 vxb0 中

    const __m128i vb1 = _mm_loadl_epi64((const __m128i*)b1);
    // 从地址 b1 处加载 8 字节数据到 vb1

    const __m128i vxb1 =
        _mm_sub_epi16(_mm_unpacklo_epi8(vb1, vzero), vb_zero_point_1);
    // 计算 vb1 的低位 8 个字节的解压缩，并减去零点偏移，结果保存在 vxb1 中

    const __m128i vb2 = _mm_loadl_epi64((const __m128i*)b2);
    // 从地址 b2 处加载 8 字节数据到 vb2

    const __m128i vxb2 =
        _mm_sub_epi16(_mm_unpacklo_epi8(vb2, vzero), vb_zero_point_2);
    // 计算 vb2 的低位 8 个字节的解压缩，并减去零点偏移，结果保存在 vxb2 中

    const __m128i vb3 = _mm_loadl_epi64((const __m128i*)b3);
    // 从地址 b3 处加载 8 字节数据到 vb3

    const __m128i vxb3 =
        _mm_sub_epi16(_mm_unpacklo_epi8(vb3, vzero), vb_zero_point_3);
    // 计算 vb3 的低位 8 个字节的解压缩，并减去零点偏移，结果保存在 vxb3 中

    vacc00 = _mm_add_epi32(vacc00, _mm_madd_epi16(vxa0, vxb0));
    // 计算 vxa0 和 vxb0 的逐元素乘积，然后将结果与 vacc00 中的对应元素相加

    vacc01 = _mm_add_epi32(vacc01, _mm_madd_epi16(vxa0, vxb1));
    // 计算 vxa0 和 vxb1 的逐元素乘积，然后将结果与 vacc01 中的对应元素相加

    vacc02 = _mm_add_epi32(vacc02, _mm_madd_epi16(vxa0, vxb2));
    // 计算 vxa0 和 vxb2 的逐元素乘积，然后将结果与 vacc02 中的对应元素相加

    vacc03 = _mm_add_epi32(vacc03, _mm_madd_epi16(vxa0, vxb3));
    // 计算 vxa0 和 vxb3 的逐元素乘积，然后将结果与 vacc03 中的对应元素相加

    vacc10 = _mm_add_epi32(vacc10, _mm_madd_epi16(vxa1, vxb0));
    // 计算 vxa1 和 vxb0 的逐元素乘积，然后将结果与 vacc10 中的对应元素相加

    vacc11 = _mm_add_epi32(vacc11, _mm_madd_epi16(vxa1, vxb1));
    // 计算 vxa1 和 vxb1 的逐元素乘积，然后将结果与 vacc11 中的对应元素相加

    vacc12 = _mm_add_epi32(vacc12, _mm_madd_epi16(vxa1, vxb2));
    // 计算 vxa1 和 vxb2 的逐元素乘积，然后将结果与 vacc12 中的对应元素相加

    vacc13 = _mm_add_epi32(vacc13, _mm_madd_epi16(vxa1, vxb3));
    // 计算 vxa1 和 vxb3 的逐元素乘积，然后将结果与 vacc13 中的对应元素相加
  }

  __m128i vacc0x0123 = pytorch_sse_reduce4_i32(vacc00, vacc01, vacc02, vacc03);
  // 对 vacc00、vacc01、vacc02、vacc03 进行横向加法归约，结果保存在 vacc0x0123 中

  __m128i vacc1x0123 = pytorch_sse_reduce4_i32(vacc10, vacc11, vacc12, vacc13);
  // 对 vacc10、vacc11、vacc12、vacc13 进行横向加法归约，结果保存在 vacc1x0123 中

  const __m128 vmultiplier =
      _mm_loadu_ps(&quantization_params->sse2.requantization_scales
          [output_channel_index]);
  // 从 quantization_params->sse2.requantization_scales 数组中加载浮点数，存储在 vmultiplier 中

  vacc0x0123 = _mm_cvtps_epi32(
                _mm_mul_ps(
                  _mm_cvtepi32_ps(vacc0x0123),
                  vmultiplier
                  )
                );
  // 将 vacc0x0123 转换为浮点数，乘以 vmultiplier，再转换回整数，并存储回 vacc0x0123 中

  vacc1x0123 = _mm_cvtps_epi32(
                _mm_mul_ps(
                  _mm_cvtepi32_ps(vacc1x0123),
                  vmultiplier
                  )
                );
  // 将 vacc1x0123 转换为浮点数，乘以 vmultiplier，再转换回整数，并存储回 vacc1x0123 中

  const __m128i voutput_zero_point = _mm_load_si128(
      (const __m128i*)quantization_params->sse2.output_zero_point);
  // 从 quantization_params->sse2.output_zero_point 加载 128 位数据到 voutput_zero_point 中

  const __m128i vacc01x0123 = _mm_adds_epi16(
      _mm_packs_epi32(vacc0x0123, vacc1x0123), voutput_zero_point);
  // 将 vacc0x0123 和 vacc1x0123 按 32 位整数打包为 16 位整数，然后加上 voutput_zero_point

  __m128i vout = _mm_packus_epi16(vacc01x0123, vacc01x0123);
  // 将 vacc01x0123 中的 16 位整数打包为无符号 8 位整数，并存储在 vout 中

  vout = _mm_min_epu8(
      vout,
      _mm_load_si128((const __m128i*)quantization_params->sse2.output_max));
  // 将 vout 中的每个元素与 quantization_params->sse2.output_max 中对应元素取最小值

  vout = _mm_max_epu8(
      vout,
      _mm_load_si128((const __m128i*)quantization_params->sse2.output_min));
  // 将 vout 中的每个元素与 quantization_params->sse2.output_min 中对应元素取最大值

  uint8_t* c0 = c;
  // 将指针 c 赋值给 c0

  uint8_t* c1 = (uint8_t*)((uintptr_t)c0 + c_stride);
  // 将指针 c0 偏移 c_stride 字节后的地址赋值给 c1，
    // 如果剩余的元素个数大于等于2，则执行以下操作
    if (nr >= 2) {
      // 将 vout 的低16位提取为 uint16_t 类型，并存储到 c0 指向的地址处，然后 c0 向后移动2个字节
      *((uint16_t*)c0) = (uint16_t)_mm_extract_epi16(vout, 0);
      c0 += 2;
      // 将 vout 的第2和第3个16位提取为 uint16_t 类型，并存储到 c1 指向的地址处，然后 c1 向后移动2个字节
      *((uint16_t*)c1) = (uint16_t)_mm_extract_epi16(vout, 2);
      c1 += 2;
      // 将 vout 向右移动16位，相当于去除高16位的数据
      vout = _mm_srli_epi32(vout, 16);
      // 减少剩余的元素个数计数器
      nr -= 2;
    }
    // 如果剩余的元素个数不为0，则执行以下操作
    if (nr != 0) {
      // 将 vout 转换为 int 类型后，提取其低8位并存储到 c0 指向的地址处
      *((uint8_t*)c0) = (uint8_t)_mm_cvtsi128_si32(vout);
      // 将 vout 的第2个16位提取为 uint8_t 类型，并存储到 c1 指向的地址处
      *((uint8_t*)c1) = (uint8_t)_mm_extract_epi16(vout, 2);
    }
  }
}



# 这是一个单独的右大括号 '}'，用于结束一个代码块或语句的定义。
```