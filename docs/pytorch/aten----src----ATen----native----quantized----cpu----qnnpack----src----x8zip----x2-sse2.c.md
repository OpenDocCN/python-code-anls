# `.\pytorch\aten\src\ATen\native\quantized\cpu\qnnpack\src\x8zip\x2-sse2.c`

```py
/*
 * 包含必要的头文件和定义，用于支持 SIMD 加速指令集 SSE2
 */
#include <emmintrin.h>

/*
 * 包含自定义的头文件，声明了本文件中实现的函数
 */
#include <qnnpack/x8zip.h>

/*
 * 实现一个函数，用于将两个输入数组交叉合并为一个输出数组，利用 SSE2 指令集进行优化
 */
void pytorch_qnnp_x8zip_x2__sse2(size_t n, const void* input, void* output) {
  /*
   * 将输入指针转换为 uint8_t 类型的指针，分别指向两个输入数组和一个输出数组
   */
  const uint8_t* x = input;
  const uint8_t* y = x + n;
  uint8_t* o = output;

  /*
   * 如果剩余的数据长度大于等于 16 字节，则使用 SSE2 指令集进行并行处理
   */
  if (n >= 16) {
    do {
      /*
       * 加载两个 128 位数据块到 xmm 寄存器中
       */
      const __m128i vx = _mm_loadu_si128((const __m128i*)x);
      x += 16;
      const __m128i vy = _mm_loadu_si128((const __m128i*)y);
      y += 16;
      
      /*
       * 拆分和交叉组合两个 128 位数据块的低位和高位部分
       */
      const __m128i vxy_lo = _mm_unpacklo_epi8(vx, vy);
      const __m128i vxy_hi = _mm_unpackhi_epi8(vx, vy);
      
      /*
       * 将结果存储到输出数组中
       */
      _mm_storeu_si128((__m128i*)o, vxy_lo);
      _mm_storeu_si128((__m128i*)(o + 16), vxy_hi);
      
      /*
       * 更新输出数组的指针位置，使其指向下一个存储位置
       */
      o = (void*)((uintptr_t)o + 32);
      n -= 16;
    } while (n >= 16);

    /*
     * 处理剩余的不足 16 字节的数据
     */
    if (n != 0) {
      /*
       * 计算地址增量，加载剩余数据块并进行拆分和交叉组合
       */
      const size_t address_increment = n - 16;
      const __m128i vx =
          _mm_loadu_si128((const __m128i*)((uintptr_t)x + address_increment));
      const __m128i vy =
          _mm_loadu_si128((const __m128i*)((uintptr_t)y + address_increment));
      const __m128i vxy_lo = _mm_unpacklo_epi8(vx, vy);
      const __m128i vxy_hi = _mm_unpackhi_epi8(vx, vy);
      
      /*
       * 更新输出数组的指针位置，存储剩余数据的结果
       */
      o = (void*)((uintptr_t)o + address_increment * 2);
      _mm_storeu_si128((__m128i*)o, vxy_lo);
      _mm_storeu_si128((__m128i*)o + 1, vxy_hi);
    }
  } else {
    /*
     * 处理长度小于 16 字节的情况，直接逐个元素进行交叉合并
     */
    do {
      const uint8_t vx = *x++;
      const uint8_t vy = *y++;
      o[0] = vx;
      o[1] = vy;
      o += 2;
    } while (--n != 0);
  }
}
```