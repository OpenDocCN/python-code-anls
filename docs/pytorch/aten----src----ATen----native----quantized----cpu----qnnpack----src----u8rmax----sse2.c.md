# `.\pytorch\aten\src\ATen\native\quantized\cpu\qnnpack\src\u8rmax\sse2.c`

```py
/*
 * 根据 Facebook, Inc. 及其附属公司的版权声明，
 * 保留所有权利。
 *
 * 此源代码根据根目录中的 LICENSE 文件中的 BSD 风格许可证进行许可。
 */

// 包含必要的头文件，assert.h 用于断言，emmintrin.h 和 qnnpack/u8rmax.h 用于 SIMD 和相关函数
#include <assert.h>
#include <emmintrin.h>
#include <qnnpack/u8rmax.h>

// 定义一个函数 pytorch_u8rmax_ukernel__sse2，接收一个大小和一个指向 uint8_t 数组的指针，并返回一个 uint8_t 值
uint8_t pytorch_u8rmax_ukernel__sse2(size_t n, const uint8_t* x) {
  // 断言 n 不为零，确保输入参数有效
  assert(n != 0);

  // 如果 n 大于等于 16，使用 SSE2 SIMD 指令加速计算
  if (PYTORCH_QNNP_LIKELY(n >= 16)) {
    // 初始化一个 __m128i 类型的变量 vmax，并将其置为零
    __m128i vmax = _mm_setzero_si128();
    // 使用 SIMD 加速的 do-while 循环，每次处理 16 个元素
    do {
      // 从地址 x 处加载 16 个字节到 vx 变量中
      const __m128i vx = _mm_loadu_si128((const __m128i*)x);
      // 指针 x 向后移动 16 个字节
      x += 16;
      // 使用 SIMD 指令求 vmax 和 vx 的每个对应元素的最大值
      vmax = _mm_max_epu8(vmax, vx);
      // 减少剩余元素数量 n
      n -= 16;
    } while (n >= 16);
    // 处理剩余不足 16 个元素的情况
    if (n != 0) {
      // 计算 x 指针需要额外增加的偏移量，以便读取剩余的元素
      const size_t x_increment = n - 16;
      x = (const uint8_t*)((uintptr_t)x + x_increment);
      // 再次加载剩余元素到 vx 变量中，并更新 vmax
      const __m128i vx = _mm_loadu_si128((const __m128i*)x);
      vmax = _mm_max_epu8(vmax, vx);
    }
    // 使用 SIMD 指令对 vmax 中的元素进行归约，求出最大值
    vmax = _mm_max_epu8(vmax, _mm_unpackhi_epi64(vmax, vmax));
    vmax = _mm_max_epu8(vmax, _mm_srli_epi64(vmax, 32));
    vmax = _mm_max_epu8(vmax, _mm_srli_epi32(vmax, 16));
    vmax = _mm_max_epu8(vmax, _mm_srli_epi16(vmax, 8));
    // 将最终结果转换为 uint8_t 类型并返回
    return (uint8_t)_mm_cvtsi128_si32(vmax);
  }
  // 如果 n 小于 16，使用普通的循环计算最大值
  else {
    // 初始化一个变量 vmax 作为最大值的初始值
    uint8_t vmax = 0;
    // 普通的 do-while 循环，逐个比较每个元素，找到最大值
    do {
      const uint8_t vx = *x++;
      vmax = vx > vmax ? vx : vmax;
    } while (--n != 0);
    // 返回普通计算得到的最大值
    return vmax;
  }
}
```