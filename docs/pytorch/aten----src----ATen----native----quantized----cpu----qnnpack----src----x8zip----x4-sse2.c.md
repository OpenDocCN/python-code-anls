# `.\pytorch\aten\src\ATen\native\quantized\cpu\qnnpack\src\x8zip\x4-sse2.c`

```py
/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <emmintrin.h>          // 引入 SSE2 指令集支持的头文件

#include <qnnpack/x8zip.h>      // 引入 x8zip 头文件

void pytorch_qnnp_x8zip_x4__sse2(size_t n, const void* input, void* output) {
  const uint8_t* x = input;     // 定义指向输入数据的指针 x
  const uint8_t* y = x + n;     // 定义指向输入数据的指针 y，偏移量为 n
  const uint8_t* z = y + n;     // 定义指向输入数据的指针 z，偏移量为 2 * n
  const uint8_t* w = z + n;     // 定义指向输入数据的指针 w，偏移量为 3 * n
  uint8_t* o = output;          // 定义指向输出数据的指针 o

  if (n >= 16) {                // 如果 n 大于等于 16，执行以下循环
    do {
      const __m128i vx = _mm_loadu_si128((const __m128i*)x);   // 从 x 加载 128 位无对齐整数数据到 vx
      x += 16;                    // x 偏移 16 字节
      const __m128i vy = _mm_loadu_si128((const __m128i*)y);   // 从 y 加载 128 位无对齐整数数据到 vy
      y += 16;                    // y 偏移 16 字节
      const __m128i vz = _mm_loadu_si128((const __m128i*)z);   // 从 z 加载 128 位无对齐整数数据到 vz
      z += 16;                    // z 偏移 16 字节
      const __m128i vw = _mm_loadu_si128((const __m128i*)w);   // 从 w 加载 128 位无对齐整数数据到 vw
      w += 16;                    // w 偏移 16 字节
      const __m128i vxy_lo = _mm_unpacklo_epi8(vx, vy);        // 拆分 vx 和 vy 的低位字节并组合成 16 位整数数据到 vxy_lo
      const __m128i vxy_hi = _mm_unpackhi_epi8(vx, vy);        // 拆分 vx 和 vy 的高位字节并组合成 16 位整数数据到 vxy_hi
      const __m128i vzw_lo = _mm_unpacklo_epi8(vz, vw);        // 拆分 vz 和 vw 的低位字节并组合成 16 位整数数据到 vzw_lo
      const __m128i vzw_hi = _mm_unpackhi_epi8(vz, vw);        // 拆分 vz 和 vw 的高位字节并组合成 16 位整数数据到 vzw_hi
      const __m128i vxyzw0 = _mm_unpacklo_epi16(vxy_lo, vzw_lo);// 拆分 vxy_lo 和 vzw_lo 的低位字节并组合成 32 位整数数据到 vxyzw0
      const __m128i vxyzw1 = _mm_unpackhi_epi16(vxy_lo, vzw_lo);// 拆分 vxy_lo 和 vzw_lo 的高位字节并组合成 32 位整数数据到 vxyzw1
      const __m128i vxyzw2 = _mm_unpacklo_epi16(vxy_hi, vzw_hi);// 拆分 vxy_hi 和 vzw_hi 的低位字节并组合成 32 位整数数据到 vxyzw2
      const __m128i vxyzw3 = _mm_unpackhi_epi16(vxy_hi, vzw_hi);// 拆分 vxy_hi 和 vzw_hi 的高位字节并组合成 32 位整数数据到 vxyzw3
      _mm_storeu_si128((__m128i*)o, vxyzw0);   // 将 vxyzw0 存储到 o
      _mm_storeu_si128((__m128i*)o + 1, vxyzw1); // 将 vxyzw1 存储到 o 的下一个 128 位位置
      _mm_storeu_si128((__m128i*)o + 2, vxyzw2); // 将 vxyzw2 存储到 o 的下下个 128 位位置
      _mm_storeu_si128((__m128i*)o + 3, vxyzw3); // 将 vxyzw3 存储到 o 的下下下个 128 位位置
      o = (void*)((uintptr_t)o + 64);   // o 偏移 64 字节
      n -= 16;    // 减少 n 的值 16
    } while (n >= 16);  // 如果 n 大于等于 16，继续循环
    if (n != 0) {   // 如果 n 不等于 0，执行以下操作
      const size_t address_increment = n - 16;  // 计算地址增量为 n - 16
      const __m128i vx =
          _mm_loadu_si128((const __m128i*)((uintptr_t)x + address_increment)); // 从 x 加载 128 位无对齐整数数据到 vx
      const __m128i vy =
          _mm_loadu_si128((const __m128i*)((uintptr_t)y + address_increment)); // 从 y 加载 128 位无对齐整数数据到 vy
      const __m128i vz =
          _mm_loadu_si128((const __m128i*)((uintptr_t)z + address_increment)); // 从 z 加载 128 位无对齐整数数据到 vz
      const __m128i vw =
          _mm_loadu_si128((const __m128i*)((uintptr_t)w + address_increment)); // 从 w 加载 128 位无对齐整数数据到 vw
      const __m128i vxy_lo = _mm_unpacklo_epi8(vx, vy);    // 拆分 vx 和 vy 的低位字节并组合成 16 位整数数据到 vxy_lo
      const __m128i vxy_hi = _mm_unpackhi_epi8(vx, vy);    // 拆分 vx 和 vy 的高位字节并组合成 16 位整数数据到 vxy_hi
      const __m128i vzw_lo = _mm_unpacklo_epi8(vz, vw);    // 拆分 vz 和 vw 的低位字节并组合成 16 位整数数据到 vzw_lo
      const __m128i vzw_hi = _mm_unpackhi_epi8(vz, vw);    // 拆分 vz 和 vw 的高位字节并组合成 16 位整数数据到 vzw_hi
      const __m128i vxyzw0 = _mm_unpacklo_epi16(vxy_lo, vzw_lo);  // 拆分 vxy_lo 和 vzw_lo 的低位字节并组合成 32 位整数数据到 vxyzw0
      const __m128i vxyzw1 = _mm_unpackhi_epi16(vxy_lo, vzw_lo);  // 拆分 vxy_lo 和 vzw_lo 的高位字节并组合成 32 位整数数据到 vxyzw1
      const __m128i vxyzw2 = _mm_unpacklo_epi16(vxy_hi, vzw_hi);  // 拆分 vxy_hi 和 vzw_hi 的低位字节并组合成 32 位整数数据到 vxyzw2
      const __m128i vxyzw3 = _mm_unpackhi_epi16(vxy_hi, vzw_hi);  // 拆分 vxy_hi 和 vzw_hi 的高位字节并组合成 32 位整数数据到 vxyzw3
      o = (void*)((uintptr_t)o + address_increment * 4);   // o 偏移 address_increment * 4 字节
      _mm_storeu_si128((__m128i*)o, vxyzw0);   // 将 vxyzw0 存储到 o
      _mm_storeu_si128((__m128i*)o + 1, vxyzw1); // 将 vxyzw1 存储到 o 的
    // 执行一个 do-while 循环，用于将四个输入数组 x, y, z, w 中的每个元素依次取出，
    // 存入输出数组 o 中，并每次将 o 的位置向后移动四个字节。
    do {
      // 从数组 x 中取出一个字节，存入变量 vx，并将 x 的位置向后移动一个字节
      const uint8_t vx = *x++;
      // 从数组 y 中取出一个字节，存入变量 vy，并将 y 的位置向后移动一个字节
      const uint8_t vy = *y++;
      // 从数组 z 中取出一个字节，存入变量 vz，并将 z 的位置向后移动一个字节
      const uint8_t vz = *z++;
      // 从数组 w 中取出一个字节，存入变量 vw，并将 w 的位置向后移动一个字节
      const uint8_t vw = *w++;
      // 将变量 vx 的值存入输出数组 o 的当前位置
      o[0] = vx;
      // 将变量 vy 的值存入输出数组 o 的下一个位置
      o[1] = vy;
      // 将变量 vz 的值存入输出数组 o 的下下一个位置
      o[2] = vz;
      // 将变量 vw 的值存入输出数组 o 的下下下一个位置
      o[3] = vw;
      // 将输出数组 o 的位置向后移动四个字节，以便存放下一组四个字节的数据
      o += 4;
    } while (--n != 0);
  }
}



# 这行代码是一个单独的右花括号 '}'，通常用于结束一个代码块，如函数定义、条件语句、循环等。
```