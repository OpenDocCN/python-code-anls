# `.\pytorch\aten\src\ATen\native\quantized\cpu\qnnpack\src\u8maxpool\sub16-sse2.c`

```
/*
 *`
/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <assert.h>  // 引入断言头文件，用于程序调试

#include <emmintrin.h>  // 引入 SSE2 指令集的头文件

#include <qnnpack/u8maxpool.h>  // 引入 QNNPACK 的 u8maxpool 头文件

// 定义函数 pytorch_u8maxpool_ukernel_sub16__sse2，执行 8 位最大池化运算，使用 SSE2 指令集
void pytorch_u8maxpool_ukernel_sub16__sse2(
    size_t n,  // 输入数据的数量
    size_t ks,  // 卷积核大小
    size_t kc,  // 输入通道数，必须小于 16
    const uint8_t** input,  // 输入数据的指针数组
    uint8_t* output,  // 输出数据的指针
    size_t input_increment,  // 输入数据指针的增量
    size_t output_increment,  // 输出数据指针的增量
    const union pytorch_qnnp_u8_clamping_params params[RESTRICT_STATIC 1]) {  // 输入的裁剪参数

  assert(n != 0);  // 断言输入数据数量不为零
  assert(ks != 0);  // 断言卷积核大小不为零
  assert(kc != 0);  // 断言输入通道数不为零
  assert(kc < 16);  // 断言输入通道数小于 16

  // 加载输出最大值参数
  const __m128i voutput_max =
      _mm_load_si128((const __m128i*)params->sse2.output_max);
  // 加载输出最小值参数
  const __m128i voutput_min =
      _mm_load_si128((const __m128i*)params->sse2.output_min);

  do {
    __m128i vmax = _mm_setzero_si128();  // 初始化 vmax 为零向量

    size_t m = ks;  // 初始化内层循环次数为卷积核大小
    do {
      const uint8_t* i = *input++;  // 获取当前输入数据的指针，并递增输入指针数组
      i += kc;  // 移动输入指针到通道的起始位置
      __m128i vi = vmax;  // 初始化 vi 为 vmax

      // 处理通道数的每个位
      if (kc & 1) {
        i -= 1;  // 处理通道数的最低位
        vi = _mm_cvtsi32_si128(*i);  // 将一个字节的数据加载到 SSE2 向量中
      }
      if (kc & 2) {
        vi = _mm_slli_epi32(vi, 16);  // 将 vi 向量左移 16 位
        i -= 2;  // 移动指针到下一个通道
        vi = _mm_insert_epi16(vi, *((const uint16_t*)i), 0);  // 插入 16 位数据到 vi 向量中
      }
      if (kc & 4) {
        i -= 4;  // 移动指针到下一个通道
        vi = _mm_unpacklo_epi32(
            _mm_cvtsi32_si128((int)*((const uint32_t*)i)), vi);  // 处理 32 位数据
      }
      if (kc & 8) {
        i -= 8;  // 移动指针到下一个通道
        vi = _mm_unpacklo_epi64(_mm_loadl_epi64((const __m128i*)i), vi);  // 处理 64 位数据
      }
      vmax = _mm_max_epu8(vmax, vi);  // 更新 vmax 为 vi 和 vmax 的最大值
    } while (--m != 0);  // 内层循环，处理卷积核的所有数据

    input = (const uint8_t**)((uintptr_t)input + input_increment);  // 更新输入指针

    __m128i vout = _mm_max_epu8(_mm_min_epu8(vmax, voutput_max), voutput_min);  // 应用最大和最小值限制

    // 根据通道数的位，保存结果到输出数据中
    if (kc & 8) {
      _mm_storel_epi64((__m128i*)output, vout);  // 存储低 64 位到输出
      output += 8;  // 移动输出指针 8 个字节
      vout = _mm_unpackhi_epi64(vout, vout);  // 拆分高低 64 位数据
    }
    if (kc & 4) {
      *((uint32_t*)output) = (uint32_t)_mm_cvtsi128_si32(vout);  // 存储低 32 位到输出
      output += 4;  // 移动输出指针 4 个字节
      vout = _mm_srli_epi64(vout, 32);  // 右移 32 位，准备处理低 32 位
    }
    if (kc & 2) {
      *((uint16_t*)output) = (uint16_t)_mm_extract_epi16(vout, 0);  // 存储低 16 位到输出
      output += 2;  // 移动输出指针 2 (uint8_t*)((uintptr_t)output + output_increment);  // 更新输出指针
  } while (--n != 0);  // 循环 n 次
}
```