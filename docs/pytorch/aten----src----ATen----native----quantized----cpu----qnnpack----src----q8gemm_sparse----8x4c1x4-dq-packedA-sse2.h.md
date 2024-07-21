# `.\pytorch\aten\src\ATen\native\quantized\cpu\qnnpack\src\q8gemm_sparse\8x4c1x4-dq-packedA-sse2.h`

```py
/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <immintrin.h>

#include <qnnpack/q8gemm_sparse.h>
#include <requantization/runtime-sse2.h>

#include "8x4c1x4-packed-sse2.h"

// 定义宏 CONVERT_TO_FP_AND_TRANSPOSE，用于将整数转换为单精度浮点数并转置
#define CONVERT_TO_FP_AND_TRANSPOSE(a, b, c, d, t_a, t_b, t_c, t_d)  \
  // 将整数向量 a, b, c, d 转换为单精度浮点数向量
  a_ps = _mm_cvtepi32_ps(a);                                         \
  b_ps = _mm_cvtepi32_ps(b);                                         \
  c_ps = _mm_cvtepi32_ps(c);                                         \
  d_ps = _mm_cvtepi32_ps(d);                                         \
  // 执行单精度浮点数向量的轮换操作，生成临时变量 tmp0, tmp1, tmp2, tmp3
  tmp0 = _mm_shuffle_ps(a_ps, b_ps, _MM_SHUFFLE(1, 0, 1, 0));        \
  tmp1 = _mm_shuffle_ps(a_ps, b_ps, _MM_SHUFFLE(3, 2, 3, 2));        \
  tmp2 = _mm_shuffle_ps(c_ps, d_ps, _MM_SHUFFLE(1, 0, 1, 0));        \
  tmp3 = _mm_shuffle_ps(c_ps, d_ps, _MM_SHUFFLE(3, 2, 3, 2));        \
  // 将轮换后的结果再次进行轮换，生成 t_a, t_b, t_c, t_d 向量
  t_a = _mm_shuffle_ps(tmp0, tmp2, _MM_SHUFFLE(2, 0, 2, 0));         \
  t_b = _mm_shuffle_ps(tmp0, tmp2, _MM_SHUFFLE(3, 1, 3, 1));         \
  t_c = _mm_shuffle_ps(tmp1, tmp3, _MM_SHUFFLE(2, 0, 2, 0));         \
  t_d = _mm_shuffle_ps(tmp1, tmp3, _MM_SHUFFLE(3, 1, 3, 1));

// 宏 KERNEL_NAME 和 W_INDEX_DTYPE 在 https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/quantized/cpu/qnnpack/src/q8gemm_sparse/8x4c1x4-dq-packedA-sse2.c 中定义
// KERNEL_NAME 函数用于执行稀疏量化矩阵乘法的 SSE2 实现
void KERNEL_NAME(
    size_t mr,                            // 行尺寸
    size_t nr,                            // 列尺寸
    const uint8_t* a_packed,              // 稀疏矩阵 A 的紧凑表示
    const uint8_t* packed_w,              // 稀疏矩阵 W 的紧凑表示
    const W_INDEX_DTYPE* w_row_ptr,       // 稀疏矩阵 W 的行指针数组
    const W_INDEX_DTYPE* w_block_ids_ptr, // 稀疏矩阵 W 的块 ID 指针数组
    const float* b,                       // 矩阵 B 的数据
    float* c,                             // 矩阵 C 的数据
    size_t c_stride,                      // 矩阵 C 的跨度
    size_t output_channel_index,
    // KERNEL_NAME 函数实现了 QNNPACK 库中的稀疏量化矩阵乘法的 SSE2 版本，具体实现可以查看对应的源文件
  const struct pytorch_qnnp_conv_dynamic_quantization_params
      quantization_params[RESTRICT_STATIC 1]) {
  // 从输入量化参数中获取输入零点，生成128位整数向量
  const __m128i va_zero_point = _mm_set1_epi16(quantization_params->input_zero_point);
  // 载入偏置向量b作为128位单精度浮点数向量vbias
  const __m128 vbias = _mm_load_ps(b);
  // 设置128位整数零向量vzero
  const __m128i vzero = _mm_setzero_si128();

  // Packed A格式说明
  // 对于8kx4m块，所有块给定4行(4m)，这些块被连续存储在内存中。
  // 原始的A
  // --------- K -----------          -- (K + 4 - 1) / 4 --
  // |                     |          |                   |
  // |                     |        (M + 8 - 1)/8         |
  // |                     | Packed   |                   |
  // M                     |  =>      |-------------------|
  // |                     |        因此Packed A有(K + 4 - 1)/4 * (M + 8 -1)/8块
  // |                     |
  // |---------------------|
  //
  // 每个8 x 4块被转置并存储。
  // 对于每个给定的8m块组的(K + 4 - 1)/4块
  // 在内存中是相邻存储的。
  // 因此，每个块:
  // |----8m-----|----8m-----|
  // 4k          |           | .....
  // |-----------|-----------|
  // 这种局部性有助于加载8kx8m激活块
  // 注意当M不是8的倍数时，剩余部分可以包含任意数据在Packed A中，因为我们不会写入这些数据。
  // 这将通过仅复制相应有效数据来处理

  __m128i vacc_low[4];  // 低位累加寄存器数组
  __m128i vacc_high[4];  // 高位累加寄存器数组
  // 载入输出通道索引处的量化参数乘数，生成128位单精度浮点数向量vmultiplier
  const __m128 vmultiplier =
      _mm_loadu_ps(&quantization_params->multipliers[output_channel_index]);
  for (int32_t n = 0; n < nr; n++) {
    // 初始化低位累加寄存器为零
    vacc_low[n] = _mm_setzero_si128();
    // 初始化高位累加寄存器为零
    vacc_high[n] = _mm_setzero_si128();
    // 获取当前输出通道索引和n的内核零点
    const int16_t b_zero_point =
      (int16_t)(uint16_t)quantization_params->kernel_zero_points[
      output_channel_index + n];

    // 计算块数，表示为w_row_ptr中下一行指针与当前行指针之差
    int32_t num_blocks = w_row_ptr[n+1] - w_row_ptr[n];
    // 压缩值的偏移量。
    // w_row_ptr[0]是压缩值中的块偏移，对应于权重矩阵某行的起始位置。
    const uint8_t* temp_packed_w = packed_w + w_row_ptr[n] * COL_BLOCK_SIZE;
    // 同样，w_row_ptr[0]也是块偏移，对应于该行的块列id的起始位置。
    // 每行的块列id数等于块值数
    const W_INDEX_DTYPE* temp_w_block_ids_ptr = w_block_ids_ptr + w_row_ptr[n];
  }
  }

  __m128 vout[8];  // 声明一个包含8个__m128类型元素的数组vout，用于存储128位数据

  __m128 a_ps, b_ps, c_ps, d_ps, tmp0, tmp1, tmp2, tmp3;  // 声明多个__m128类型的变量，用于临时存储数据

  // Transform low half of 4x8 result
  // That is 4x4 block (4n x 4m)
  // Convert to FP and transpose: 4m x 4n
  // 将4x8结果的低半部分转换为浮点数并转置，形成4m x 4n的矩阵
  CONVERT_TO_FP_AND_TRANSPOSE(vacc_low[0],
                              vacc_low[1],
                              vacc_low[2],
                              vacc_low[3],
                              vout[0],
                              vout[1],
                              vout[2],
                              vout[3])

  CONVERT_TO_FP_AND_TRANSPOSE(vacc_high[0],
                              vacc_high[1],
                              vacc_high[2],
                              vacc_high[3],
                              vout[4],
                              vout[5],
                              vout[6],
                              vout[7])

  vout[0] = _mm_mul_ps(vmultiplier, vout[0]);  // 使用vmultiplier对vout数组的元素进行乘法运算
  vout[1] = _mm_mul_ps(vmultiplier, vout[1]);
  vout[2] = _mm_mul_ps(vmultiplier, vout[2]);
  vout[3] = _mm_mul_ps(vmultiplier, vout[3]);
  vout[4] = _mm_mul_ps(vmultiplier, vout[4]);
  vout[5] = _mm_mul_ps(vmultiplier, vout[5]);
  vout[6] = _mm_mul_ps(vmultiplier, vout[6]);
  vout[7] = _mm_mul_ps(vmultiplier, vout[7]);

  vout[0] = _mm_add_ps(vout[0], vbias);  // 将vbias加到vout数组的元素上
  vout[1] = _mm_add_ps(vout[1], vbias);
  vout[2] = _mm_add_ps(vout[2], vbias);
  vout[3] = _mm_add_ps(vout[3], vbias);
  vout[4] = _mm_add_ps(vout[4], vbias);
  vout[5] = _mm_add_ps(vout[5], vbias);
  vout[6] = _mm_add_ps(vout[6], vbias);
  vout[7] = _mm_add_ps(vout[7], vbias);

  float* c0 = c;  // 声明指向浮点数数组c的指针c0，指向第一个元素
  float* c1 = c0 + c_stride;  // 声明指向浮点数数组c的指针c1，指向第二个元素
  if (mr < 2) {
    c1 = c0;  // 如果mr小于2，则将c1指向c0，且将vout[1]设置为vout[0]
    vout[1] = vout[0];
  }
  float* c2 = c1 + c_stride;  // 声明指向浮点数数组c的指针c2，指向第三个元素
  if (mr < 3) {
    c2 = c0;  // 如果mr小于3，则将c2指向c0，且将vout[2]设置为vout[0]
    vout[2] = vout[0];
  }
  float* c3 = c2 + c_stride;  // 声明指向浮点数数组c的指针c3，指向第四个元素
  if (mr < 4) {
    c3 = c0;  // 如果mr小于4，则将c3指向c0，且将vout[3]设置为vout[0]
    vout[3] = vout[0];
  }
  float* c4 = c3 + c_stride;  // 声明指向浮点数数组c的指针c4，指向第五个元素
  if (mr < 5) {
    c4 = c0;  // 如果mr小于5，则将c4指向c0，且将vout[4]设置为vout[0]
    vout[4] = vout[0];
  }
  float* c5 = c4 + c_stride;  // 声明指向浮点数数组c的指针c5，指向第六个元素
  if (mr < 6) {
    c5 = c0;  // 如果mr小于6，则将c5指向c0，且将vout[5]设置为vout[0]
    vout[5] = vout[0];
  }
  float* c6 = c5 + c_stride;  // 声明指向浮点数数组c的指针c6，指向第七个元素
  if (mr < 7) {
    c6 = c0;  // 如果mr小于7，则将c6指向c0，且将vout[6]设置为vout[0]
    vout[6] = vout[0];
  }
  float* c7 = c6 + c_stride;  // 声明指向浮点数数组c的指针c7，指向第八个元素
  if (mr < 8) {
    c7 = c0;  // 如果mr小于8，则将c7指向c0，且将vout[7]设置为vout[0]
    vout[7] = vout[0];
  }

  if (nr == 4) {
    _mm_storeu_ps(c0, vout[0]);  // 将vout[0]的值存储到c0所指向的地址
    _mm_storeu_ps(c1, vout[1]);  // 将vout[1]的值存储到c1所指向的地址
    _mm_storeu_ps(c2, vout[2]);  // 将vout[2]的值存储到c2所指向的地址
    _mm_storeu_ps(c3, vout[3]);  // 将vout[3]的值存储到c3所指向的地址
    _mm_storeu_ps(c4, vout[4]);  // 将vout[4]的值存储到c4所指向的地址
    _mm_storeu_ps(c5, vout[5]);  // 将vout[5]的值存储到c5所指向的地址
    _mm_storeu_ps(c6, vout[6]);  // 将vout[6]的值存储到c6所指向的地址
    _mm_storeu_ps(c7, vout[7]);  // 将vout[7]的值存储到c7所指向的地址
  } else {
    # 如果剩余处理的数量大于等于2，则执行以下操作
    if (nr >= 2) {
      # 将 vout 中的向量数据存储到 c0 到 c7 指向的内存中
      _mm_storel_pi((__m64*)c0, vout[0]);
      _mm_storel_pi((__m64*)c1, vout[1]);
      _mm_storel_pi((__m64*)c2, vout[2]);
      _mm_storel_pi((__m64*)c3, vout[3]);
      _mm_storel_pi((__m64*)c4, vout[4]);
      _mm_storel_pi((__m64*)c5, vout[5]);
      _mm_storel_pi((__m64*)c6, vout[6]);
      _mm_storel_pi((__m64*)c7, vout[7]);

      # 减少剩余处理数量 nr
      nr -= 2;

      # 指向下一个位置
      c0 += 2;
      c1 += 2;
      c2 += 2;
      c3 += 2;
      c4 += 2;
      c5 += 2;
      c6 += 2;
      c7 += 2;

      # 对 vout 中的向量进行数据重组，将第二个单精度浮点数复制到所有位置
      vout[0] = _mm_shuffle_ps(vout[0], vout[0], _MM_SHUFFLE(2, 2, 2, 2));
      vout[1] = _mm_shuffle_ps(vout[1], vout[1], _MM_SHUFFLE(2, 2, 2, 2));
      vout[2] = _mm_shuffle_ps(vout[2], vout[2], _MM_SHUFFLE(2, 2, 2, 2));
      vout[3] = _mm_shuffle_ps(vout[3], vout[3], _MM_SHUFFLE(2, 2, 2, 2));
      vout[4] = _mm_shuffle_ps(vout[4], vout[4], _MM_SHUFFLE(2, 2, 2, 2));
      vout[5] = _mm_shuffle_ps(vout[5], vout[5], _MM_SHUFFLE(2, 2, 2, 2));
      vout[6] = _mm_shuffle_ps(vout[6], vout[6], _MM_SHUFFLE(2, 2, 2, 2));
      vout[7] = _mm_shuffle_ps(vout[7], vout[7], _MM_SHUFFLE(2, 2, 2, 2));
    }
    # 如果剩余处理的数量不为0，则执行以下操作
    if (nr != 0) {
      # 将 vout 中的向量的第一个单精度浮点数存储到 c0 到 c7 指向的内存中
      *c0 = _mm_cvtss_f32(vout[0]);
      *c1 = _mm_cvtss_f32(vout[1]);
      *c2 = _mm_cvtss_f32(vout[2]);
      *c3 = _mm_cvtss_f32(vout[3]);
      *c4 = _mm_cvtss_f32(vout[4]);
      *c5 = _mm_cvtss_f32(vout[5]);
      *c6 = _mm_cvtss_f32(vout[6]);
      *c7 = _mm_cvtss_f32(vout[7]);
    }
}



# 这行代码表示一个单独的右花括号 '}'，通常用于闭合代码块，例如函数、循环或条件语句的结束。
```