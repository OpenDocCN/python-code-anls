# `.\pytorch\aten\src\ATen\native\quantized\cpu\qnnpack\src\sgemm\6x8-neon.c`

```
/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <arm_neon.h>  // 包含 ARM NEON 指令集的头文件

#include <qnnpack/sgemm.h>  // 包含 QNNPACK 中的矩阵乘法函数声明

void pytorch_sgemm_ukernel_6x8__neon(
    size_t mr,
    size_t nr,
    size_t k,
    const float* restrict a,
    size_t a_stride,
    const float* restrict w,
    float* restrict c,
    size_t c_stride,
    const struct pytorch_qnnp_fp32_clamping_params
        clamping_params[restrict static 1]) {
  float32x4_t vacc0x0123 = vld1q_f32(w);  // 加载 w 到 NEON 寄存器 vacc0x0123
  w += 4;  // 更新 w 指针到下一个寄存器位置
  float32x4_t vacc0x4567 = vld1q_f32(w);  // 加载 w 到 NEON 寄存器 vacc0x4567
  w += 4;  // 更新 w 指针到下一个寄存器位置
  float32x4_t vacc1x0123 = vacc0x0123;  // 复制 vacc0x0123 到 vacc1x0123
  float32x4_t vacc1x4567 = vacc0x4567;  // 复制 vacc0x4567 到 vacc1x4567
  float32x4_t vacc2x0123 = vacc0x0123;  // 复制 vacc0x0123 到 vacc2x0123
  float32x4_t vacc2x4567 = vacc0x4567;  // 复制 vacc0x4567 到 vacc2x4567
  float32x4_t vacc3x0123 = vacc0x0123;  // 复制 vacc0x0123 到 vacc3x0123
  float32x4_t vacc3x4567 = vacc0x4567;  // 复制 vacc0x4567 到 vacc3x4567
  float32x4_t vacc4x0123 = vacc0x0123;  // 复制 vacc0x0123 到 vacc4x0123
  float32x4_t vacc4x4567 = vacc0x4567;  // 复制 vacc0x4567 到 vacc4x4567
  float32x4_t vacc5x0123 = vacc0x0123;  // 复制 vacc0x0123 到 vacc5x0123
  float32x4_t vacc5x4567 = vacc0x4567;  // 复制 vacc0x4567 到 vacc5x4567

  const float* a0 = a;  // 设置指针 a0 指向输入矩阵 a 的首行
  const float* a1 = (const float*)((uintptr_t)a0 + a_stride);  // 设置指针 a1 指向下一行
  if (mr < 2) {
    a1 = a0;  // 如果行数 mr 小于 2，a1 指向 a0，即重复使用第一行数据
  }
  const float* a2 = (const float*)((uintptr_t)a1 + a_stride);  // 设置指针 a2 指向下一行
  if (mr <= 2) {
    a2 = a1;  // 如果行数 mr 小于等于 2，a2 指向 a1，即重复使用第二行数据
  }
  const float* a3 = (const float*)((uintptr_t)a2 + a_stride);  // 设置指针 a3 指向下一行
  if (mr < 4) {
    a3 = a2;  // 如果行数 mr 小于 4，a3 指向 a2，即重复使用第三行数据
  }
  const float* a4 = (const float*)((uintptr_t)a3 + a_stride);  // 设置指针 a4 指向下一行
  if (mr <= 4) {
    a4 = a3;  // 如果行数 mr 小于等于 4，a4 指向 a3，即重复使用第四行数据
  }
  const float* a5 = (const float*)((uintptr_t)a4 + a_stride);  // 设置指针 a5 指向下一行
  if (mr != 6) {
    a5 = a4;  // 如果行数 mr 不等于 6，a5 指向 a4，即重复使用第五行数据
  }

  for (; k >= 2; k -= 2) {
    const float32x2_t va0 = vld1_f32(a0);  // 加载输入矩阵 a 的元素到 NEON 寄存器 va0
    a0 += 2;  // 更新 a0 指针到下两个元素位置
    const float32x2_t va1 = vld1_f32(a1);  // 加载输入矩阵 a 的元素到 NEON 寄存器 va1
    a1 += 2;  // 更新 a1 指针到下两个元素位置
    const float32x2_t va2 = vld1_f32(a2);  // 加载输入矩阵 a 的元素到 NEON 寄存器 va2
    a2 += 2;  // 更新 a2 指针到下两个元素位置
    const float32x2_t va3 = vld1_f32(a3);  // 加载输入矩阵 a 的元素到 NEON 寄存器 va3
    a3 += 2;  // 更新 a3 指针到下两个元素位置
    const float32x2_t va4 = vld1_f32(a4);  // 加载输入矩阵 a 的元素到 NEON 寄存器 va4
    a4 += 2;  // 更新 a4 指针到下两个元素位置
    const float32x2_t va5 = vld1_f32(a5);  // 加载输入矩阵 a 的元素到 NEON 寄存器 va5
    a5 += 2;  // 更新 a5 指针到下两个元素位置

    {
      const float32x4_t vb0123 = vld1q_f32(w);  // 加载权重矩阵 w 的元素到 NEON 寄存器 vb0123
      w += 4;  // 更新 w 指针到下四个元素位置
      const float32x4_t vb4567 = vld1q_f32(w);  // 加载权重矩阵 w 的元素到 NEON 寄存器 vb4567
      w += 4;  // 更新 w 指针到下四个元素位置

#if defined(__aarch64__)
      vacc0x0123 = vfmaq_lane_f32(vacc0x0123, vb0123, va0, 0);  // NEON 寄存器加法和乘法操作
      vacc0x4567 = vfmaq_lane_f32(vacc0x4567, vb4567, va0, 0);  // NEON 寄存器加法和乘法操作
      vacc1x0123 = vfmaq_lane_f32(vacc1x0123, vb0123, va1, 0);  // NEON 寄存器加法和乘法操作
      vacc1x4567 = vfmaq_lane_f32(vacc1x4567, vb4567, va1, 0);  // NEON 寄存器加法和乘法操作
      vacc2x0123 = vfmaq_lane_f32(vacc2x0123, vb0123, va2, 0);  // NEON 寄存器加法和乘法操作
      vacc2x4567 = vfmaq_lane_f32(vacc2x4567, vb4567, va2, 0);  // NEON 寄存器加法和乘法操作
      vacc3x0123 = vfmaq_lane_f32(vacc3x0123, vb0123, va3, 0);  // NEON 寄存器加法和乘法操作
      vacc3x4567 = vfmaq_lane_f32(vacc3x4567, vb4567, va3, 0);
#else
      // 如果不是 ARM 64 位架构，使用 vmlaq_lane_f32 函数进行向量加乘操作
      vacc0x0123 = vmlaq_lane_f32(vacc0x0123, vb0123, va0, 0);
      vacc0x4567 = vmlaq_lane_f32(vacc0x4567, vb4567, va0, 0);
      vacc1x0123 = vmlaq_lane_f32(vacc1x0123, vb0123, va1, 0);
      vacc1x4567 = vmlaq_lane_f32(vacc1x4567, vb4567, va1, 0);
      vacc2x0123 = vmlaq_lane_f32(vacc2x0123, vb0123, va2, 0);
      vacc2x4567 = vmlaq_lane_f32(vacc2x4567, vb4567, va2, 0);
      vacc3x0123 = vmlaq_lane_f32(vacc3x0123, vb0123, va3, 0);
      vacc3x4567 = vmlaq_lane_f32(vacc3x4567, vb4567, va3, 0);
      vacc4x0123 = vmlaq_lane_f32(vacc4x0123, vb0123, va4, 0);
      vacc4x4567 = vmlaq_lane_f32(vacc4x4567, vb4567, va4, 0);
      vacc5x0123 = vmlaq_lane_f32(vacc5x0123, vb0123, va5, 0);
      vacc5x4567 = vmlaq_lane_f32(vacc5x4567, vb4567, va5, 0);
#endif
    }

    {
      const float32x4_t vb0123 = vld1q_f32(w);   // 从地址 w 处加载一个 float32x4_t 类型的向量到 vb0123
      w += 4;  // 将指针 w 向前移动 4 个字节，指向下一个向量的地址
      const float32x4_t vb4567 = vld1q_f32(w);   // 从地址 w 处加载另一个 float32x4_t 类型的向量到 vb4567
      w += 4;  // 将指针 w 向前移动 4 个字节，指向下一个向量的地址

#if defined(__aarch64__)
      // 如果是 ARM 64 位架构，使用 vfmaq_lane_f32 函数进行带有乘法累加的向量操作
      vacc0x0123 = vfmaq_lane_f32(vacc0x0123, vb0123, va0, 1);
      vacc0x4567 = vfmaq_lane_f32(vacc0x4567, vb4567, va0, 1);
      vacc1x0123 = vfmaq_lane_f32(vacc1x0123, vb0123, va1, 1);
      vacc1x4567 = vfmaq_lane_f32(vacc1x4567, vb4567, va1, 1);
      vacc2x0123 = vfmaq_lane_f32(vacc2x0123, vb0123, va2, 1);
      vacc2x4567 = vfmaq_lane_f32(vacc2x4567, vb4567, va2, 1);
      vacc3x0123 = vfmaq_lane_f32(vacc3x0123, vb0123, va3, 1);
      vacc3x4567 = vfmaq_lane_f32(vacc3x4567, vb4567, va3, 1);
      vacc4x0123 = vfmaq_lane_f32(vacc4x0123, vb0123, va4, 1);
      vacc4x4567 = vfmaq_lane_f32(vacc4x4567, vb4567, va4, 1);
      vacc5x0123 = vfmaq_lane_f32(vacc5x0123, vb0123, va5, 1);
      vacc5x4567 = vfmaq_lane_f32(vacc5x4567, vb4567, va5, 1);
#else
      // 如果不是 ARM 64 位架构，使用 vmlaq_lane_f32 函数进行向量加乘操作
      vacc0x0123 = vmlaq_lane_f32(vacc0x0123, vb0123, va0, 1);
      vacc0x4567 = vmlaq_lane_f32(vacc0x4567, vb4567, va0, 1);
      vacc1x0123 = vmlaq_lane_f32(vacc1x0123, vb0123, va1, 1);
      vacc1x4567 = vmlaq_lane_f32(vacc1x4567, vb4567, va1, 1);
      vacc2x0123 = vmlaq_lane_f32(vacc2x0123, vb0123, va2, 1);
      vacc2x4567 = vmlaq_lane_f32(vacc2x4567, vb4567, va2, 1);
      vacc3x0123 = vmlaq_lane_f32(vacc3x0123, vb0123, va3, 1);
      vacc3x4567 = vmlaq_lane_f32(vacc3x4567, vb4567, va3, 1);
      vacc4x0123 = vmlaq_lane_f32(vacc4x0123, vb0123, va4, 1);
      vacc4x4567 = vmlaq_lane_f32(vacc4x4567, vb4567, va4, 1);
      vacc5x0123 = vmlaq_lane_f32(vacc5x0123, vb0123, va5, 1);
      vacc5x4567 = vmlaq_lane_f32(vacc5x4567, vb4567, va5, 1);
#endif
    }
  }
  if (k != 0) {
    // 如果 k 不等于 0，继续处理剩余的向量操作

    // 从地址 a0 处加载一个 float32x4_t 向量，复制到 va0
    const float32x4_t va0 = vld1q_dup_f32(a0);
    // 从地址 a1 处加载一个 float32x4_t 向量，复制到 va1
    const float32x4_t va1 = vld1q_dup_f32(a1);
    // 从地址 a2 处加载一个 float32x4_t 向量，复制到 va2
    const float32x4_t va2 = vld1q_dup_f32(a2);
    // 从地址 a3 处加载一个 float32x4_t 向量，复制到 va3
    const float32x4_t va3 = vld1q_dup_f32(a3);
    // 从地址 a4 处加载一个 float32x4_t 向量，复制到 va4
    const float32x4_t va4 = vld1q_dup_f32(a4);
    // 从地址 a5 处加载一个 float32x4_t 向量，复制到 va5
    const float32x4_t va5 = vld1q_dup_f32(a5);

    // 从地址 w 处加载一个 float32x4_t 向量，复制到 vb0123
    const float32x4_t vb0123 = vld1q_f32(w);
    // 将指针 w 向前移动 4 个字节，指向下一个向量的地址
    w += 4;
    // 从地址 w 处加载一个 float32x4_t 向量，复制到 vb4567
    const float32x4_t vb4567 = vld1q_f32(w);
    // 将指针 w 向前移动 4 个字节，指向下一个向量的地址
    w += 4;

#if defined(__aarch64__)
    // 如果是 ARM 64 位架构，使用 vfmaq_f32 函数进行带有乘法累加的向量操作
    vacc0x0123 = vfmaq_f32(vacc0x0123, vb0123, va0);


这段代码看起来很专业！
    # 对向量寄存器中的值进行乘加操作，将结果累加到指定的累加寄存器中
    vacc0x4567 = vfmaq_f32(vacc0x4567, vb4567, va0);
    # 对 vacc0x4567 寄存器执行 vfmaq_f32 操作，将 vb4567 和 va0 的乘加结果累加到 vacc0x4567 中
    
    vacc1x0123 = vfmaq_f32(vacc1x0123, vb0123, va1);
    # 对 vacc1x0123 寄存器执行 vfmaq_f32 操作，将 vb0123 和 va1 的乘加结果累加到 vacc1x0123 中
    
    vacc1x4567 = vfmaq_f32(vacc1x4567, vb4567, va1);
    # 对 vacc1x4567 寄存器执行 vfmaq_f32 操作，将 vb4567 和 va1 的乘加结果累加到 vacc1x4567 中
    
    vacc2x0123 = vfmaq_f32(vacc2x0123, vb0123, va2);
    # 对 vacc2x0123 寄存器执行 vfmaq_f32 操作，将 vb0123 和 va2 的乘加结果累加到 vacc2x0123 中
    
    vacc2x4567 = vfmaq_f32(vacc2x4567, vb4567, va2);
    # 对 vacc2x4567 寄存器执行 vfmaq_f32 操作，将 vb4567 和 va2 的乘加结果累加到 vacc2x4567 中
    
    vacc3x0123 = vfmaq_f32(vacc3x0123, vb0123, va3);
    # 对 vacc3x0123 寄存器执行 vfmaq_f32 操作，将 vb0123 和 va3 的乘加结果累加到 vacc3x0123 中
    
    vacc3x4567 = vfmaq_f32(vacc3x4567, vb4567, va3);
    # 对 vacc3x4567 寄存器执行 vfmaq_f32 操作，将 vb4567 和 va3 的乘加结果累加到 vacc3x4567 中
    
    vacc4x0123 = vfmaq_f32(vacc4x0123, vb0123, va4);
    # 对 vacc4x0123 寄存器执行 vfmaq_f32 操作，将 vb0123 和 va4 的乘加结果累加到 vacc4x0123 中
    
    vacc4x4567 = vfmaq_f32(vacc4x4567, vb4567, va4);
    # 对 vacc4x4567 寄存器执行 vfmaq_f32 操作，将 vb4567 和 va4 的乘加结果累加到 vacc4x4567 中
    
    vacc5x0123 = vfmaq_f32(vacc5x0123, vb0123, va5);
    # 对 vacc5x0123 寄存器执行 vfmaq_f32 操作，将 vb0123 和 va5 的乘加结果累加到 vacc5x0123 中
    
    vacc5x4567 = vfmaq_f32(vacc5x4567, vb4567, va5);
    # 对 vacc5x4567 寄存器执行 vfmaq_f32 操作，将 vb4567 和 va5 的乘加结果累加到 vacc5x4567 中
#else
    // 如果不是条件满足的情况，执行以下操作：

    // 计算累加器的乘积并累加到累加器向量中
    vacc0x0123 = vmlaq_f32(vacc0x0123, vb0123, va0);
    vacc0x4567 = vmlaq_f32(vacc0x4567, vb4567, va0);
    vacc1x0123 = vmlaq_f32(vacc1x0123, vb0123, va1);
    vacc1x4567 = vmlaq_f32(vacc1x4567, vb4567, va1);
    vacc2x0123 = vmlaq_f32(vacc2x0123, vb0123, va2);
    vacc2x4567 = vmlaq_f32(vacc2x4567, vb4567, va2);
    vacc3x0123 = vmlaq_f32(vacc3x0123, vb0123, va3);
    vacc3x4567 = vmlaq_f32(vacc3x4567, vb4567, va3);
    vacc4x0123 = vmlaq_f32(vacc4x0123, vb0123, va4);
    vacc4x4567 = vmlaq_f32(vacc4x4567, vb4567, va4);
    vacc5x0123 = vmlaq_f32(vacc5x0123, vb0123, va5);
    vacc5x4567 = vmlaq_f32(vacc5x4567, vb4567, va5);
#endif

  }

  // 从输入的 clamping_params 结构体中加载最大值并复制到 vmax 向量中
  const float32x4_t vmax = vld1q_dup_f32(&clamping_params->max);

  // 将累加器向量中的值与 vmax 向量中的最大值进行比较，保留较小的值
  vacc0x0123 = vminq_f32(vacc0x0123, vmax);
  vacc0x4567 = vminq_f32(vacc0x4567, vmax);
  vacc1x0123 = vminq_f32(vacc1x0123, vmax);
  vacc1x4567 = vminq_f32(vacc1x4567, vmax);
  vacc2x0123 = vminq_f32(vacc2x0123, vmax);
  vacc2x4567 = vminq_f32(vacc2x4567, vmax);
  vacc3x0123 = vminq_f32(vacc3x0123, vmax);
  vacc3x4567 = vminq_f32(vacc3x4567, vmax);
  vacc4x0123 = vminq_f32(vacc4x0123, vmax);
  vacc4x4567 = vminq_f32(vacc4x4567, vmax);
  vacc5x0123 = vminq_f32(vacc5x0123, vmax);
  vacc5x4567 = vminq_f32(vacc5x4567, vmax);

  // 从输入的 clamping_params 结构体中加载最小值并复制到 vmin 向量中
  const float32x4_t vmin = vld1q_dup_f32(&clamping_params->min);

  // 将累加器向量中的值与 vmin 向量中的最小值进行比较，保留较大的值
  vacc0x0123 = vmaxq_f32(vacc0x0123, vmin);
  vacc0x4567 = vmaxq_f32(vacc0x4567, vmin);
  vacc1x0123 = vmaxq_f32(vacc1x0123, vmin);
  vacc1x4567 = vmaxq_f32(vacc1x4567, vmin);
  vacc2x0123 = vmaxq_f32(vacc2x0123, vmin);
  vacc2x4567 = vmaxq_f32(vacc2x4567, vmin);
  vacc3x0123 = vmaxq_f32(vacc3x0123, vmin);
  vacc3x4567 = vmaxq_f32(vacc3x4567, vmin);
  vacc4x0123 = vmaxq_f32(vacc4x0123, vmin);
  vacc4x4567 = vmaxq_f32(vacc4x4567, vmin);
  vacc5x0123 = vmaxq_f32(vacc5x0123, vmin);
  vacc5x4567 = vmaxq_f32(vacc5x4567, vmin);

  // 初始化输出矩阵指针 c0 和 c1
  float* c0 = c;
  float* c1 = (float*)((uintptr_t)c0 + c_stride);

  // 如果 mr 小于 2，则将 c1 指向 c0，否则继续偏移 c1
  if (mr < 2) {
    c1 = c0;
  }

  // 初始化输出矩阵指针 c2
  float* c2 = (float*)((uintptr_t)c1 + c_stride);

  // 如果 mr 小于等于 2，则将 c2 指向 c1，否则继续偏移 c2
  if (mr <= 2) {
    c2 = c1;
  }

  // 初始化输出矩阵指针 c3
  float* c3 = (float*)((uintptr_t)c2 + c_stride);

  // 如果 mr 小于 4，则将 c3 指向 c2，否则继续偏移 c3
  if (mr < 4) {
    c3 = c2;
  }

  // 初始化输出矩阵指针 c4
  float* c4 = (float*)((uintptr_t)c3 + c_stride);

  // 如果 mr 小于等于 4，则将 c4 指向 c3，否则继续偏移 c4
  if (mr <= 4) {
    c4 = c3;
  }

  // 初始化输出矩阵指针 c5
  float* c5 = (float*)((uintptr_t)c4 + c_stride);

  // 如果 mr 不等于 6，则将 c5 指向 c4，否则保持 c5 不变
  if (mr != 6) {
    c5 = c4;
  }

  // 如果 nr 等于 8，将累加器向量的值写入 c0 到 c5 指向的内存位置，并更新指针
  if (nr == 8) {
    vst1q_f32(c0, vacc0x0123);
    c0 += 4;
    vst1q_f32(c1, vacc1x0123);
    c1 += 4;
    vst1q_f32(c2, vacc2x0123);
    c2 += 4;
    vst1q_f32(c3, vacc3x0123);
    c3 += 4;
    vst1q_f32(c4, vacc4x0123);
    c4 += 4;
    vst1q_f32(c5, vacc5x0123);
    c5 += 4;

    vst1q_f32(c0, vacc0x4567);
    vst1q_f32(c1, vacc1x4567);
    vst1q_f32(c2, vacc2x4567);
    vst1q_f32(c3, vacc3x4567);
    vst1q_f32(c4, vacc4x4567);
    vst1q_f32(c5, vacc5x4567);
  } else {
    # 如果剩余寄存器数大于等于4，则执行以下操作
    if (nr >= 4) {
      # 将寄存器中的四个浮点数存储到c0指向的地址处，并移动c0指针
      vst1q_f32(c0, vacc0x0123);
      c0 += 4;
      # 同上，存储到c1并移动c1指针
      vst1q_f32(c1, vacc1x0123);
      c1 += 4;
      # 同上，存储到c2并移动c2指针
      vst1q_f32(c2, vacc2x0123);
      c2 += 4;
      # 同上，存储到c3并移动c3指针
      vst1q_f32(c3, vacc3x0123);
      c3 += 4;
      # 同上，存储到c4并移动c4指针
      vst1q_f32(c4, vacc4x0123);
      c4 += 4;
      # 同上，存储到c5并移动c5指针
      vst1q_f32(c5, vacc5x0123);
      c5 += 4;
      # 更新寄存器中的数值为下一个四个浮点数数据
      vacc0x0123 = vacc0x4567;
      vacc1x0123 = vacc1x4567;
      vacc2x0123 = vacc2x4567;
      vacc3x0123 = vacc3x4567;
      vacc4x0123 = vacc4x4567;
      vacc5x0123 = vacc5x4567;
      # 减少剩余寄存器数
      nr -= 4;
    }
    # 如果剩余寄存器数大于等于2，则执行以下操作
    if (nr >= 2) {
      # 将寄存器中的前两个浮点数的低位存储到c0指向的地址处，并移动c0指针
      vst1_f32(c0, vget_low_f32(vacc0x0123));
      c0 += 2;
      # 同上，存储到c1并移动c1指针
      vst1_f32(c1, vget_low_f32(vacc1x0123));
      c1 += 2;
      # 同上，存储到c2并移动c2指针
      vst1_f32(c2, vget_low_f32(vacc2x0123));
      c2 += 2;
      # 同上，存储到c3并移动c3指针
      vst1_f32(c3, vget_low_f32(vacc3x0123));
      c3 += 2;
      # 同上，存储到c4并移动c4指针
      vst1_f32(c4, vget_low_f32(vacc4x0123));
      c4 += 2;
      # 同上，存储到c5并移动c5指针
      vst1_f32(c5, vget_low_f32(vacc5x0123));
      c5 += 2;
      # 将寄存器中的数据向左移动两个位置，更新寄存器内容
      vacc0x0123 = vextq_f32(vacc0x0123, vacc0x0123, 2);
      vacc1x0123 = vextq_f32(vacc1x0123, vacc1x0123, 2);
      vacc2x0123 = vextq_f32(vacc2x0123, vacc2x0123, 2);
      vacc3x0123 = vextq_f32(vacc3x0123, vacc3x0123, 2);
      vacc4x0123 = vextq_f32(vacc4x0123, vacc4x0123, 2);
      vacc5x0123 = vextq_f32(vacc5x0123, vacc5x0123, 2);
      # 减少剩余寄存器数
      nr -= 2;
    }
    # 如果剩余寄存器数不为0，则执行以下操作
    if (nr != 0) {
      # 将寄存器中的单个浮点数数据存储到c0指向的地址处
      vst1q_lane_f32(c0, vacc0x0123, 0);
      # 同上，存储到c1指向的地址处
      vst1q_lane_f32(c1, vacc1x0123, 0);
      # 同上，存储到c2指向的地址处
      vst1q_lane_f32(c2, vacc2x0123, 0);
      # 同上，存储到c3指向的地址处
      vst1q_lane_f32(c3, vacc3x0123, 0);
      # 同上，存储到c4指向的地址处
      vst1q_lane_f32(c4, vacc4x0123, 0);
      # 同上，存储到c5指向的地址处
      vst1q_lane_f32(c5, vacc5x0123, 0);
    }
}


注释：


# 这是一个代码块的结束标记，表示当前的代码块结束。
```