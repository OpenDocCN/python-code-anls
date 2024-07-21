# `.\pytorch\aten\src\ATen\native\quantized\cpu\qnnpack\src\sgemm\5x8-neon.c`

```py
/*
 * 包含头文件 arm_neon.h，这是 ARM 平台的 NEON 向量指令库
 */
#include <arm_neon.h>

/*
 * 包含头文件 qnnpack/sgemm.h，可能是一个库文件，定义了 sgemm 函数
 */
#include <qnnpack/sgemm.h>

/*
 * 定义了一个名为 pytorch_sgemm_ukernel_5x8__neon 的函数
 * 这个函数使用 NEON 指令集执行矩阵乘法加法操作
 */
void pytorch_sgemm_ukernel_5x8__neon(
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
  
  /*
   * 从指针 w 处加载 4 个单精度浮点数到寄存器 vacc0x0123 和 vacc0x4567
   */
  float32x4_t vacc0x0123 = vld1q_f32(w);
  w += 4;
  float32x4_t vacc0x4567 = vld1q_f32(w);
  w += 4;
  
  /*
   * 复制寄存器的值，初始化其余 4 个累加器寄存器 vacc1x0123 到 vacc4x4567
   */
  float32x4_t vacc1x0123 = vacc0x0123;
  float32x4_t vacc1x4567 = vacc0x4567;
  float32x4_t vacc2x0123 = vacc0x0123;
  float32x4_t vacc2x4567 = vacc0x4567;
  float32x4_t vacc3x0123 = vacc0x0123;
  float32x4_t vacc3x4567 = vacc0x4567;
  float32x4_t vacc4x0123 = vacc0x0123;
  float32x4_t vacc4x4567 = vacc0x4567;

  /*
   * 初始化指针 a0 - a4，分别指向矩阵 A 的不同行
   */
  const float* a0 = a;
  const float* a1 = (const float*)((uintptr_t)a0 + a_stride);
  if (mr < 2) {
    a1 = a0;
  }
  const float* a2 = (const float*)((uintptr_t)a1 + a_stride);
  if (mr <= 2) {
    a2 = a1;
  }
  const float* a3 = (const float*)((uintptr_t)a2 + a_stride);
  if (mr < 4) {
    a3 = a2;
  }
  const float* a4 = (const float*)((uintptr_t)a3 + a_stride);
  if (mr <= 4) {
    a4 = a3;
  }

  /*
   * 使用 NEON 指令处理矩阵乘法和加法
   */
  for (; k >= 2; k -= 2) {
    const float32x2_t va0 = vld1_f32(a0);
    a0 += 2;
    const float32x2_t va1 = vld1_f32(a1);
    a1 += 2;
    const float32x2_t va2 = vld1_f32(a2);
    a2 += 2;
    const float32x2_t va3 = vld1_f32(a3);
    a3 += 2;
    const float32x2_t va4 = vld1_f32(a4);
    a4 += 2;

    /*
     * 加载权重向量 w，执行乘法累加操作到累加器寄存器
     */
    {
      const float32x4_t vb0123 = vld1q_f32(w);
      w += 4;
      const float32x4_t vb4567 = vld1q_f32(w);
      w += 4;

#if defined(__aarch64__)
      vacc0x0123 = vfmaq_lane_f32(vacc0x0123, vb0123, va0, 0);
      vacc0x4567 = vfmaq_lane_f32(vacc0x4567, vb4567, va0, 0);
      vacc1x0123 = vfmaq_lane_f32(vacc1x0123, vb0123, va1, 0);
      vacc1x4567 = vfmaq_lane_f32(vacc1x4567, vb4567, va1, 0);
      vacc2x0123 = vfmaq_lane_f32(vacc2x0123, vb0123, va2, 0);
      vacc2x4567 = vfmaq_lane_f32(vacc2x4567, vb4567, va2, 0);
      vacc3x0123 = vfmaq_lane_f32(vacc3x0123, vb0123, va3, 0);
      vacc3x4567 = vfmaq_lane_f32(vacc3x4567, vb4567, va3, 0);
      vacc4x0123 = vfmaq_lane_f32(vacc4x0123, vb0123, va4, 0);
      vacc4x4567 = vfmaq_lane_f32(vacc4x4567, vb4567, va4, 0);
#else
      // 如果不是 __aarch64__ 架构，则使用 vmlaq_lane_f32 进行向量乘加操作
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
#endif
    }

    {
      const float32x4_t vb0123 = vld1q_f32(w);
      w += 4;
      const float32x4_t vb4567 = vld1q_f32(w);
      w += 4;

#if defined(__aarch64__)
      // 如果是 __aarch64__ 架构，则使用 vfmaq_lane_f32 进行向量乘加操作
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
#else
      // 如果不是 __aarch64__ 架构，则继续使用 vmlaq_lane_f32 进行向量乘加操作
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
#endif
    }
  }
  if (k != 0) {
    // 对剩余的 k 不为 0 的情况处理
    const float32x4_t va0 = vld1q_dup_f32(a0);
    const float32x4_t va1 = vld1q_dup_f32(a1);
    const float32x4_t va2 = vld1q_dup_f32(a2);
    const float32x4_t va3 = vld1q_dup_f32(a3);
    const float32x4_t va4 = vld1q_dup_f32(a4);

    const float32x4_t vb0123 = vld1q_f32(w);
    w += 4;
    const float32x4_t vb4567 = vld1q_f32(w);
    w += 4;

#if defined(__aarch64__)
    // 如果是 __aarch64__ 架构，则使用 vfmaq_f32 进行向量乘加操作
    vacc0x0123 = vfmaq_f32(vacc0x0123, vb0123, va0);
    vacc0x4567 = vfmaq_f32(vacc0x4567, vb4567, va0);
    vacc1x0123 = vfmaq_f32(vacc1x0123, vb0123, va1);
    vacc1x4567 = vfmaq_f32(vacc1x4567, vb4567, va1);
    vacc2x0123 = vfmaq_f32(vacc2x0123, vb0123, va2);
    vacc2x4567 = vfmaq_f32(vacc2x4567, vb4567, va2);
    vacc3x0123 = vfmaq_f32(vacc3x0123, vb0123, va3);
    vacc3x4567 = vfmaq_f32(vacc3x4567, vb4567, va3);
    vacc4x0123 = vfmaq_f32(vacc4x0123, vb0123, va4);
#else
    // 如果不是 __aarch64__ 架构，则继续使用 vmlaq_f32 进行向量乘加操作
    vacc0x0123 = vmlaq_f32(vacc0x0123, vb0123, va0);
    vacc0x4567 = vmlaq_f32(vacc0x4567, vb4567, va0);
    vacc1x0123 = vmlaq_f32(vacc1x0123, vb0123, va1);
    vacc1x4567 = vmlaq_f32(vacc1x4567, vb4567, va1);
    vacc2x0123 = vmlaq_f32(vacc2x0123, vb0123, va2);
    vacc2x4567 = vmlaq_f32(vacc2x4567, vb4567, va2);
    vacc3x0123 = vmlaq_f32(vacc3x0123, vb0123, va3);
    vacc3x4567 = vmlaq_f32(vacc3x4567, vb4567, va3);
    vacc4x0123 = vmlaq
    // 使用 NEON 指令 vfmaq_f32 计算乘加操作，将结果累加到 vacc4x4567 中
    vacc4x4567 = vfmaq_f32(vacc4x4567, vb4567, va4);
#else
    // 对 vmlaq_f32 函数的调用，执行向量乘法和累加操作，结果存储在 vacc0x0123 中
    vacc0x0123 = vmlaq_f32(vacc0x0123, vb0123, va0);
    // 对 vmlaq_f32 函数的调用，执行向量乘法和累加操作，结果存储在 vacc0x4567 中
    vacc0x4567 = vmlaq_f32(vacc0x4567, vb4567, va0);
    // 对 vmlaq_f32 函数的调用，执行向量乘法和累加操作，结果存储在 vacc1x0123 中
    vacc1x0123 = vmlaq_f32(vacc1x0123, vb0123, va1);
    // 对 vmlaq_f32 函数的调用，执行向量乘法和累加操作，结果存储在 vacc1x4567 中
    vacc1x4567 = vmlaq_f32(vacc1x4567, vb4567, va1);
    // 对 vmlaq_f32 函数的调用，执行向量乘法和累加操作，结果存储在 vacc2x0123 中
    vacc2x0123 = vmlaq_f32(vacc2x0123, vb0123, va2);
    // 对 vmlaq_f32 函数的调用，执行向量乘法和累加操作，结果存储在 vacc2x4567 中
    vacc2x4567 = vmlaq_f32(vacc2x4567, vb4567, va2);
    // 对 vmlaq_f32 函数的调用，执行向量乘法和累加操作，结果存储在 vacc3x0123 中
    vacc3x0123 = vmlaq_f32(vacc3x0123, vb0123, va3);
    // 对 vmlaq_f32 函数的调用，执行向量乘法和累加操作，结果存储在 vacc3x4567 中
    vacc3x4567 = vmlaq_f32(vacc3x4567, vb4567, va3);
    // 对 vmlaq_f32 函数的调用，执行向量乘法和累加操作，结果存储在 vacc4x0123 中
    vacc4x0123 = vmlaq_f32(vacc4x0123, vb0123, va4);
    // 对 vmlaq_f32 函数的调用，执行向量乘法和累加操作，结果存储在 vacc4x4567 中
    vacc4x4567 = vmlaq_f32(vacc4x4567, vb4567, va4);
#endif
  }
  // 使用 clamping_params 中的最大值创建一个常量 SIMD 向量
  const float32x4_t vmax = vld1q_dup_f32(&clamping_params->max);
  // 对所有 vaccx0123 向量执行最小值截断操作，结果存储回原向量
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

  // 使用 clamping_params 中的最小值创建一个常量 SIMD 向量
  const float32x4_t vmin = vld1q_dup_f32(&clamping_params->min);
  // 对所有 vaccx0123 向量执行最大值截断操作，结果存储回原向量
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

  // 初始化指向输出矩阵 c 的指针
  float* c0 = c;
  // 初始化指向输出矩阵 c1 的指针，偏移量为 c_stride
  float* c1 = (float*)((uintptr_t)c0 + c_stride);
  // 如果 mr 小于 2，将 c1 指向 c0，实现复用
  if (mr < 2) {
    c1 = c0;
  }
  // 初始化指向输出矩阵 c2 的指针，偏移量为 c_stride
  float* c2 = (float*)((uintptr_t)c1 + c_stride);
  // 如果 mr 小于等于 2，将 c2 指向 c1，实现复用
  if (mr <= 2) {
    c2 = c1;
  }
  // 初始化指向输出矩阵 c3 的指针，偏移量为 c_stride
  float* c3 = (float*)((uintptr_t)c2 + c_stride);
  // 如果 mr 小于 4，将 c3 指向 c2，实现复用
  if (mr < 4) {
    c3 = c2;
  }
  // 初始化指向输出矩阵 c4 的指针，偏移量为 c_stride
  float* c4 = (float*)((uintptr_t)c3 + c_stride);
  // 如果 mr 小于等于 4，将 c4 指向 c3，实现复用
  if (mr <= 4) {
    c4 = c3;
  }
  // 如果 nr 等于 8，则将向量结果存储到输出矩阵中
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

    vst1q_f32(c0, vacc0x4567);
    vst1q_f32(c1, vacc1x4567);
    vst1q_f32(c2, vacc2x4567);
    vst1q_f32(c3, vacc3x4567);
    vst1q_f32(c4, vacc4x4567);
  } else {
    // 如果 nr 大于等于 4，则将向量结果存储到输出矩阵中
    if (nr >= 4) {
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
      // 将后半部分的向量结果拷贝到前半部分，以减少处理的数据量
      vacc0x0123 = vacc0x4567;
      vacc1x0123 = vacc1x4567;
      vacc2x0123 = vacc2x4567;
      vacc3x0123 = vacc3x4567;
      vacc4x0123 = vacc4
    # 如果剩余可处理的向量寄存器数量大于等于2，则执行以下操作
    if (nr >= 2) {
      # 将vacc0x0123寄存器中低位的四个单精度浮点数存储到地址c0指向的内存中，并将c0向后移动2个单精度浮点数的位置
      vst1_f32(c0, vget_low_f32(vacc0x0123));
      c0 += 2;
      # 类似地，将vacc1x0123寄存器中低位的四个单精度浮点数存储到地址c1指向的内存中，并将c1向后移动2个单精度浮点数的位置
      vst1_f32(c1, vget_low_f32(vacc1x0123));
      c1 += 2;
      # 类似地，将vacc2x0123寄存器中低位的四个单精度浮点数存储到地址c2指向的内存中，并将c2向后移动2个单精度浮点数的位置
      vst1_f32(c2, vget_low_f32(vacc2x0123));
      c2 += 2;
      # 类似地，将vacc3x0123寄存器中低位的四个单精度浮点数存储到地址c3指向的内存中，并将c3向后移动2个单精度浮点数的位置
      vst1_f32(c3, vget_low_f32(vacc3x0123));
      c3 += 2;
      # 类似地，将vacc4x0123寄存器中低位的四个单精度浮点数存储到地址c4指向的内存中，并将c4向后移动2个单精度浮点数的位置
      vst1_f32(c4, vget_low_f32(vacc4x0123));
      c4 += 2;
      # 将vacc0x0123寄存器中的数据向左循环移动两个位置，即第0、1位置移到第2、3位置
      vacc0x0123 = vextq_f32(vacc0x0123, vacc0x0123, 2);
      # 将vacc1x0123寄存器中的数据向左循环移动两个位置，即第0、1位置移到第2、3位置
      vacc1x0123 = vextq_f32(vacc1x0123, vacc1x0123, 2);
      # 将vacc2x0123寄存器中的数据向左循环移动两个位置，即第0、1位置移到第2、3位置
      vacc2x0123 = vextq_f32(vacc2x0123, vacc2x0123, 2);
      # 将vacc3x0123寄存器中的数据向左循环移动两个位置，即第0、1位置移到第2、3位置
      vacc3x0123 = vextq_f32(vacc3x0123, vacc3x0123, 2);
      # 将vacc4x0123寄存器中的数据向左循环移动两个位置，即第0、1位置移到第2、3位置
      vacc4x0123 = vextq_f32(vacc4x0123, vacc4x0123, 2);
      # 更新剩余可处理的向量寄存器数量，减去已处理的2个
      nr -= 2;
    }
    # 如果剩余可处理的向量寄存器数量不等于0，则执行以下操作
    if (nr != 0) {
      # 将vacc0x0123寄存器中第0个单精度浮点数存储到地址c0指向的内存中
      vst1q_lane_f32(c0, vacc0x0123, 0);
      # 将vacc1x0123寄存器中第0个单精度浮点数存储到地址c1指向的内存中
      vst1q_lane_f32(c1, vacc1x0123, 0);
      # 将vacc2x0123寄存器中第0个单精度浮点数存储到地址c2指向的内存中
      vst1q_lane_f32(c2, vacc2x0123, 0);
      # 将vacc3x0123寄存器中第0个单精度浮点数存储到地址c3指向的内存中
      vst1q_lane_f32(c3, vacc3x0123, 0);
      # 将vacc4x0123寄存器中第0个单精度浮点数存储到地址c4指向的内存中
      vst1q_lane_f32(c4, vacc4x0123, 0);
    }
}



# 这是一个单独的右花括号 '}'，用于结束一个代码块或语句。在这个上下文中，它应该是某个代码结构（如函数、条件语句、循环等）的闭合。
```