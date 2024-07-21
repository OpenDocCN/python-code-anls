# `.\pytorch\aten\src\ATen\native\quantized\cpu\qnnpack\src\hgemm\8x8-neonfp16arith.c`

```py
/*
 * 导入 ARM NEON 头文件，以便使用 NEON 指令集优化加速计算
 */
#include <arm_neon.h>

/*
 * 导入 QNNPACK 中的半精度矩阵乘法头文件
 */
#include <qnnpack/hgemm.h>

/*
 * 定义一个 NEON 加速的 8x8 矩阵乘法的函数，用于半精度浮点运算
 *
 * @param mr        指定 A 矩阵的行数
 * @param nr        指定 B 矩阵的列数
 * @param k         指定 A 矩阵的列数 / B 矩阵的行数
 * @param a         指向 A 矩阵的指针
 * @param a_stride  A 矩阵的行步长
 * @param w         指向 B 矩阵的指针
 * @param c         指向 C 矩阵的指针
 * @param c_stride  C 矩阵的行步长
 * @param clamping_params  半精度浮点数的夹紧参数
 */
void pytorch_hgemm_ukernel_8x8__neonfp16arith(
    size_t mr,
    size_t nr,
    size_t k,
    const void* restrict a,
    size_t a_stride,
    const void* restrict w,
    void* restrict c,
    size_t c_stride,
    const struct pytorch_qnnp_fp16_clamping_params
        clamping_params[restrict static 1]) {
  
  /*
   * 加载 B 矩阵的起始部分，即一列数据到 NEON 寄存器中
   */
  float16x8_t vacc0x01234567 = vld1q_f16(w);

  /*
   * 移动指针 w 到下一列数据的起始位置
   */
  w = (void*)((uintptr_t)w + sizeof(float16x8_t));

  /*
   * 复制 B 矩阵的起始列数据到其他寄存器，以便并行计算
   */
  float16x8_t vacc1x01234567 = vacc0x01234567;
  float16x8_t vacc2x01234567 = vacc0x01234567;
  float16x8_t vacc3x01234567 = vacc0x01234567;
  float16x8_t vacc4x01234567 = vacc0x01234567;
  float16x8_t vacc5x01234567 = vacc0x01234567;
  float16x8_t vacc6x01234567 = vacc0x01234567;
  float16x8_t vacc7x01234567 = vacc0x01234567;

  /*
   * 初始化 A 矩阵的指针，根据矩阵行数 mr 来适应不同情况
   */
  const __fp16* a0 = a;
  const __fp16* a1 = (const __fp16*)((uintptr_t)a0 + a_stride);
  if (mr < 2) {
    a1 = a0;
  }
  const __fp16* a2 = (const __fp16*)((uintptr_t)a1 + a_stride);
  if (mr <= 2) {
    a2 = a1;
  }
  const __fp16* a3 = (const __fp16*)((uintptr_t)a2 + a_stride);
  if (mr < 4) {
    a3 = a2;
  }
  const __fp16* a4 = (const __fp16*)((uintptr_t)a3 + a_stride);
  if (mr <= 4) {
    a4 = a3;
  }
  const __fp16* a5 = (const __fp16*)((uintptr_t)a4 + a_stride);
  if (mr < 6) {
    a5 = a4;
  }
  const __fp16* a6 = (const __fp16*)((uintptr_t)a5 + a_stride);
  if (mr <= 6) {
    a6 = a5;
  }
  const __fp16* a7 = (const __fp16*)((uintptr_t)a6 + a_stride);
  if (mr != 8) {
    a7 = a6;
  }

  /*
   * 循环计算主要的乘加操作，每次处理四列数据
   */
  for (; k >= 4; k -= 4) {
    const float16x4_t va0 = vld1_f16(a0);
    a0 += 4;
    const float16x4_t va1 = vld1_f16(a1);
    a1 += 4;
    const float16x4_t va2 = vld1_f16(a2);
    a2 += 4;
    const float16x4_t va3 = vld1_f16(a3);
    a3 += 4;
    const float16x4_t va4 = vld1_f16(a4);
    a4 += 4;
    const float16x4_t va5 = vld1_f16(a5);
    a5 += 4;
    const float16x4_t va6 = vld1_f16(a6);
    a6 += 4;
    const float16x4_t va7 = vld1_f16(a7);
    a7 += 4;

    /*
     * 加载下一列 B 矩阵数据到 NEON 寄存器
     */
    const float16x8_t vb01234567 = vld1q_f16(w);
    w = (void*)((uintptr_t)w + sizeof(float16x8_t));

    /*
     * 使用 NEON 指令进行乘加运算，更新累加器寄存器的值
     */
    vacc0x01234567 = vmlaq_lane_f16(vacc0x01234567, vb01234567, va0, 0);
    vacc1x01234567 = vmlaq_lane_f16(vacc1x01234567, vb01234567, va1, 0);
    vacc2x01234567 = vmlaq_lane_f16(vacc2x01234567, vb01234567, va2, 0);
    vacc3x01234567 = vmlaq_lane_f16(vacc3x01234567, vb01234567, va3, 0);
    vacc4x01234567 = vmlaq_lane_f16(vacc4x01234567, vb01234567, va4, 0);
    vacc5x01234567 = vmlaq_lane_f16(vacc5x01234567, vb01234567, va5, 0);
    vacc6x01234567 = vmlaq_lane_f16(vacc6x01234567, vb01234567, va6, 0);
    vacc7x01234567 = vmlaq_lane_f16(vacc7x01234567, vb01234567, va7, 0);
  }
    {
      // 从指针 w 处加载一个 float16x8_t 类型的向量，并存储到 vb01234567 中
      const float16x8_t vb01234567 = vld1q_f16(w);
      // 将指针 w 向前移动 sizeof(float16x8_t) 字节
      w = (void*)((uintptr_t)w + sizeof(float16x8_t));
    
      // 使用 va0 中的第 1 个元素对 vb01234567 中的每个元素进行乘法累加，结果存储到 vacc0x01234567
      vacc0x01234567 = vmlaq_lane_f16(vacc0x01234567, vb01234567, va0, 1);
      // 使用 va1 中的第 1 个元素对 vb01234567 中的每个元素进行乘法累加，结果存储到 vacc1x01234567
      vacc1x01234567 = vmlaq_lane_f16(vacc1x01234567, vb01234567, va1, 1);
      // 使用 va2 中的第 1 个元素对 vb01234567 中的每个元素进行乘法累加，结果存储到 vacc2x01234567
      vacc2x01234567 = vmlaq_lane_f16(vacc2x01234567, vb01234567, va2, 1);
      // 使用 va3 中的第 1 个元素对 vb01234567 中的每个元素进行乘法累加，结果存储到 vacc3x01234567
      vacc3x01234567 = vmlaq_lane_f16(vacc3x01234567, vb01234567, va3, 1);
      // 使用 va4 中的第 1 个元素对 vb01234567 中的每个元素进行乘法累加，结果存储到 vacc4x01234567
      vacc4x01234567 = vmlaq_lane_f16(vacc4x01234567, vb01234567, va4, 1);
      // 使用 va5 中的第 1 个元素对 vb01234567 中的每个元素进行乘法累加，结果存储到 vacc5x01234567
      vacc5x01234567 = vmlaq_lane_f16(vacc5x01234567, vb01234567, va5, 1);
      // 使用 va6 中的第 1 个元素对 vb01234567 中的每个元素进行乘法累加，结果存储到 vacc6x01234567
      vacc6x01234567 = vmlaq_lane_f16(vacc6x01234567, vb01234567, va6, 1);
      // 使用 va7 中的第 1 个元素对 vb01234567 中的每个元素进行乘法累加，结果存储到 vacc7x01234567
      vacc7x01234567 = vmlaq_lane_f16(vacc7x01234567, vb01234567, va7, 1);
    }
    
    {
      // 从指针 w 处加载一个 float16x8_t 类型的向量，并存储到 vb01234567 中
      const float16x8_t vb01234567 = vld1q_f16(w);
      // 将指针 w 向前移动 sizeof(float16x8_t) 字节
      w = (void*)((uintptr_t)w + sizeof(float16x8_t));
    
      // 使用 va0 中的第 2 个元素对 vb01234567 中的每个元素进行乘法累加，结果存储到 vacc0x01234567
      vacc0x01234567 = vmlaq_lane_f16(vacc0x01234567, vb01234567, va0, 2);
      // 使用 va1 中的第 2 个元素对 vb01234567 中的每个元素进行乘法累加，结果存储到 vacc1x01234567
      vacc1x01234567 = vmlaq_lane_f16(vacc1x01234567, vb01234567, va1, 2);
      // 使用 va2 中的第 2 个元素对 vb01234567 中的每个元素进行乘法累加，结果存储到 vacc2x01234567
      vacc2x01234567 = vmlaq_lane_f16(vacc2x01234567, vb01234567, va2, 2);
      // 使用 va3 中的第 2 个元素对 vb01234567 中的每个元素进行乘法累加，结果存储到 vacc3x01234567
      vacc3x01234567 = vmlaq_lane_f16(vacc3x01234567, vb01234567, va3, 2);
      // 使用 va4 中的第 2 个元素对 vb01234567 中的每个元素进行乘法累加，结果存储到 vacc4x01234567
      vacc4x01234567 = vmlaq_lane_f16(vacc4x01234567, vb01234567, va4, 2);
      // 使用 va5 中的第 2 个元素对 vb01234567 中的每个元素进行乘法累加，结果存储到 vacc5x01234567
      vacc5x01234567 = vmlaq_lane_f16(vacc5x01234567, vb01234567, va5, 2);
      // 使用 va6 中的第 2 个元素对 vb01234567 中的每个元素进行乘法累加，结果存储到 vacc6x01234567
      vacc6x01234567 = vmlaq_lane_f16(vacc6x01234567, vb01234567, va6, 2);
      // 使用 va7 中的第 2 个元素对 vb01234567 中的每个元素进行乘法累加，结果存储到 vacc7x01234567
      vacc7x01234567 = vmlaq_lane_f16(vacc7x01234567, vb01234567, va7, 2);
    }
    
    {
      // 从指针 w 处加载一个 float16x8_t 类型的向量，并存储到 vb01234567 中
      const float16x8_t vb01234567 = vld1q_f16(w);
      // 将指针 w 向前移动 sizeof(float16x8_t) 字节
      w = (void*)((uintptr_t)w + sizeof(float16x8_t));
    
      // 使用 va0 中的第 3 个元素对 vb01234567 中的每个元素进行乘法累加，结果存储到 vacc0x01234567
      vacc0x01234567 = vmlaq_lane_f16(vacc0x01234567, vb01234567, va0, 3);
      // 使用 va1 中的第 3 个元素对 vb01234567 中的每个元素进行乘法累加，结果存储到 vacc1x01234567
      vacc1x01234567 = vmlaq_lane_f16(vacc1x01234567, vb01234567, va1, 3);
      // 使用 va2 中的第 3 个元素对 vb01234567 中的每个元素进行乘法累加，结果存储到 vacc2x01234567
      vacc2x01234567 = vmlaq_lane_f16(vacc2x01234567, vb01234567, va2, 3);
      // 使用 va3 中的第 3 个元素对 vb01234567 中的每个元素进行乘法累加，结果存储到 vacc3x01234567
      vacc3x01234567 = vmlaq_lane_f16(vacc3x01234567, vb01234567, va3, 3);
      // 使用 va4 中的第 3 个元素对 vb01234567 中的每个元素进行乘法累加，结果存储到 vacc4x01234567
      vacc4x01234567 = vmlaq_lane_f16(vacc4x01234567, vb01234567, va4, 3);
      // 使用 va5 中的第 3 个元素对 vb01234567 中的每个元素进行乘法累加，结果存储到 vacc5x01234567
      vacc5x01234567 = vmlaq_lane_f16(vacc5x01234567, vb01234567, va5, 3);
      // 使用 va6 中的第 3 个元素对 vb01234567 中的每个元素进行乘法累加，结果存储到 vacc6x012
    // 从内存地址 a5 - a_predecrement 处加载 float16x4_t 数据，转换为 u64，左移 va_shift 位后再转换为 float16x4_t 类型的向量 va5
    const float16x4_t va5 = vreinterpret_f16_u64(vshl_u64(
        vreinterpret_u64_f16(vld1_f16(a5 - a_predecrement)), va_shift));
    
    // 同上，加载 a6 - a_predecrement 处的数据并生成 va6 向量
    const float16x4_t va6 = vreinterpret_f16_u64(vshl_u64(
        vreinterpret_u64_f16(vld1_f16(a6 - a_predecrement)), va_shift));
    
    // 同上，加载 a7 - a_predecrement 处的数据并生成 va7 向量
    const float16x4_t va7 = vreinterpret_f16_u64(vshl_u64(
        vreinterpret_u64_f16(vld1_f16(a7 - a_predecrement)), va_shift));

    {
      // 加载指针 w 指向的 float16x8_t 数据到向量 vb01234567
      const float16x8_t vb01234567 = vld1q_f16(w);
      // 将指针 w 向后移动 sizeof(float16x8_t) 字节
      w = (void*)((uintptr_t)w + sizeof(float16x8_t));

      // 使用 va0 的第 0 个元素对 vb01234567 进行乘加运算并更新 vacc0x01234567 向量
      vacc0x01234567 = vmlaq_lane_f16(vacc0x01234567, vb01234567, va0, 0);
      // 使用 va1 的第 0 个元素对 vb01234567 进行乘加运算并更新 vacc1x01234567 向量
      vacc1x01234567 = vmlaq_lane_f16(vacc1x01234567, vb01234567, va1, 0);
      // 使用 va2 的第 0 个元素对 vb01234567 进行乘加运算并更新 vacc2x01234567 向量
      vacc2x01234567 = vmlaq_lane_f16(vacc2x01234567, vb01234567, va2, 0);
      // 使用 va3 的第 0 个元素对 vb01234567 进行乘加运算并更新 vacc3x01234567 向量
      vacc3x01234567 = vmlaq_lane_f16(vacc3x01234567, vb01234567, va3, 0);
      // 使用 va4 的第 0 个元素对 vb01234567 进行乘加运算并更新 vacc4x01234567 向量
      vacc4x01234567 = vmlaq_lane_f16(vacc4x01234567, vb01234567, va4, 0);
      // 使用 va5 的第 0 个元素对 vb01234567 进行乘加运算并更新 vacc5x01234567 向量
      vacc5x01234567 = vmlaq_lane_f16(vacc5x01234567, vb01234567, va5, 0);
      // 使用 va6 的第 0 个元素对 vb01234567 进行乘加运算并更新 vacc6x01234567 向量
      vacc6x01234567 = vmlaq_lane_f16(vacc6x01234567, vb01234567, va6, 0);
      // 使用 va7 的第 0 个元素对 vb01234567 进行乘加运算并更新 vacc7x01234567 向量
      vacc7x01234567 = vmlaq_lane_f16(vacc7x01234567, vb01234567, va7, 0);
    }
    # 如果 k 大于等于 2，则执行以下操作
    if (k >= 2) {
      # 从指针 w 处加载一个 float16x8_t 类型的数据到 vb01234567
      const float16x8_t vb01234567 = vld1q_f16(w);
      # 将指针 w 向后移动 sizeof(float16x8_t) 字节的位置
      w = (void*)((uintptr_t)w + sizeof(float16x8_t));

      # 使用向量化指令 vmlaq_lane_f16 将 va0 的第 1 个元素与 vb01234567 的每个元素相乘并加到 vacc0x01234567 中
      vacc0x01234567 = vmlaq_lane_f16(vacc0x01234567, vb01234567, va0, 1);
      # 同上，对 va1 到 va7 执行相同的操作
      vacc1x01234567 = vmlaq_lane_f16(vacc1x01234567, vb01234567, va1, 1);
      vacc2x01234567 = vmlaq_lane_f16(vacc2x01234567, vb01234567, va2, 1);
      vacc3x01234567 = vmlaq_lane_f16(vacc3x01234567, vb01234567, va3, 1);
      vacc4x01234567 = vmlaq_lane_f16(vacc4x01234567, vb01234567, va4, 1);
      vacc5x01234567 = vmlaq_lane_f16(vacc5x01234567, vb01234567, va5, 1);
      vacc6x01234567 = vmlaq_lane_f16(vacc6x01234567, vb01234567, va6, 1);
      vacc7x01234567 = vmlaq_lane_f16(vacc7x01234567, vb01234567, va7, 1);

      # 如果 k 大于 2，则继续执行以下操作
      if (k > 2) {
        # 从指针 w 处加载一个 float16x8_t 类型的数据到 vb01234567
        const float16x8_t vb01234567 = vld1q_f16(w);
        # 将指针 w 向后移动 sizeof(float16x8_t) 字节的位置
        w = (void*)((uintptr_t)w + sizeof(float16x8_t));

        # 使用向量化指令 vmlaq_lane_f16 将 va0 的第 2 个元素与 vb01234567 的每个元素相乘并加到 vacc0x01234567 中
        vacc0x01234567 = vmlaq_lane_f16(vacc0x01234567, vb01234567, va0, 2);
        # 同上，对 va1 到 va7 执行相同的操作
        vacc1x01234567 = vmlaq_lane_f16(vacc1x01234567, vb01234567, va1, 2);
        vacc2x01234567 = vmlaq_lane_f16(vacc2x01234567, vb01234567, va2, 2);
        vacc3x01234567 = vmlaq_lane_f16(vacc3x01234567, vb01234567, va3, 2);
        vacc4x01234567 = vmlaq_lane_f16(vacc4x01234567, vb01234567, va4, 2);
        vacc5x01234567 = vmlaq_lane_f16(vacc5x01234567, vb01234567, va5, 2);
        vacc6x01234567 = vmlaq_lane_f16(vacc6x01234567, vb01234567, va6, 2);
        vacc7x01234567 = vmlaq_lane_f16(vacc7x01234567, vb01234567, va7, 2);

        # 如果 k 大于等于 4，则继续执行以下操作
        if (k >= 4) {
          # 从指针 w 处加载一个 float16x8_t 类型的数据到 vb01234567
          const float16x8_t vb01234567 = vld1q_f16(w);

          # 使用向量化指令 vmlaq_lane_f16 将 va0 的第 3 个元素与 vb01234567 的每个元素相乘并加到 vacc0x01234567 中
          vacc0x01234567 = vmlaq_lane_f16(vacc0x01234567, vb01234567, va0, 3);
          # 同上，对 va1 到 va7 执行相同的操作
          vacc1x01234567 = vmlaq_lane_f16(vacc1x01234567, vb01234567, va1, 3);
          vacc2x01234567 = vmlaq_lane_f16(vacc2x01234567, vb01234567, va2, 3);
          vacc3x01234567 = vmlaq_lane_f16(vacc3x01234567, vb01234567, va3, 3);
          vacc4x01234567 = vmlaq_lane_f16(vacc4x01234567, vb01234567, va4, 3);
          vacc5x01234567 = vmlaq_lane_f16(vacc5x01234567, vb01234567, va5, 3);
          vacc6x01234567 = vmlaq_lane_f16(vacc6x01234567, vb01234567, va6, 3);
          vacc7x01234567 = vmlaq_lane_f16(vacc7x01234567, vb01234567, va7, 3);
        }
      }
  }
}  // 结束大括号，结束函数

const float16x8_t vscale =
    vld1q_dup_f16((const __fp16*)&clamping_params->scale);
// 加载缩放参数到一个float16x8_t类型的变量vscale

vacc0x01234567 = vmulq_f16(vacc0x01234567, vscale);
vacc1x01234567 = vmulq_f16(vacc1x01234567, vscale);
vacc2x01234567 = vmulq_f16(vacc2x01234567, vscale);
vacc3x01234567 = vmulq_f16(vacc3x01234567, vscale);
vacc4x01234567 = vmulq_f16(vacc4x01234567, vscale);
vacc5x01234567 = vmulq_f16(vacc5x01234567, vscale);
vacc6x01234567 = vmulq_f16(vacc6x01234567, vscale);
vacc7x01234567 = vmulq_f16(vacc7x01234567, vscale);
// 对8个向量进行与vscale的乘法操作

const float16x8_t vmax = vld1q_dup_f16((const __fp16*)&clamping_params->max);
// 加载最大值参数到一个float16x8_t类型的变量vmax

vacc0x01234567 = vminq_f16(vacc0x01234567, vmax);
vacc1x01234567 = vminq_f16(vacc1x01234567, vmax);
vacc2x01234567 = vminq_f16(vacc2x01234567, vmax);
vacc3x01234567 = vminq_f16(vacc3x01234567, vmax);
vacc4x01234567 = vminq_f16(vacc4x01234567, vmax);
vacc5x01234567 = vminq_f16(vacc5x01234567, vmax);
vacc6x01234567 = vminq_f16(vacc6x01234567, vmax);
vacc7x01234567 = vminq_f16(vacc7x01234567, vmax);
// 对8个向量进行与vmax的最小值操作

const float16x8_t vmin = vld1q_dup_f16((const __fp16*)&clamping_params->min);
// 加载最小值参数到一个float16x8_t类型的变量vmin

vacc0x01234567 = vmaxq_f16(vacc0x01234567, vmin);
vacc1x01234567 = vmaxq_f16(vacc1x01234567, vmin);
vacc2x01234567 = vmaxq_f16(vacc2x01234567, vmin);
vacc3x01234567 = vmaxq_f16(vacc3x01234567, vmin);
vacc4x01234567 = vmaxq_f16(vacc4x01234567, vmin);
vacc5x01234567 = vmaxq_f16(vacc5x01234567, vmin);
vacc6x01234567 = vmaxq_f16(vacc6x01234567, vmin);
vacc7x01234567 = vmaxq_f16(vacc7x01234567, vmin);
// 对8个向量进行与vmin的最大值操作

__fp16* c0 = c;
// 设置指针c0指向c的地址
__fp16* c1 = (__fp16*)((uintptr_t)c0 + c_stride);
// 设置指针c1指向c0的地址偏移c_stride
if (mr < 2) {
  c1 = c0;
}
// 如果mr小于2，则将c1指向c0
__fp16* c2 = (__fp16*)((uintptr_t)c1 + c_stride);
// 设置指针c2指向c1的地址偏移c_stride
if (mr <= 2) {
  c2 = c1;
}
// 如果mr小于等于2，则将c2指向c1
__fp16* c3 = (__fp16*)((uintptr_t)c2 + c_stride);
// 设置指针c3指向c2的地址偏移c_stride
if (mr < 4) {
  c3 = c2;
}
// 如果mr小于4，则将c3指向c2
__fp16* c4 = (__fp16*)((uintptr_t)c3 + c_stride);
// 设置指针c4指向c3的地址偏移c_stride
if (mr <= 4) {
  c4 = c3;
}
// 如果mr小于等于4，则将c4指向c3
__fp16* c5 = (__fp16*)((uintptr_t)c4 + c_stride);
// 设置指针c5指向c4的地址偏移c_stride
if (mr < 6) {
  c5 = c4;
}
// 如果mr小于6，则将c5指向c4
__fp16* c6 = (__fp16*)((uintptr_t)c5 + c_stride);
// 设置指针c6指向c5的地址偏移c_stride
if (mr <= 6) {
  c6 = c5;
}
// 如果mr小于等于6，则将c6指向c5
__fp16* c7 = (__fp16*)((uintptr_t)c6 + c_stride);
// 设置指针c7指向c6的地址偏移c_stride
if (mr != 8) {
  c7 = c6;
}
// 如果mr不等于8，则将c7指向c6

if (nr == 8) {
  vst1q_f16(c0, vacc0x01234567);
  vst1q_f16(c1, vacc1x01234567);
  vst1q_f16(c2, vacc2x01234567);
  vst1q_f16(c3, vacc3x01234567);
  vst1q_f16(c4, vacc4x01234567);
  vst1q_f16(c5, vacc5x01234567);
  vst1q_f16(c6, vacc6x01234567);
  vst1q_f16(c7, vacc7x01234567);
} else {
    // 检查 nr 的第二位是否为1，表示需要处理4个float16向量
    if (nr & 4) {
      // 将第一个累加器的前半部分写入到 c0 指向的内存中
      vst1_f16(c0, vget_low_f16(vacc0x01234567));
      c0 += 4;
      // 将第二个累加器的前半部分写入到 c1 指向的内存中
      vst1_f16(c1, vget_low_f16(vacc1x01234567));
      c1 += 4;
      // 将第三个累加器的前半部分写入到 c2 指向的内存中
      vst1_f16(c2, vget_low_f16(vacc2x01234567));
      c2 += 4;
      // 将第四个累加器的前半部分写入到 c3 指向的内存中
      vst1_f16(c3, vget_low_f16(vacc3x01234567));
      c3 += 4;
      // 将第五个累加器的前半部分写入到 c4 指向的内存中
      vst1_f16(c4, vget_low_f16(vacc4x01234567));
      c4 += 4;
      // 将第六个累加器的前半部分写入到 c5 指向的内存中
      vst1_f16(c5, vget_low_f16(vacc5x01234567));
      c5 += 4;
      // 将第七个累加器的前半部分写入到 c6 指向的内存中
      vst1_f16(c6, vget_low_f16(vacc6x01234567));
      c6 += 4;
      // 将第八个累加器的前半部分写入到 c7 指向的内存中
      vst1_f16(c7, vget_low_f16(vacc7x01234567));
      c7 += 4;
      // 将第一个累加器的后半部分移到前半部分，实现向左移动4个元素
      vacc0x01234567 = vextq_f16(vacc0x01234567, vacc0x01234567, 4);
      // 将第二个累加器的后半部分移到前半部分，实现向左移动4个元素
      vacc1x01234567 = vextq_f16(vacc1x01234567, vacc1x01234567, 4);
      // 将第三个累加器的后半部分移到前半部分，实现向左移动4个元素
      vacc2x01234567 = vextq_f16(vacc2x01234567, vacc2x01234567, 4);
      // 将第四个累加器的后半部分移到前半部分，实现向左移动4个元素
      vacc3x01234567 = vextq_f16(vacc3x01234567, vacc3x01234567, 4);
      // 将第五个累加器的后半部分移到前半部分，实现向左移动4个元素
      vacc4x01234567 = vextq_f16(vacc4x01234567, vacc4x01234567, 4);
      // 将第六个累加器的后半部分移到前半部分，实现向左移动4个元素
      vacc5x01234567 = vextq_f16(vacc5x01234567, vacc5x01234567, 4);
      // 将第七个累加器的后半部分移到前半部分，实现向左移动4个元素
      vacc6x01234567 = vextq_f16(vacc6x01234567, vacc6x01234567, 4);
      // 将第八个累加器的后半部分移到前半部分，实现向左移动4个元素
      vacc7x01234567 = vextq_f16(vacc7x01234567, vacc7x01234567, 4);
    }
    // 检查 nr 的第一位是否为1，表示需要处理2个float16向量
    if (nr & 2) {
      // 将第一个累加器的前半部分的低位元素写入到 c0 指向的内存中
      vst1_lane_u32(
          __builtin_assume_aligned(c0, 1),
          vreinterpret_u32_f16(vget_low_f16(vacc0x01234567)),
          0);
      c0 += 2;
      // 将第二个累加器的前半部分的低位元素写入到 c1 指向的内存中
      vst1_lane_u32(
          __builtin_assume_aligned(c1, 1),
          vreinterpret_u32_f16(vget_low_f16(vacc1x01234567)),
          0);
      c1 += 2;
      // 将第三个累加器的前半部分的低位元素写入到 c2 指向的内存中
      vst1_lane_u32(
          __builtin_assume_aligned(c2, 1),
          vreinterpret_u32_f16(vget_low_f16(vacc2x01234567)),
          0);
      c2 += 2;
      // 将第四个累加器的前半部分的低位元素写入到 c3 指向的内存中
      vst1_lane_u32(
          __builtin_assume_aligned(c3, 1),
          vreinterpret_u32_f16(vget_low_f16(vacc3x01234567)),
          0);
      c3 += 2;
      // 将第五个累加器的前半部分的低位元素写入到 c4 指向的内存中
      vst1_lane_u32(
          __builtin_assume_aligned(c4, 1),
          vreinterpret_u32_f16(vget_low_f16(vacc4x01234567)),
          0);
      c4 += 2;
      // 将第六个累加器的前半部分的低位元素写入到 c5 指向的内存中
      vst1_lane_u32(
          __builtin_assume_aligned(c5, 1),
          vreinterpret_u32_f16(vget_low_f16(vacc5x01234567)),
          0);
      c5 += 2;
      // 将第七个累加器的前半部分的低位元素写入到 c6 指向的内存中
      vst1_lane_u32(
          __builtin_assume_aligned(c6, 1),
          vreinterpret_u32_f16(vget_low_f16(vacc6x01234567)),
          0);
      c6 += 2;
      // 将第八个累加器的前半部分的低位元素写入到 c7 指向的内存中
      vst1_lane_u32(
          __builtin_assume_aligned(c7, 1),
          vreinterpret_u32_f16(vget_low_f16(vacc7x01234567)),
          0);
      c7 += 2;
      // 将第一个累加器的后半部分移到前半部分，实现向左移动2个元素
      vacc0x01234567 = vextq_f16(vacc0x01234567, vacc0x01234567, 2);
      // 将第二个累加器的后半部分移到前半部分，实现向左移动2个元素
      vacc1x01234567 = vextq_f16(vacc1x01234567, vacc1x01234567, 2);
      // 将第三个累加器的后半部分移到前半部分，实现向左移动2个元素
      vacc2x01234567 = vextq_f16(vacc2x01234567, vacc2x01234567, 2);
      // 将第四个累加器的后半部分移到前半部分，实现向左移动2个元素
      vacc3x01234567 = vextq_f16(vacc3x01234567, vacc3x012
    # 检查 nr 是否为奇数，如果是则执行以下操作
    if (nr & 1) {
      # 将 vacc0x01234567 中的第一个 float16 值写入 c0 中
      vst1q_lane_f16(c0, vacc0x01234567, 0);
      # 将 vacc1x01234567 中的第一个 float16 值写入 c1 中
      vst1q_lane_f16(c1, vacc1x01234567, 0);
      # 将 vacc2x01234567 中的第一个 float16 值写入 c2 中
      vst1q_lane_f16(c2, vacc2x01234567, 0);
      # 将 vacc3x01234567 中的第一个 float16 值写入 c3 中
      vst1q_lane_f16(c3, vacc3x01234567, 0);
      # 将 vacc4x01234567 中的第一个 float16 值写入 c4 中
      vst1q_lane_f16(c4, vacc4x01234567, 0);
      # 将 vacc5x01234567 中的第一个 float16 值写入 c5 中
      vst1q_lane_f16(c5, vacc5x01234567, 0);
      # 将 vacc6x01234567 中的第一个 float16 值写入 c6 中
      vst1q_lane_f16(c6, vacc6x01234567, 0);
      # 将 vacc7x01234567 中的第一个 float16 值写入 c7 中
      vst1q_lane_f16(c7, vacc7x01234567, 0);
    }
  }
}



# 这行代码仅仅是一个闭合的大括号，通常用于结束代码块或函数定义。
```