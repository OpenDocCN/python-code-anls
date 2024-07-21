# `.\pytorch\aten\src\ATen\native\quantized\cpu\qnnpack\src\q8gemm\4x8-neon.c`

```
/*
 * 从 Facebook, Inc. 及其附属公司获得版权。
 * 保留所有权利。
 *
 * 此源代码在位于根目录下的 LICENSE 文件中找到的 BSD 风格许可下许可。
 */

#include <arm_neon.h> // 引入 ARM NEON 指令集的头文件

#include <qnnpack/q8gemm.h> // 引入 QNNPACK 库中 Q8 GEMM 的头文件
#include <requantization/runtime-neon.h> // 引入 NEON 运行时支持的头文件

void pytorch_q8gemm_ukernel_4x8__neon(
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
        quantization_params[restrict static 1]) {
  int32x4_t vacc0x0123 = vld1q_s32(w); // 从地址 w 加载 4 个 int32 值到寄存器 vacc0x0123
  w = (const void*)((uintptr_t)w + 16); // 更新地址 w，偏移 16 字节（一个 NEON 寄存器大小）

  int32x4_t vacc0x4567 = vld1q_s32(w); // 从更新后的地址 w 加载下一个 4 个 int32 值到寄存器 vacc0x4567
  w = (const void*)((uintptr_t)w + 16); // 再次更新地址 w，偏移 16 字节

  // 复制加载的寄存器值到其他相关寄存器，以备后续计算使用
  int32x4_t vacc1x0123 = vacc0x0123;
  int32x4_t vacc1x4567 = vacc0x4567;
  int32x4_t vacc2x0123 = vacc0x0123;
  int32x4_t vacc2x4567 = vacc0x4567;
  int32x4_t vacc3x0123 = vacc0x0123;
  int32x4_t vacc3x4567 = vacc0x4567;

  const uint8_t* a0 = a; // 设置指针 a0 指向输入矩阵 a 的起始位置
  const uint8_t* a1 = (const uint8_t*)((uintptr_t)a0 + a_stride); // 设置指针 a1，偏移一个行距
  if (mr < 2) {
    a1 = a0; // 如果 mr 小于 2，将 a1 重置为 a0，避免超出矩阵范围
  }
  const uint8_t* a2 = (const uint8_t*)((uintptr_t)a1 + a_stride); // 设置指针 a2，偏移一个行距
  if (mr <= 2) {
    a2 = a1; // 如果 mr 小于等于 2，将 a2 重置为 a1，避免超出矩阵范围
  }
  const uint8_t* a3 = (const uint8_t*)((uintptr_t)a2 + a_stride); // 设置指针 a3，偏移一个行距
  if (mr != 4) {
    a3 = a2; // 如果 mr 不等于 4，将 a3 重置为 a2，避免超出矩阵范围
  }

  // 加载输入量化参数中的输入零点值到向量寄存器 va_zero_point
  const uint8x8_t va_zero_point =
      vld1_dup_u8((const uint8_t*)&quantization_params->neon.input_zero_point);

  // 加载输入量化参数中的卷积核零点值到向量寄存器 vb_zero_point
  const uint8x8_t vb_zero_point =
      vld1_u8((const uint8_t*)&quantization_params->neon.kernel_zero_points
          [output_channel_index]);

  // 对于每次循环处理 8 个元素，直到 k 小于 8
  for (; k >= 8; k -= 8) {
    const uint8x8_t va0 = vld1_u8(a0); // 加载地址 a0 处的 8 个 uint8 值到向量寄存器 va0
    a0 += 8; // 更新地址 a0，移动到下一个 8 个元素

    // 将 va0 与 va_zero_point 相减，并将结果转换为 int16 型向量寄存器 vxa0
    const int16x8_t vxa0 =
        vreinterpretq_s16_u16(sub_zero_point(va0, va_zero_point));

    // 依次类推，加载 a1、a2、a3 的数据到向量寄存器，并进行零点对齐和类型转换
    const uint8x8_t va1 = vld1_u8(a1);
    a1 += 8;
    const int16x8_t vxa1 =
        vreinterpretq_s16_u16(sub_zero_point(va1, va_zero_point));

    const uint8x8_t va2 = vld1_u8(a2);
    a2 += 8;
    const int16x8_t vxa2 =
        vreinterpretq_s16_u16(sub_zero_point(va2, va_zero_point));

    const uint8x8_t va3 = vld1_u8(a3);
    a3 += 8;
    const int16x8_t vxa3 =
        vreinterpretq_s16_u16(sub_zero_point(va3, va_zero_point));

    // 加载权重数据到向量寄存器 vb01234567c0，并进行零点对齐和类型转换
    const uint8x8_t vb01234567c0 = vld1_u8(w);
    w = (const void*)((uintptr_t)w + 8);
    const int16x8_t vxb01234567c0 =
        vreinterpretq_s16_u16(vsubl_u8(vb01234567c0, vb_zero_point));

    // 使用 SIMD 指令计算乘加累加到累加寄存器中
    vacc0x0123 = vmlal_lane_s16(
        vacc0x0123, vget_low_s16(vxb01234567c0), vget_low_s16(vxa0), 0);
    vacc0x4567 = vmlal_lane_s16(
        vacc0x4567, vget_high_s16(vxb01234567c0), vget_low_s16(vxa0), 0);
    vacc1x0123 = vmlal_lane_s16(
        vacc1x0123, vget_low_s16(vxb01234567c0), vget_low_s16(vxa1), 0);
    vacc1x4567 = vmlal_lane_s16(
        vacc1x4567, vget_high_s16(vxb01234567c0), vget_low_s16(vxa1), 0);
    // 在累加寄存器vacc2x0123中，使用vxb01234567c0的低位数据和vxa2的低位数据进行饱和加法累加
    vacc2x0123 = vmlal_lane_s16(
        vacc2x0123, vget_low_s16(vxb01234567c0), vget_low_s16(vxa2), 0);
    // 在累加寄存器vacc2x4567中，使用vxb01234567c0的高位数据和vxa2的低位数据进行饱和加法累加
    vacc2x4567 = vmlal_lane_s16(
        vacc2x4567, vget_high_s16(vxb01234567c0), vget_low_s16(vxa2), 0);
    // 在累加寄存器vacc3x0123中，使用vxb01234567c0的低位数据和vxa3的低位数据进行饱和加法累加
    vacc3x0123 = vmlal_lane_s16(
        vacc3x0123, vget_low_s16(vxb01234567c0), vget_low_s16(vxa3), 0);
    // 在累加寄存器vacc3x4567中，使用vxb01234567c0的高位数据和vxa3的低位数据进行饱和加法累加
    vacc3x4567 = vmlal_lane_s16(
        vacc3x4567, vget_high_s16(vxb01234567c0), vget_low_s16(vxa3), 0);

    // 加载向量w指向的8字节数据到向量寄存器vb01234567c1
    const uint8x8_t vb01234567c1 = vld1_u8(w);
    // 更新指针w，使其指向下一个8字节数据
    w = (const void*)((uintptr_t)w + 8);
    // 将vb01234567c1减去vb_zero_point后转换成有符号16位整数，并存入vxb01234567c1
    const int16x8_t vxb01234567c1 =
        vreinterpretq_s16_u16(vsubl_u8(vb01234567c1, vb_zero_point));

    // 在累加寄存器vacc0x0123中，使用vxb01234567c1的低位数据和vxa0的低位数据进行饱和加法累加
    vacc0x0123 = vmlal_lane_s16(
        vacc0x0123, vget_low_s16(vxb01234567c1), vget_low_s16(vxa0), 1);
    // 在累加寄存器vacc0x4567中，使用vxb01234567c1的高位数据和vxa0的低位数据进行饱和加法累加
    vacc0x4567 = vmlal_lane_s16(
        vacc0x4567, vget_high_s16(vxb01234567c1), vget_low_s16(vxa0), 1);
    // 在累加寄存器vacc1x0123中，使用vxb01234567c1的低位数据和vxa1的低位数据进行饱和加法累加
    vacc1x0123 = vmlal_lane_s16(
        vacc1x0123, vget_low_s16(vxb01234567c1), vget_low_s16(vxa1), 1);
    // 在累加寄存器vacc1x4567中，使用vxb01234567c1的高位数据和vxa1的低位数据进行饱和加法累加
    vacc1x4567 = vmlal_lane_s16(
        vacc1x4567, vget_high_s16(vxb01234567c1), vget_low_s16(vxa1), 1);
    // 在累加寄存器vacc2x0123中，使用vxb01234567c1的低位数据和vxa2的低位数据进行饱和加法累加
    vacc2x0123 = vmlal_lane_s16(
        vacc2x0123, vget_low_s16(vxb01234567c1), vget_low_s16(vxa2), 1);
    // 在累加寄存器vacc2x4567中，使用vxb01234567c1的高位数据和vxa2的低位数据进行饱和加法累加
    vacc2x4567 = vmlal_lane_s16(
        vacc2x4567, vget_high_s16(vxb01234567c1), vget_low_s16(vxa2), 1);
    // 在累加寄存器vacc3x0123中，使用vxb01234567c1的低位数据和vxa3的低位数据进行饱和加法累加
    vacc3x0123 = vmlal_lane_s16(
        vacc3x0123, vget_low_s16(vxb01234567c1), vget_low_s16(vxa3), 1);
    // 在累加寄存器vacc3x4567中，使用vxb01234567c1的高位数据和vxa3的低位数据进行饱和加法累加
    vacc3x4567 = vmlal_lane_s16(
        vacc3x4567, vget_high_s16(vxb01234567c1), vget_low_s16(vxa3), 1);

    // 加载向量w指向的8字节数据到向量寄存器vb01234567c2
    const uint8x8_t vb01234567c2 = vld1_u8(w);
    // 更新指针w，使其指向下一个8字节数据
    w = (const void*)((uintptr_t)w + 8);
    // 将vb01234567c2减去vb_zero_point后转换成有符号16位整数，并存入vxb01234567c2
    const int16x8_t vxb01234567c2 =
        vreinterpretq_s16_u16(vsubl_u8(vb01234567c2, vb_zero_point));

    // 在累加寄存器vacc0x0123中，使用vxb01234567c2的低位数据和vxa0的低位数据进行饱和加法累加
    vacc0x0123 = vmlal_lane_s16(
        vacc0x0123, vget_low_s16(vxb01234567c2), vget_low_s16(vxa0), 2);
    // 在累加寄存器vacc0x4567中，使用vxb01234567c2的高位数据和vxa0的低位数据进行饱和加法累加
    vacc0x4567 = vmlal_lane_s16(
        vacc0x4567, vget_high_s16(vxb01234567c2), vget_low_s16(vxa0), 2);
    // 在累加寄存器vacc1x0123中，使用vxb01234567c2的低位数据和vxa1的低位数据进行饱和加法累加
    vacc1x0123 = vmlal_lane_s16(
        vacc1x0123, vget_low_s16(vxb01234567c2), vget_low_s16(vxa1), 2);
    // 在累加寄存器vacc1x4567中，使用vxb01234567c2的高位数据和vxa1的低位数据进行饱和加法累加
    vacc1x4567 = vmlal_lane_s16(
        vacc1x4567, vget_high_s16(vxb01234567c2), vget_low_s16(vxa1), 2);
    // 在累加寄存器vacc2x0123中，使用vxb01234567c2的低位数据和vxa2的低位数据进行饱和加法累加
    vacc2x0123 = vmlal_lane_s16(
        vacc2x0123, vget_low_s16(vxb01234567c2), vget_low_s16(vxa2), 2);
    // 在累加寄存器vacc2x4567中，使用vxb01234567c2的高位数据和vxa2的低位数据进行饱和加法累加
    vacc2x4567 = vmlal_lane_s16(
        vacc2x4567, vget_high_s16(vxb01234567
    // 使用 vmlal_lane_s16 函数将 vxa1 的低16位元素与 vxb01234567c3 的低16位元素乘积加到 vacc1x0123
    vacc1x0123 = vmlal_lane_s16(
        vacc1x0123, vget_low_s16(vxb01234567c3), vget_low_s16(vxa1), 3);
    // 使用 vmlal_lane_s16 函数将 vxa1 的高16位元素与 vxb01234567c3 的高16位元素乘积加到 vacc1x4567
    vacc1x4567 = vmlal_lane_s16(
        vacc1x4567, vget_high_s16(vxb01234567c3), vget_low_s16(vxa1), 3);
    // 使用 vmlal_lane_s16 函数将 vxa2 的低16位元素与 vxb01234567c3 的低16位元素乘积加到 vacc2x0123
    vacc2x0123 = vmlal_lane_s16(
        vacc2x0123, vget_low_s16(vxb01234567c3), vget_low_s16(vxa2), 3);
    // 使用 vmlal_lane_s16 函数将 vxa2 的高16位元素与 vxb01234567c3 的高16位元素乘积加到 vacc2x4567
    vacc2x4567 = vmlal_lane_s16(
        vacc2x4567, vget_high_s16(vxb01234567c3), vget_low_s16(vxa2), 3);
    // 使用 vmlal_lane_s16 函数将 vxa3 的低16位元素与 vxb01234567c3 的低16位元素乘积加到 vacc3x0123
    vacc3x0123 = vmlal_lane_s16(
        vacc3x0123, vget_low_s16(vxb01234567c3), vget_low_s16(vxa3), 3);
    // 使用 vmlal_lane_s16 函数将 vxa3 的高16位元素与 vxb01234567c3 的高16位元素乘积加到 vacc3x4567
    vacc3x4567 = vmlal_lane_s16(
        vacc3x4567, vget_high_s16(vxb01234567c3), vget_low_s16(vxa3), 3);

    // 加载并解释 vb01234567c4 的数据到 vb01234567c4
    const uint8x8_t vb01234567c4 = vld1_u8(w);
    // 将指针 w 向前移动 8 字节
    w = (const void*)((uintptr_t)w + 8);
    // 将 vb01234567c4 减去 vb_zero_point，并转换为 int16x8_t 类型存入 vxb01234567c4
    const int16x8_t vxb01234567c4 =
        vreinterpretq_s16_u16(vsubl_u8(vb01234567c4, vb_zero_point));

    // 使用 vmlal_lane_s16 函数将 vxa0 的低16位元素与 vxb01234567c4 的低16位元素乘积加到 vacc0x0123
    vacc0x0123 = vmlal_lane_s16(
        vacc0x0123, vget_low_s16(vxb01234567c4), vget_high_s16(vxa0), 0);
    // 使用 vmlal_lane_s16 函数将 vxa0 的高16位元素与 vxb01234567c4 的高16位元素乘积加到 vacc0x4567
    vacc0x4567 = vmlal_lane_s16(
        vacc0x4567, vget_high_s16(vxb01234567c4), vget_high_s16(vxa0), 0);
    // 使用 vmlal_lane_s16 函数将 vxa1 的低16位元素与 vxb01234567c4 的低16位元素乘积加到 vacc1x0123
    vacc1x0123 = vmlal_lane_s16(
        vacc1x0123, vget_low_s16(vxb01234567c4), vget_high_s16(vxa1), 0);
    // 使用 vmlal_lane_s16 函数将 vxa1 的高16位元素与 vxb01234567c4 的高16位元素乘积加到 vacc1x4567
    vacc1x4567 = vmlal_lane_s16(
        vacc1x4567, vget_high_s16(vxb01234567c4), vget_high_s16(vxa1), 0);
    // 使用 vmlal_lane_s16 函数将 vxa2 的低16位元素与 vxb01234567c4 的低16位元素乘积加到 vacc2x0123
    vacc2x0123 = vmlal_lane_s16(
        vacc2x0123, vget_low_s16(vxb01234567c4), vget_high_s16(vxa2), 0);
    // 使用 vmlal_lane_s16 函数将 vxa2 的高16位元素与 vxb01234567c4 的高16位元素乘积加到 vacc2x4567
    vacc2x4567 = vmlal_lane_s16(
        vacc2x4567, vget_high_s16(vxb01234567c4), vget_high_s16(vxa2), 0);
    // 使用 vmlal_lane_s16 函数将 vxa3 的低16位元素与 vxb01234567c4 的低16位元素乘积加到 vacc3x0123
    vacc3x0123 = vmlal_lane_s16(
        vacc3x0123, vget_low_s16(vxb01234567c4), vget_high_s16(vxa3), 0);
    // 使用 vmlal_lane_s16 函数将 vxa3 的高16位元素与 vxb01234567c4 的高16位元素乘积加到 vacc3x4567
    vacc3x4567 = vmlal_lane_s16(
        vacc3x4567, vget_high_s16(vxb01234567c4), vget_high_s16(vxa3), 0);

    // 加载并解释 vb01234567c5 的数据到 vb01234567c5
    const uint8x8_t vb01234567c5 = vld1_u8(w);
    // 将指针 w 向前移动 8 字节
    w = (const void*)((uintptr_t)w + 8);
    // 将 vb01234567c5 减去 vb_zero_point，并转换为 int16x8_t 类型存入 vxb01234567c5
    const int16x8_t vxb01234567c5 =
        vreinterpretq_s16_u16(vsubl_u8(vb01234567c5, vb_zero_point));

    // 使用 vmlal_lane_s16 函数将 vxa0 的低16位元素与 vxb01234567c5 的低16位元素乘积加到 vacc0x0123
    vacc0x0123 = vmlal_lane_s16(
        vacc0x0123, vget_low_s16(vxb01234567c5), vget_high_s16(vxa0), 1);
    // 使用 vmlal_lane_s16 函数将 vxa0 的高16位元素与 vxb01234567c5 的高16位元素乘积加到 vacc0x4567
    vacc0x4567 = vmlal_lane_s16(
        vacc0x4567, vget_high_s16(vxb01234567c5), vget_high_s16(vxa0), 1);
    // 使用 vmlal_lane_s16 函数将 vxa1 的低16位元素与 vxb01234567c5 的低16位元素乘积加到 vacc1x0123
    vacc1x0123 = vmlal_lane_s16(
        vacc1x0123, vget_low_s16(vxb01234567c5), vget_high_s16(vxa1), 1);
    // 使用 vmlal_lane_s16 函数将 vxa1 的高16位元素与 vxb01234567c5 的高16位元素乘积加到 vacc1x4567
    vacc1x4567 = vmlal_lane_s16(
        vacc1x4567, vget_high_s16(vxb01234567c5), vget_high_s16(vxa1), 1);
    // 使用 vmlal_lane_s16 函数将 vxa2 的低16位元素与 vxb01234567c5 的低16位元素乘积加到 vacc
    // 使用 vget_low_s16 函数从 vxb01234567c6 中提取低位的 int16x4_t 向量，并将其与 vacc0x0123 进行 2 位的 S16 加法累加操作
    vacc0x0123 = vmlal_lane_s16(
        vacc0x0123, vget_low_s16(vxb01234567c6), vget_high_s16(vxa0), 2);
    // 使用 vget_high_s16 函数从 vxb01234567c6 中提取高位的 int16x4_t 向量，并将其与 vacc0x4567 进行 2 位的 S16 加法累加操作
    vacc0x4567 = vmlal_lane_s16(
        vacc0x4567, vget_high_s16(vxb01234567c6), vget_high_s16(vxa0), 2);
    // 类似上述操作，但是操作的是 vacc1x0123 和 vxa1 中的向量
    vacc1x0123 = vmlal_lane_s16(
        vacc1x0123, vget_low_s16(vxb01234567c6), vget_high_s16(vxa1), 2);
    vacc1x4567 = vmlal_lane_s16(
        vacc1x4567, vget_high_s16(vxb01234567c6), vget_high_s16(vxa1), 2);
    // 类似上述操作，但是操作的是 vacc2x0123 和 vxa2 中的向量
    vacc2x0123 = vmlal_lane_s16(
        vacc2x0123, vget_low_s16(vxb01234567c6), vget_high_s16(vxa2), 2);
    vacc2x4567 = vmlal_lane_s16(
        vacc2x4567, vget_high_s16(vxb01234567c6), vget_high_s16(vxa2), 2);
    // 类似上述操作，但是操作的是 vacc3x0123 和 vxa3 中的向量
    vacc3x0123 = vmlal_lane_s16(
        vacc3x0123, vget_low_s16(vxb01234567c6), vget_high_s16(vxa3), 2);
    vacc3x4567 = vmlal_lane_s16(
        vacc3x4567, vget_high_s16(vxb01234567c6), vget_high_s16(vxa3), 2);

    // 从指针 w 处加载 8 个 uint8_t 数据，并使 w 指针向后移动 8 个字节
    const uint8x8_t vb01234567c7 = vld1_u8(w);
    w = (const void*)((uintptr_t)w + 8);
    // 将加载的 vb01234567c7 向量中的 uint8_t 数据转换为 int16x8_t 类型的向量 vxb01234567c7，并减去 vb_zero_point 中对应的值
    const int16x8_t vxb01234567c7 =
        vreinterpretq_s16_u16(vsubl_u8(vb01234567c7, vb_zero_point));

    // 类似前面的操作，但是使用 vget_high_s16(vxa0) 的第 3 个元素进行 S16 加法累加
    vacc0x0123 = vmlal_lane_s16(
        vacc0x0123, vget_low_s16(vxb01234567c7), vget_high_s16(vxa0), 3);
    vacc0x4567 = vmlal_lane_s16(
        vacc0x4567, vget_high_s16(vxb01234567c7), vget_high_s16(vxa0), 3);
    // 类似上述操作，但是操作的是 vacc1x0123 和 vxa1 中的向量
    vacc1x0123 = vmlal_lane_s16(
        vacc1x0123, vget_low_s16(vxb01234567c7), vget_high_s16(vxa1), 3);
    vacc1x4567 = vmlal_lane_s16(
        vacc1x4567, vget_high_s16(vxb01234567c7), vget_high_s16(vxa1), 3);
    // 类似上述操作，但是操作的是 vacc2x0123 和 vxa2 中的向量
    vacc2x0123 = vmlal_lane_s16(
        vacc2x0123, vget_low_s16(vxb01234567c7), vget_high_s16(vxa2), 3);
    vacc2x4567 = vmlal_lane_s16(
        vacc2x4567, vget_high_s16(vxb01234567c7), vget_high_s16(vxa2), 3);
    // 类似上述操作，但是操作的是 vacc3x0123 和 vxa3 中的向量
    vacc3x0123 = vmlal_lane_s16(
        vacc3x0123, vget_low_s16(vxb01234567c7), vget_high_s16(vxa3), 3);
    vacc3x4567 = vmlal_lane_s16(
        vacc3x4567, vget_high_s16(vxb01234567c7), vget_high_s16(vxa3), 3);
  }
  // 如果 k 不等于 0，则执行以下操作
  if (k != 0) {
    // 计算 a_predecrement，表示需要从 a0, a1, a2, a3 中减去的偏移量
    const size_t a_predecrement = 8 - k;
    // 使用 va_shift 将 -8 * a_predecrement 转换为 int64x1_t 类型
    const int64x1_t va_shift = vmov_n_s64(-8 * a_predecrement);
    // 从 a0 中加载 8 个 uint8_t 数据，并左移 va_shift 指定的位数，然后将结果转换为 int16x8_t 类型的向量 vxa0
    const uint8x8_t va0 = vreinterpret_u8_u64(
        vshl_u64(vreinterpret_u64_u8(vld1_u8(a0 - a_predecrement)), va_shift));
    const int16x8_t vxa0 =
        vreinterpretq_s16_u16(sub_zero_point(va0, va_zero_point));
    // 类似上述操作，但是操作的是 a1 中的数据
    const uint8x8_t va1 = vreinterpret_u8_u64(
        vshl_u64(vreinterpret_u64_u8(vld1_u8(a1 - a_predecrement)), va_shift));
    const int16x8_t vxa1 =
        vreinterpretq_s16_u16(sub_zero_point(va1, va_zero_point));
    // 类似上述操作，但是操作的是 a2 中的数据
    const uint8x8_t va2 = vreinterpret_u8_u64(
        vshl_u64(vreinterpret_u64_u8(vld1_u8(a2 - a_predecrement)), va_shift));
    const int16x8_t vxa2 =
        vreinterpretq_s16_u16(sub_zero_point(va2, va_zero_point));
    // 类似上述操作，但是操作的是 a3 中的数据
    const uint8x8_t va3 = vreinterpret_u8_u64(
        vshl_u64(vreinterpret_u64_u8(vld1_u8(a3 - a_predecrement)), va_shift));
    const int16x8_t vxa3 =
        vreinterpretq_s16_u16(sub_zero_point(va3, va_zero_point));

    // 从指针 w 处加载 8 个 uint8_t 数据到 vb01234567c0 向量中
    const uint8x8_t vb01234567c0 = vld1_u8(w);
    // 将指针 w 增加 8 个字节的偏移量
    w = (const void*)((uintptr_t)w + 8);
    
    // 计算 vb01234567c0 与 vb_zero_point 的差，并转换为 int16x8_t 类型
    const int16x8_t vxb01234567c0 =
        vreinterpretq_s16_u16(vsubl_u8(vb01234567c0, vb_zero_point));
    
    // 使用 vxa0 的低位元素和 vxb01234567c0 的低位元素执行 S16 向量乘法累加
    vacc0x0123 = vmlal_lane_s16(
        vacc0x0123, vget_low_s16(vxb01234567c0), vget_low_s16(vxa0), 0);
    // 使用 vxa0 的低位元素和 vxb01234567c0 的高位元素执行 S16 向量乘法累加
    vacc0x4567 = vmlal_lane_s16(
        vacc0x4567, vget_high_s16(vxb01234567c0), vget_low_s16(vxa0), 0);
    
    // 使用 vxa1 的低位元素和 vxb01234567c0 的低位元素执行 S16 向量乘法累加
    vacc1x0123 = vmlal_lane_s16(
        vacc1x0123, vget_low_s16(vxb01234567c0), vget_low_s16(vxa1), 0);
    // 使用 vxa1 的低位元素和 vxb01234567c0 的高位元素执行 S16 向量乘法累加
    vacc1x4567 = vmlal_lane_s16(
        vacc1x4567, vget_high_s16(vxb01234567c0), vget_low_s16(vxa1), 0);
    
    // 使用 vxa2 的低位元素和 vxb01234567c0 的低位元素执行 S16 向量乘法累加
    vacc2x0123 = vmlal_lane_s16(
        vacc2x0123, vget_low_s16(vxb01234567c0), vget_low_s16(vxa2), 0);
    // 使用 vxa2 的低位元素和 vxb01234567c0 的高位元素执行 S16 向量乘法累加
    vacc2x4567 = vmlal_lane_s16(
        vacc2x4567, vget_high_s16(vxb01234567c0), vget_low_s16(vxa2), 0);
    
    // 使用 vxa3 的低位元素和 vxb01234567c0 的低位元素执行 S16 向量乘法累加
    vacc3x0123 = vmlal_lane_s16(
        vacc3x0123, vget_low_s16(vxb01234567c0), vget_low_s16(vxa3), 0);
    // 使用 vxa3 的低位元素和 vxb01234567c0 的高位元素执行 S16 向量乘法累加
    vacc3x4567 = vmlal_lane_s16(
        vacc3x4567, vget_high_s16(vxb01234567c0), vget_low_s16(vxa3), 0);

    }
  }

  // 使用 VLD1 而不是 VLD2，因为 A75 处理器的延迟更高，VLD1 的吞吐量比 VLD2 更好
  // 每个周期的吞吐量为 2 个。因此，可能 VLD1 更为合适。
  // 从 quantization_params->neon.requantization_scales 中加载 FP32 值到向量 requantization_scale_c0123
  const float32x4_t requantization_scale_c0123 =
      vld1q_f32(
          &quantization_params->neon.requantization_scales[output_channel_index]
          );
  // 从 quantization_params->neon.requantization_scales 中加载 FP32 值到向量 requantization_scale_c4567
  const float32x4_t requantization_scale_c4567 =
      vld1q_f32(
          &quantization_params->neon.requantization_scales[
              output_channel_index + 4]);

  /*
   * 将 int32_t 类型的输入转换为 FP32，并乘以 FP32 的缩放因子。
   * 这两个操作都涉及统计上无偏的舍入：
   * - 大的 int32_t 值不能完全表示为 FP32。ARM NEON 中的转换指令会将其舍入为最接近的 FP32 值（偶数舍入）。
   * - 两个 FP32 值的乘积通常不能完全表示为一个 FP32 值，并将被舍入为最接近的 FP32 值（偶数舍入）。
   */
  // 将 vacc0x0123 转换为 FP32，然后乘以 requantization_scale_c0123
  const float32x4_t vacc0x0123_f =
    vmulq_f32(vcvtq_f32_s32(vacc0x0123), requantization_scale_c0123);
  // 将 vacc1x0123 转换为 FP32，然后乘以 requantization_scale_c0123
  const float32x4_t vacc1x0123_f =
    vmulq_f32(vcvtq_f32_s32(vacc1x0123), requantization_scale_c0123);
  // 将 vacc2x0123 转换为 FP32，然后乘以 requantization_scale_c0123
  const float32x4_t vacc2x0123_f =
    vmulq_f32(vcvtq_f32_s32(vacc2x0123), requantization_scale_c0123);
  // 将 vacc3x0123 转换为 FP32，然后乘以 requantization_scale_c0123
  const float32x4_t vacc3x0123_f =
    vmulq_f32(vcvtq_f32_s32(vacc3x0123), requantization_scale_c0123);
  // 将 vacc0x4567 转换为 FP32，然后乘以 requantization_scale_c4567
  const float32x4_t vacc0x4567_f =
    vmulq_f32(vcvtq_f32_s32(vacc0x4567), requantization_scale_c4567);
  // 将 vacc1x4567 转换为 FP32，然后乘以 requantization_scale_c4567
  const float32x4_t vacc1x4567_f =
    vmulq_f32(vcvtq_f32_s32(vacc1x4567), requantization_scale_c4567);
  // 将 vacc2x4567 转换为 FP32，然后乘以 requantization_scale_c4567
  const float32x4_t vacc2x4567_f =
    vmulq_f32(vcvtq_f32_s32(vacc2x4567), requantization_scale_c4567);
  // 将 vacc3x4567 转换为 FP32，然后乘以 requantization_scale_c4567
  const float32x4_t vacc3x4567_f =
    vmulq_f32(vcvtq_f32_s32(vacc3x4567), requantization_scale_c4567);
#ifdef __aarch64__
  // 加载输出的零点值到向量寄存器中
  const int16x8_t voutput_zero_point =
      vld1q_dup_s16(&quantization_params->neon.output_zero_point);
  /*
   * 使用 ARMv8 指令进行浮点数转换为有符号整数，四舍五入到最近的偶数。
   * 这里对四组向量进行转换，结果会饱和在溢出时，不需要特别处理，因为在最后一阶段会被夹到合适的范围内。
   */
  vacc0x0123 = vcvtnq_s32_f32(vacc0x0123_f);
  vacc1x0123 = vcvtnq_s32_f32(vacc1x0123_f);
  vacc2x0123 = vcvtnq_s32_f32(vacc2x0123_f);
  vacc3x0123 = vcvtnq_s32_f32(vacc3x0123_f);
  vacc0x4567 = vcvtnq_s32_f32(vacc0x4567_f);
  vacc1x4567 = vcvtnq_s32_f32(vacc1x4567_f);
  vacc2x4567 = vcvtnq_s32_f32(vacc2x4567_f);
  vacc3x4567 = vcvtnq_s32_f32(vacc3x4567_f);

  // 计算每组向量的累加结果，同时加上输出零点值
  const int16x8_t vacc0x01234567 = vqaddq_s16(
      vqmovn_high_s32(vqmovn_s32(vacc0x0123), vacc0x4567), voutput_zero_point);
  const int16x8_t vacc1x01234567 = vqaddq_s16(
      vqmovn_high_s32(vqmovn_s32(vacc1x0123), vacc1x4567), voutput_zero_point);
  const int16x8_t vacc2x01234567 = vqaddq_s16(
      vqmovn_high_s32(vqmovn_s32(vacc2x0123), vacc2x4567), voutput_zero_point);
  const int16x8_t vacc3x01234567 = vqaddq_s16(
      vqmovn_high_s32(vqmovn_s32(vacc3x0123), vacc3x4567), voutput_zero_point);

  // 将每组累加结果转换为无符号8位整数，并合并为两个向量
  uint8x16_t vout0x01234567_1x01234567 =
      vqmovun_high_s16(vqmovun_s16(vacc0x01234567), vacc1x01234567);
  uint8x16_t vout2x01234567_3x01234567 =
      vqmovun_high_s16(vqmovun_s16(vacc2x01234567), vacc3x01234567);

  // 加载输出最小和最大值到向量寄存器中
  const uint8x16_t voutput_min =
      vld1q_dup_u8(&quantization_params->neon.output_min);
  const uint8x16_t voutput_max =
      vld1q_dup_u8(&quantization_params->neon.output_max);

  // 对每个输出向量应用上下限制
  vout0x01234567_1x01234567 = vmaxq_u8(vout0x01234567_1x01234567, voutput_min);
  vout2x01234567_3x01234567 = vmaxq_u8(vout2x01234567_3x01234567, voutput_min);
  vout0x01234567_1x01234567 = vminq_u8(vout0x01234567_1x01234567, voutput_max);
  vout2x01234567_3x01234567 = vminq_u8(vout2x01234567_3x01234567, voutput_max);
#endif

// 设置指针 c0、c1、c2、c3，根据 mr 和 nr 的值确定它们的偏移
uint8_t* c0 = c;
uint8_t* c1 = (uint8_t*)((uintptr_t)c0 + c_stride);
if (mr < 2) {
  c1 = c0;  // 如果 mr < 2，c1 指向与 c0 相同的位置
}
uint8_t* c2 = (uint8_t*)((uintptr_t)c1 + c_stride);
if (mr <= 2) {
  c2 = c1;  // 如果 mr <= 2，c2 指向与 c1 相同的位置
}
uint8_t* c3 = (uint8_t*)((uintptr_t)c2 + c_stride);
if (mr != 4) {
  c3 = c2;  // 如果 mr 不等于 4，c3 指向与 c2 相同的位置
}

// 根据 nr 的值，将结果向量写入相应的内存位置
if (nr == 8) {
  vst1_u8(c0, vget_low_u8(vout0x01234567_1x01234567));
  vst1_u8(c1, vget_high_u8(vout0x01234567_1x01234567));
  vst1_u8(c2, vget_low_u8(vout2x01234567_3x01234567));
  vst1_u8(c3, vget_high_u8(vout2x01234567_3x01234567));
} else {
    if (nr >= 4) {
      // 将 vout0x01234567_1x01234567 的第一个 32 位数据写入 c0 指向的地址
      vst1q_lane_u32(
          __builtin_assume_aligned(c0, 1),
          vreinterpretq_u32_u8(vout0x01234567_1x01234567),
          0);
      c0 += 4;
      // 将 vout0x01234567_1x01234567 的第三个 32 位数据写入 c1 指向的地址
      vst1q_lane_u32(
          __builtin_assume_aligned(c1, 1),
          vreinterpretq_u32_u8(vout0x01234567_1x01234567),
          2);
      c1 += 4;
      // 将 vout2x01234567_3x01234567 的第一个 32 位数据写入 c2 指向的地址
      vst1q_lane_u32(
          __builtin_assume_aligned(c2, 1),
          vreinterpretq_u32_u8(vout2x01234567_3x01234567),
          0);
      c2 += 4;
      // 将 vout2x01234567_3x01234567 的第三个 32 位数据写入 c3 指向的地址
      vst1q_lane_u32(
          __builtin_assume_aligned(c3, 1),
          vreinterpretq_u32_u8(vout2x01234567_3x01234567),
          2);
      c3 += 4;
      // 将 vout0x01234567_1x01234567 向左循环移动 4 个字节
      vout0x01234567_1x01234567 =
          vextq_u8(vout0x01234567_1x01234567, vout0x01234567_1x01234567, 4);
      // 将 vout2x01234567_3x01234567 向左循环移动 4 个字节
      vout2x01234567_3x01234567 =
          vextq_u8(vout2x01234567_3x01234567, vout2x01234567_3x01234567, 4);
      nr -= 4;
    }
    if (nr >= 2) {
      // 将 vout0x01234567_1x01234567 的第一个 16 位数据写入 c0 指向的地址
      vst1q_lane_u16(
          __builtin_assume_aligned(c0, 1),
          vreinterpretq_u16_u8(vout0x01234567_1x01234567),
          0);
      c0 += 2;
      // 将 vout0x01234567_1x01234567 的第五个 16 位数据写入 c1 指向的地址
      vst1q_lane_u16(
          __builtin_assume_aligned(c1, 1),
          vreinterpretq_u16_u8(vout0x01234567_1x01234567),
          4);
      c1 += 2;
      // 将 vout2x01234567_3x01234567 的第一个 16 位数据写入 c2 指向的地址
      vst1q_lane_u16(
          __builtin_assume_aligned(c2, 1),
          vreinterpretq_u16_u8(vout2x01234567_3x01234567),
          0);
      c2 += 2;
      // 将 vout2x01234567_3x01234567 的第五个 16 位数据写入 c3 指向的地址
      vst1q_lane_u16(
          __builtin_assume_aligned(c3, 1),
          vreinterpretq_u16_u8(vout2x01234567_3x01234567),
          4);
      c3 += 2;
      // 将 vout0x01234567_1x01234567 向左循环移动 2 个字节
      vout0x01234567_1x01234567 =
          vextq_u8(vout0x01234567_1x01234567, vout0x01234567_1x01234567, 2);
      // 将 vout2x01234567_3x01234567 向左循环移动 2 个字节
      vout2x01234567_3x01234567 =
          vextq_u8(vout2x01234567_3x01234567, vout2x01234567_3x01234567, 2);
      nr -= 2;
    }
    if (nr != 0) {
      // 将 vout0x01234567_1x01234567 的第一个 8 位数据写入 c0 指向的地址
      vst1q_lane_u8(c0, vout0x01234567_1x01234567, 0);
      // 将 vout0x01234567_1x01234567 的第九个 8 位数据写入 c1 指向的地址
      vst1q_lane_u8(c1, vout0x01234567_1x01234567, 8);
      // 将 vout2x01234567_3x01234567 的第一个 8 位数据写入 c2 指向的地址
      vst1q_lane_u8(c2, vout2x01234567_3x01234567, 0);
      // 将 vout2x01234567_3x01234567 的第九个 8 位数据写入 c3 指向的地址
      vst1q_lane_u8(c3, vout2x01234567_3x01234567, 8);
    }
}



# 这行代码表示一个代码块的结束，通常在大多数编程语言中用于结束函数定义、循环、条件语句等代码块。
```