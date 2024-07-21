# `.\pytorch\aten\src\ATen\native\quantized\cpu\qnnpack\src\q8gavgpool\mp8x7p7q-neon.c`

```py
  /*
   * 函数：pytorch_q8gavgpool_ukernel_mp8x7p7q__neon
   * 功能：执行8x7平均池化操作的NEON优化内核函数
   * 参数：
   *   - m: 输入矩阵的行数（要求大于7）
   *   - n: 输入矩阵的列数（至少为8）
   *   - input: 输入数据矩阵的起始地址
   *   - input_stride: 输入数据矩阵的行跨度（每行数据之间的字节数）
   *   - zero: 零填充值（未使用）
   *   - buffer: 缓冲区，用于累积计算中间结果
   *   - output: 输出数据矩阵的起始地址
   *   - quantization_params: 量化参数结构体，包含NEON加权偏置值
   */
  assert(m > 7);  // 确保输入行数大于7
  assert(n >= 8); // 确保输入列数至少为8

  // 初始化输入数据的7行指针
  const uint8_t* i0 = input;
  const uint8_t* i1 = i0 + input_stride;
  const uint8_t* i2 = i1 + input_stride;
  const uint8_t* i3 = i2 + input_stride;
  const uint8_t* i4 = i3 + input_stride;
  const uint8_t* i5 = i4 + input_stride;
  const uint8_t* i6 = i5 + input_stride;

  // 计算实际处理的列数（按8字节对齐）
  const size_t packed_n = (n + 7) & -8;
  // 计算每行数据的增量（每次移动7行）
  const size_t input_increment = 7 * input_stride;
  // 加载量化参数中的NEON偏置值到向量vbias
  const int32x4_t vbias = vld1q_dup_s32(&quantization_params->neon.bias);

  /* 注意：以下计算会超出7个元素的边界 */

  // 初始化累加器指针为缓冲区起始地址
  int32_t* acc = buffer;
  for (size_t k = 0; k < n; k += 8) {
    // 依次加载7行数据的8字节数据到NEON寄存器
    const uint8x8_t vi0 = vld1_u8(i0);
    i0 += 8;
    const uint8x8_t vi1 = vld1_u8(i1);
    i1 += 8;
    const uint8x8_t vi2 = vld1_u8(i2);
    i2 += 8;
    const uint8x8_t vi3 = vld1_u8(i3);
    i3 += 8;
    const uint8x8_t vi4 = vld1_u8(i4);
    i4 += 8;
    const uint8x8_t vi5 = vld1_u8(i5);
    i5 += 8;
    const uint8x8_t vi6 = vld1_u8(i6);
    i6 += 8;

    // 计算每行数据的加权和
    const int16x8_t vsum016 =
        vreinterpretq_s16_u16(vaddw_u8(vaddl_u8(vi0, vi1), vi6));
    const int16x8_t vsum23 = vreinterpretq_s16_u16(vaddl_u8(vi2, vi3));
    const int16x8_t vsum45 = vreinterpretq_s16_u16(vaddl_u8(vi4, vi5));

    // 将偏置值添加到加权和中
    int32x4_t vacc_lo = vaddw_s16(vbias, vget_low_s16(vsum23));
    int32x4_t vacc_hi = vaddw_s16(vbias, vget_high_s16(vsum23));
    vacc_lo = vaddw_s16(vacc_lo, vget_low_s16(vsum45));
    vacc_hi = vaddw_s16(vacc_hi, vget_high_s16(vsum45));
    vacc_lo = vaddw_s16(vacc_lo, vget_low_s16(vsum016));
    vacc_hi = vaddw_s16(vacc_hi, vget_high_s16(vsum016));

    // 存储累加器结果到缓冲区
    vst1q_s32(acc, vacc_lo);
    acc += 4;
    vst1q_s32(acc, vacc_hi);
    acc += 4;
  }

  // 处理剩余的行数（小于等于7行）
  for (m -= 7; m > 7; m -= 7) {
    // 重置累加器指针并更新输入数据指针
    acc = buffer;
    i0 = (const uint8_t*)((uintptr_t)i0 + input_increment);
    i1 = (const uint8_t*)((uintptr_t)i1 + input_increment);
    i2 = (const uint8_t*)((uintptr_t)i2 + input_increment);
    i3 = (const uint8_t*)((uintptr_t)i3 + input_increment);
    i4 = (const uint8_t*)((uintptr_t)i4 + input_increment);
    i5 = (const uint8_t*)((uintptr_t)i5 + input_increment);
    i6 = (const uint8_t*)((uintptr_t)i6 + input_increment);

    /* 注意：以下计算会超出7个元素的边界 */

    // 在处理每8列的循环中，重复之前相同的操作
    // 加载7行数据的8字节数据到NEON寄存器
    const uint8x8_t vi0 = vld1_u8(i0);
    i0 += 8;
    const uint8x8_t vi1 = vld1_u8(i1);
    i1 += 8;
    const uint8x8_t vi2 = vld1_u8(i2);
    i2 += 8;
    const uint8x8_t vi3 = vld1_u8(i3);
    i3 += 8;
    const uint8x8_t vi4 = vld1_u8(i4);
    i4 += 8;
    const uint8x8_t vi5 = vld1_u8(i5);
    i5 += 8;
    const uint8x8_t vi6 = vld1_u8(i6);
    i6 += 8;

    // 计算每行数据的加权和
    const int16x8_t vsum016 =
        vreinterpretq_s16_u16(vaddw_u8(vaddl_u8(vi0, vi1), vi6));
    const int16x8_t vsum23 = vreinterpretq_s16_u16(vaddl_u8(vi2, vi3));
    const int16x8_t vsum45 = vreinterpretq_s16_u16(vaddl_u8(vi4, vi5));

    // 将偏置值添加到加权和中
    int32x4_t vacc_lo = vaddw_s16(vbias, vget_low_s16(vsum23));
    int32x4_t vacc_hi = vaddw_s16(vbias, vget_high_s16(vsum23));
    vacc_lo = vaddw_s16(vacc_lo, vget_low_s16(vsum45));
    vacc_hi = vaddw_s16(vacc_hi, vget_high_s16(vsum45));
    vacc_lo = vaddw_s16(vacc_lo, vget_low_s16(vsum016));
    vacc_hi = vaddw_s16(vacc_hi, vget_high_s16(vsum016));

    // 存储累加器结果到缓冲区
    vst1q_s32(acc, vacc_lo);
    acc += 4;
    vst1q_s32(acc, vacc_hi);
    acc += 4;
  }
}
    // 以步长为8遍历处理输入数据，每次处理8个元素
    for (size_t k = 0; k < n; k += 8) {
      // 从指针 i0 加载 8 个无符号 8 位整数，存入 vi0，并递增指针 i0
      const uint8x8_t vi0 = vld1_u8(i0);
      i0 += 8;
      // 从指针 i1 加载 8 个无符号 8 位整数，存入 vi1，并递增指针 i1
      const uint8x8_t vi1 = vld1_u8(i1);
      i1 += 8;
      // 从指针 i2 加载 8 个无符号 8 位整数，存入 vi2，并递增指针 i2
      const uint8x8_t vi2 = vld1_u8(i2);
      i2 += 8;
      // 从指针 i3 加载 8 个无符号 8 位整数，存入 vi3，并递增指针 i3
      const uint8x8_t vi3 = vld1_u8(i3);
      i3 += 8;
      // 从指针 i4 加载 8 个无符号 8 位整数，存入 vi4，并递增指针 i4
      const uint8x8_t vi4 = vld1_u8(i4);
      i4 += 8;
      // 从指针 i5 加载 8 个无符号 8 位整数，存入 vi5，并递增指针 i5
      const uint8x8_t vi5 = vld1_u8(i5);
      i5 += 8;
      // 从指针 i6 加载 8 个无符号 8 位整数，存入 vi6，并递增指针 i6
      const uint8x8_t vi6 = vld1_u8(i6);
      i6 += 8;
      // 从指针 acc 加载 4 个有符号 32 位整数，存入 vacc_lo 和 vacc_hi
      int32x4_t vacc_lo = vld1q_s32(acc);
      int32x4_t vacc_hi = vld1q_s32(acc + 4);

      // 计算 vi0、vi1 和 vi6 的无符号整数相加，并将结果转换为有符号 16 位整数
      const int16x8_t vsum016 =
          vreinterpretq_s16_u16(vaddw_u8(vaddl_u8(vi0, vi1), vi6));
      // 计算 vi2 和 vi3 的无符号整数相加，并将结果转换为有符号 16 位整数
      const int16x8_t vsum23 = vreinterpretq_s16_u16(vaddl_u8(vi2, vi3));
      // 计算 vi4 和 vi5 的无符号整数相加，并将结果转换为有符号 16 位整数
      const int16x8_t vsum45 = vreinterpretq_s16_u16(vaddl_u8(vi4, vi5));

      // 将 vsum23 的低位元素加到 vacc_lo 中
      vacc_lo = vaddw_s16(vacc_lo, vget_low_s16(vsum23));
      // 将 vsum23 的高位元素加到 vacc_hi 中
      vacc_hi = vaddw_s16(vacc_hi, vget_high_s16(vsum23));
      // 将 vsum45 的低位元素加到 vacc_lo 中
      vacc_lo = vaddw_s16(vacc_lo, vget_low_s16(vsum45));
      // 将 vsum45 的高位元素加到 vacc_hi 中
      vacc_hi = vaddw_s16(vacc_hi, vget_high_s16(vsum45));
      // 将 vsum016 的低位元素加到 vacc_lo 中
      vacc_lo = vaddw_s16(vacc_lo, vget_low_s16(vsum016));
      // 将 vsum016 的高位元素加到 vacc_hi 中
      vacc_hi = vaddw_s16(vacc_hi, vget_high_s16(vsum016));
      // 将 vacc_lo 存回 acc 指向的位置，并递增指针 acc
      vst1q_s32(acc, vacc_lo);
      acc += 4;
      // 将 vacc_hi 存回 acc 指向的位置，并递增指针 acc
      vst1q_s32(acc, vacc_hi);
      acc += 4;
    }
  }

  // 从 quantization_params 指向的结构体中获取 NEON 的缩放参数，构造为一个常量 float32x4_t
  const float32x4_t vscale =
      vdupq_n_f32(quantization_params->neon.scale);
#if defined(__aarch64__)
  // 如果目标平台为 AArch64 架构，则使用 NEON 指令集加载输出零点值
  const int16x8_t voutput_zero_point =
      vld1q_dup_s16(&quantization_params->neon.output_zero_point);
  // 加载 NEON 指令集输出最小值
  const uint8x8_t voutput_min =
      vld1_dup_u8(&quantization_params->neon.output_min);
  // 加载 NEON 指令集输出最大值
  const uint8x8_t voutput_max =
      vld1_dup_u8(&quantization_params->neon.output_max);
#else
  // 如果不是 AArch64 架构，则加载浮点数运算需要的常量
  const float32x4_t vfmin = vdupq_n_f32(quantization_params->neon.vfmin);
  const float32x4_t vfmax = vdupq_n_f32(quantization_params->neon.vfmax);
  const float32x4_t vfmagic = vdupq_n_f32(quantization_params->neon.vfmagic);
  const int32x4_t vimagic = vdupq_n_s32(quantization_params->neon.vimagic);
#endif

// 更新指针以跳到下一个输入数据块
i0 = (const uint8_t*)((uintptr_t)i0 + input_increment);
i1 = (const uint8_t*)((uintptr_t)i1 + input_increment);
// 如果 m 小于 2，将 i1 指向零向量
if (m < 2) {
  i1 = zero;
}
i2 = (const uint8_t*)((uintptr_t)i2 + input_increment);
// 如果 m 小于等于 2，将 i2 指向零向量
if (m <= 2) {
  i2 = zero;
}
i3 = (const uint8_t*)((uintptr_t)i3 + input_increment);
// 如果 m 小于 4，将 i3 指向零向量
if (m < 4) {
  i3 = zero;
}
i4 = (const uint8_t*)((uintptr_t)i4 + input_increment);
// 如果 m 小于等于 4，将 i4 指向零向量
if (m <= 4) {
  i4 = zero;
}
i5 = (const uint8_t*)((uintptr_t)i5 + input_increment);
// 如果 m 小于 6，将 i5 指向零向量
if (m < 6) {
  i5 = zero;
}
i6 = (const uint8_t*)((uintptr_t)i6 + input_increment);
// 如果 m 小于等于 6，将 i6 指向零向量
if (m <= 6) {
  i6 = zero;
}

// 初始化累加器为缓冲区
acc = buffer;
// 执行向量化累加操作
do {
  // 加载并递增输入数据块 i0 到 i6
  const uint8x8_t vi0 = vld1_u8(i0);
  i0 += 8;
  const uint8x8_t vi1 = vld1_u8(i1);
  i1 += 8;
  const uint8x8_t vi2 = vld1_u8(i2);
  i2 += 8;
  const uint8x8_t vi3 = vld1_u8(i3);
  i3 += 8;
  const uint8x8_t vi4 = vld1_u8(i4);
  i4 += 8;
  const uint8x8_t vi5 = vld1_u8(i5);
  i5 += 8;
  const uint8x8_t vi6 = vld1_u8(i6);
  i6 += 8;
  // 加载并递增累加器中的低位和高位数据
  int32x4_t vacc_lo = vld1q_s32(acc);
  acc += 4;
  int32x4_t vacc_hi = vld1q_s32(acc);
  acc += 4;

  // 计算每个输入数据块的和，并转换为有符号 16 位整数
  const int16x8_t vsum016 =
      vreinterpretq_s16_u16(vaddw_u8(vaddl_u8(vi0, vi1), vi6));
  const int16x8_t vsum23 = vreinterpretq_s16_u16(vaddl_u8(vi2, vi3));
  const int16x8_t vsum45 = vreinterpretq_s16_u16(vaddl_u8(vi4, vi5));

  // 累加计算结果到低位和高位累加器
  vacc_lo = vaddw_s16(vacc_lo, vget_low_s16(vsum23));
  vacc_hi = vaddw_s16(vacc_hi, vget_high_s16(vsum23));
  vacc_lo = vaddw_s16(vacc_lo, vget_low_s16(vsum45));
  vacc_hi = vaddw_s16(vacc_hi, vget_high_s16(vsum45));
  vacc_lo = vaddw_s16(vacc_lo, vget_low_s16(vsum016));
  vacc_hi = vaddw_s16(vacc_hi, vget_high_s16(vsum016));

  // 将累加器数据转换为浮点数，并乘以缩放因子
  float32x4_t vacc_lo_f = vcvtq_f32_s32(vacc_lo);
  float32x4_t vacc_hi_f = vcvtq_f32_s32(vacc_hi);

  vacc_lo_f = vmulq_f32(vacc_lo_f, vscale);
  vacc_hi_f = vmulq_f32(vacc_hi_f, vscale);

#if defined(__aarch64__)
  // 如果是 AArch64 架构，将浮点数累加器数据转换为有符号 16 位整数
  vacc_lo = vcvtnq_s32_f32(vacc_lo_f);
  vacc_hi = vcvtnq_s32_f32(vacc_hi_f);
  // 对累加器数据应用输出零点值，并转换为无符号 8 位整数
  const int16x8_t vacc = vqaddq_s16(
      vqmovn_high_s32(vqmovn_s32(vacc_lo), vacc_hi), voutput_zero_point);
  uint8x8_t vout = vqmovun_s16(vacc);
  vout = vmax_u8(vout, voutput_min);
  vout = vmin_u8(vout, voutput_max);
#else
  // 如果不是 AArch64 架构，对浮点数累加器数据进行范围限制
  vacc_lo_f = vminq_f32(vmaxq_f32(vacc_lo_f, vfmin), vfmax);
  vacc_hi_f = vminq_f32(vmaxq_f32(vacc_hi_f, vfmin), vfmax);

  // 对浮点数累加器数据进行范围限制
  // 计算下界和上界
  vacc_lo_f = vminq_f32(vmaxq_f32(vacc_lo_f, vfmin), vfmax);
  vacc_hi_f = vminq_f32(vmaxq_f32(vacc_hi_f, vfmin), vfmax);

  // 如果不是 AArch64 架构，对浮点数累加器数据进行范围限制
  // 计算下界和上界，并将结果存储回浮点数累加器变量
  vacc_lo_f = vminq_f32(vmaxq_f32(vacc_lo_f, vfmin), vfmax);
  vacc_hi_f = vminq_f32(vmaxq_f32(vacc_hi_f, vfmin), vfmax);
#endif


这样完成了对代码的注释，确保每一行都有详细的解释，符合注释的要求和格式。
    # 将vacc_lo_f与vfmagic相加，然后将结果重新解释为32位整数向量，再减去vimagic
    vacc_lo = vsubq_s32(
        vreinterpretq_s32_f32(vaddq_f32(vacc_lo_f, vfmagic)), vimagic);
    
    # 将vacc_hi_f与vfmagic相加，然后将结果重新解释为32位整数向量，再减去vimagic
    vacc_hi = vsubq_s32(
        vreinterpretq_s32_f32(vaddq_f32(vacc_hi_f, vfmagic)), vimagic);
    
    # 将vacc_lo和vacc_hi转换为16位整数向量，并合并成一个新的16位整数向量vacc
    const int16x8_t vacc =
        vcombine_s16(vqmovn_s32(vacc_lo), vqmovn_s32(vacc_hi));
    
    # 将vacc中的16位整数向量转换为无符号8位整数向量vout
    uint8x8_t vout = vqmovun_s16(vacc);
#endif
    // 创建一个 int16x8_t 类型的变量 vacc，使用 vcombine_s16 将两个 int16x4_t 类型的变量合并成一个 int16x8_t 类型的变量
    const int16x8_t vacc =
        vcombine_s16(vqmovn_s32(vacc_lo), vqmovn_s32(vacc_hi));

    // 创建一个 uint8x8_t 类型的变量 vout，使用 vqmovun_s16 将 int16x8_t 类型的变量 vacc 转换为 uint8x8_t 类型
    uint8x8_t vout = vqmovun_s16(vacc);
#endif

// 检查是否需要处理剩余元素
if (n & 4) {
  // 将 32 位无符号整数向量中的第一个元素写入内存，假设输出地址已对齐
  vst1_lane_u32(
      __builtin_assume_aligned(output, 1), vreinterpret_u32_u8(vout), 0);
  // 移动输出指针到下一个位置
  output += 4;
  // 向左循环移位 8 位向量 vout，将结果存储回 vout
  vout = vext_u8(vout, vout, 4);
}
if (n & 2) {
  // 将 16 位无符号整数向量中的第一个元素写入内存，假设输出地址已对齐
  vst1_lane_u16(
      __builtin_assume_aligned(output, 1), vreinterpret_u16_u8(vout), 0);
  // 移动输出指针到下一个位置
  output += 2;
  // 向左循环移位 8 位向量 vout，将结果存储回 vout
  vout = vext_u8(vout, vout, 2);
}
if (n & 1) {
  // 将 8 位无符号整数向量中的第一个元素写入内存
  vst1_lane_u8(output, vout, 0);
}
// 函数结束
}
```