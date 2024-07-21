# `.\pytorch\aten\src\ATen\native\quantized\cpu\qnnpack\src\q8avgpool\mp8x9p8q-neon.c`

```py
/*
 * 确保输入数量非零
 * 参数 n 表示输入的数量
 */
assert(n != 0);

/*
 * 确保内核大小大于9
 * 参数 ks 表示内核的大小
 */
assert(ks > 9);

/*
 * 确保通道数不少于8
 * 参数 kc 表示通道数
 */
assert(kc >= 8);

/*
 * 使用量化参数中的偏置创建 NEON 向量
 * 参数 quantization_params 包含量化参数的联合体
 */
const int32x4_t vbias = vld1q_dup_s32(&quantization_params->neon.bias);

/*
 * 使用量化参数中的缩放因子创建 NEON 浮点向量
 * 参数 quantization_params 包含量化参数的联合体
 */
const float32x4_t vscale = vdupq_n_f32(quantization_params->neon.scale);

#if defined(__aarch64__)
/*
 * 在 AArch64 架构下，使用量化参数中的输出零点创建 NEON 16位整数向量
 * 参数 quantization_params 包含量化参数的联合体
 */
const int16x8_t voutput_zero_point =
    vld1q_dup_s16(&quantization_params->neon.output_zero_point);

/*
 * 在 AArch64 架构下，使用量化参数中的输出最小值创建 NEON 8位无符号整数向量
 * 参数 quantization_params 包含量化参数的联合体
 */
const uint8x8_t voutput_min =
    vld1_dup_u8(&quantization_params->neon.output_min);

/*
 * 在 AArch64 架构下，使用量化参数中的输出最大值创建 NEON 8位无符号整数向量
 * 参数 quantization_params 包含量化参数的联合体
 */
const uint8x8_t voutput_max =
    vld1_dup_u8(&quantization_params->neon.output_max);
#else
/*
 * 在非 AArch64 架构下，使用量化参数中的浮点最小值创建 NEON 浮点向量
 * 参数 quantization_params 包含量化参数的联合体
 */
const float32x4_t vfmin = vdupq_n_f32(quantization_params->neon.vfmin);

/*
 * 在非 AArch64 架构下，使用量化参数中的浮点最大值创建 NEON 浮点向量
 * 参数 quantization_params 包含量化参数的联合体
 */
const float32x4_t vfmax = vdupq_n_f32(quantization_params->neon.vfmax);

/*
 * 在非 AArch64 架构下，使用量化参数中的魔数创建 NEON 浮点向量
 * 参数 quantization_params 包含量化参数的联合体
 */
const float32x4_t vfmagic = vdupq_n_f32(quantization_params->neon.vfmagic);

/*
 * 在非 AArch64 架构下，使用量化参数中的魔数创建 NEON 整数向量
 * 参数 quantization_params 包含量化参数的联合体
 */
const int32x4_t vimagic = vdupq_n_s32(quantization_params->neon.vimagic);
#endif

/*
 * 执行池化操作的主循环，处理输入数据
 * 使用给定的输入和输出增量
 * 参数 input_increment 表示输入增量大小
 * 参数 output_increment 表示输出增量大小
 */
do {
    // 待实现的池化操作
    // 此处省略具体实现细节
} while (--n != 0);

/*
 * 保存内核大小的副本
 * 参数 ks 表示内核大小
 */
size_t m = ks;
      // 定义指向输入数组中不同索引位置的指针，从0到7
      const uint8_t* i0 = input[0];
      const uint8_t* i1 = input[1];
      const uint8_t* i2 = input[2];
      const uint8_t* i3 = input[3];
      const uint8_t* i4 = input[4];
      const uint8_t* i5 = input[5];
      const uint8_t* i6 = input[6];
      const uint8_t* i7 = input[7];
      // 将输入指针数组整体向后偏移指定的增量
      input = (const uint8_t**)((uintptr_t)input + input_increment);
      // 根据条件 m 的值，将部分指针设置为 zero 指针
      if (m < 2) {
        i1 = zero;
      }
      if (m <= 2) {
        i2 = zero;
      }
      if (m < 4) {
        i3 = zero;
      }
      if (m <= 4) {
        i4 = zero;
      }
      if (m < 6) {
        i5 = zero;
      }
      if (m <= 6) {
        i6 = zero;
      }
      if (m != 8) {
        i7 = zero;
      }

      // 初始化循环计数器 k 为 kc，初始化累加器指针 acc 为 buffer
      size_t k = kc;
      int32_t* acc = buffer;
      // 当 k 大于等于 8 时执行循环
      while (k >= 8) {
        // 加载连续内存中的 8 个 uint8_t 元素，转换为 uint8x8_t 类型
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
        const uint8x8_t vi7 = vld1_u8(i7);
        i7 += 8;
        // 加载累加器中的 8 个 int32_t 元素，转换为 int32x4_t 类型
        int32x4_t vacc_lo = vld1q_s32(acc);
        acc += 4;
        int32x4_t vacc_hi = vld1q_s32(acc);
        acc += 4;

        // 将 vi0 到 vi7 中的每一对 uint8x8_t 对象，相加并转换为 int16x8_t 类型
        const int16x8_t vsum01 = vreinterpretq_s16_u16(vaddl_u8(vi0, vi1));
        const int16x8_t vsum23 = vreinterpretq_s16_u16(vaddl_u8(vi2, vi3));
        const int16x8_t vsum45 = vreinterpretq_s16_u16(vaddl_u8(vi4, vi5));
        const int16x8_t vsum67 = vreinterpretq_s16_u16(vaddl_u8(vi6, vi7));

        // 分别计算两组相邻的 vsum 对象的和，结果为 int16x8_t 类型
        const int16x8_t vsum0123 = vaddq_s16(vsum01, vsum23);
        const int16x8_t vsum4567 = vaddq_s16(vsum45, vsum67);
        const int16x8_t vsum = vaddq_s16(vsum0123, vsum4567);

        // 将 vsum 的低四个和高四个 int16_t 值，加到对应的 vacc_lo 和 vacc_hi 中
        vacc_lo = vaddw_s16(vacc_lo, vget_low_s16(vsum));
        vacc_hi = vaddw_s16(vacc_hi, vget_high_s16(vsum));

        // 将 vacc_lo 和 vacc_hi 中的 int32_t 值转换为 float32x4_t 类型
        float32x4_t vacc_lo_f = vcvtq_f32_s32(vacc_lo);
        float32x4_t vacc_hi_f = vcvtq_f32_s32(vacc_hi);

        // 分别将 vacc_lo_f 和 vacc_hi_f 中的每个 float32x4_t 值乘以 vscale
        vacc_lo_f = vmulq_f32(vacc_lo_f, vscale);
        vacc_hi_f = vmulq_f32(vacc_hi_f, vscale);
#if defined(__aarch64__)
        # 如果定义了__aarch64__宏，则执行以下操作（ARM 64位架构特定）
        vacc_lo = vcvtnq_s32_f32(vacc_lo_f);
        # 将vacc_lo_f中的每个32位浮点数转换为32位整数，结果存储在vacc_lo中
        vacc_hi = vcvtnq_s32_f32(vacc_hi_f);
        # 将vacc_hi_f中的每个32位浮点数转换为32位整数，结果存储在vacc_hi中
        const int16x8_t vacc = vqaddq_s16(
            vqmovn_high_s32(vqmovn_s32(vacc_lo), vacc_hi), voutput_zero_point);
        # 使用vqmovn_s32函数将vacc_lo和vacc_hi中的每个32位整数转换为16位整数，
        # 然后将结果合并为一个int16x8_t类型的向量vacc
        # 使用vqmovn_high_s32函数将vacc_lo的高位和vacc_hi中的数据结合
        # vqaddq_s16将vacc中的每个16位整数与voutput_zero_point相加，结果存储在vacc中
        uint8x8_t vout = vqmovun_s16(vacc);
        # 将vacc中的每个16位整数转换为无符号8位整数，结果存储在vout中
        vout = vmax_u8(vout, voutput_min);
        # 将vout中的每个8位整数与voutput_min中的每个8位整数相比较，将每个位置上较大的值存储在vout中
        vout = vmin_u8(vout, voutput_max);
        # 将vout中的每个8位整数与voutput_max中的每个8位整数相比较，将每个位置上较小的值存储在vout中
#else
        # 如果未定义__aarch64__宏，则执行以下操作（通用处理）
        vacc_lo_f = vminq_f32(vmaxq_f32(vacc_lo_f, vfmin), vfmax);
        # 将vacc_lo_f中的每个32位浮点数与vfmin中的每个32位浮点数比较，将每个位置上较大的值存储在vacc_lo_f中
        # 然后将结果与vfmax中的每个32位浮点数比较，将每个位置上较小的值存储在vacc_lo_f中
        vacc_hi_f = vminq_f32(vmaxq_f32(vacc_hi_f, vfmin), vfmax);
        # 将vacc_hi_f中的每个32位浮点数与vfmin中的每个32位浮点数比较，将每个位置上较大的值存储在vacc_hi_f中
        # 然后将结果与vfmax中的每个32位浮点数比较，将每个位置上较小的值存储在vacc_hi_f中

        vacc_lo = vsubq_s32(
            vreinterpretq_s32_f32(vaddq_f32(vacc_lo_f, vfmagic)), vimagic);
        # 将vacc_lo_f中的每个32位浮点数加上vfmagic中的每个32位浮点数，然后将结果转换为32位整数
        # 再将结果与vimagic中的每个32位整数相减，将每个位置上的结果存储在vacc_lo中
        vacc_hi = vsubq_s32(
            vreinterpretq_s32_f32(vaddq_f32(vacc_hi_f, vfmagic)), vimagic);
        # 将vacc_hi_f中的每个32位浮点数加上vfmagic中的每个32位浮点数，然后将结果转换为32位整数
        # 再将结果与vimagic中的每个32位整数相减，将每个位置上的结果存储在vacc_hi中

        const int16x8_t vacc =
            vcombine_s16(vqmovn_s32(vacc_lo), vqmovn_s32(vacc_hi));
        # 使用vqmovn_s32函数将vacc_lo和vacc_hi中的每个32位整数转换为16位整数，
        # 然后将结果合并为一个int16x8_t类型的向量vacc

        uint8x8_t vout = vqmovun_s16(vacc);
        # 将vacc中的每个16位整数转换为无符号8位整数，结果存储在vout中
        // 将计算结果写入输出内存中
        vst1_u8(output, vout);
        // 更新输出指针，指向下一个输出位置
        output += 8;

        // 更新剩余元素个数
        k -= 8;
      }
      // 处理剩余元素个数不足8的情况
      if (k != 0) {
        // 计算地址增量
        const size_t address_increment = k - 8;
        // 更新输入指针，根据地址增量
        i0 = (const uint8_t*)((uintptr_t)i0 + address_increment);
        i1 = (const uint8_t*)((uintptr_t)i1 + address_increment);
        i2 = (const uint8_t*)((uintptr_t)i2 + address_increment);
        i3 = (const uint8_t*)((uintptr_t)i3 + address_increment);
        i4 = (const uint8_t*)((uintptr_t)i4 + address_increment);
        i5 = (const uint8_t*)((uintptr_t)i5 + address_increment);
        i6 = (const uint8_t*)((uintptr_t)i6 + address_increment);
        i7 = (const uint8_t*)((uintptr_t)i7 + address_increment);
        // 计算移位量并创建移位向量
        const int64x1_t vshift = vmov_n_s64(8 * address_increment);

        // 加载并左移输入向量的数据
        const uint8x8_t vi0 = vreinterpret_u8_u64(
            vshl_u64(vreinterpret_u64_u8(vld1_u8(i0)), vshift));
        const uint8x8_t vi1 = vreinterpret_u8_u64(
            vshl_u64(vreinterpret_u64_u8(vld1_u8(i1)), vshift));
        const uint8x8_t vi2 = vreinterpret_u8_u64(
            vshl_u64(vreinterpret_u64_u8(vld1_u8(i2)), vshift));
        const uint8x8_t vi3 = vreinterpret_u8_u64(
            vshl_u64(vreinterpret_u64_u8(vld1_u8(i3)), vshift));
        const uint8x8_t vi4 = vreinterpret_u8_u64(
            vshl_u64(vreinterpret_u64_u8(vld1_u8(i4)), vshift));
        const uint8x8_t vi5 = vreinterpret_u8_u64(
            vshl_u64(vreinterpret_u64_u8(vld1_u8(i5)), vshift));
        const uint8x8_t vi6 = vreinterpret_u8_u64(
            vshl_u64(vreinterpret_u64_u8(vld1_u8(i6)), vshift));
        const uint8x8_t vi7 = vreinterpret_u8_u64(
            vshl_u64(vreinterpret_u64_u8(vld1_u8(i7)), vshift));

        // 加载累加器中的数据
        int32x4_t vacc_lo = vld1q_s32(acc);
        acc += 4;
        int32x4_t vacc_hi = vld1q_s32(acc);

        // 计算输入向量的数据之和
        const int16x8_t vsum01 = vreinterpretq_s16_u16(vaddl_u8(vi0, vi1));
        const int16x8_t vsum23 = vreinterpretq_s16_u16(vaddl_u8(vi2, vi3));
        const int16x8_t vsum45 = vreinterpretq_s16_u16(vaddl_u8(vi4, vi5));
        const int16x8_t vsum67 = vreinterpretq_s16_u16(vaddl_u8(vi6, vi7));

        // 计算累加和
        const int16x8_t vsum0123 = vaddq_s16(vsum01, vsum23);
        const int16x8_t vsum4567 = vaddq_s16(vsum45, vsum67);
        const int16x8_t vsum = vaddq_s16(vsum0123, vsum4567);

        // 更新累加器
        vacc_lo = vaddw_s16(vacc_lo, vget_low_s16(vsum));
        vacc_hi = vaddw_s16(vacc_hi, vget_high_s16(vsum));

        // 将累加器的整数部分转换为浮点数
        float32x4_t vacc_lo_f = vcvtq_f32_s32(vacc_lo);
        float32x4_t vacc_hi_f = vcvtq_f32_s32(vacc_hi);

        // 乘以缩放因子
        vacc_lo_f = vmulq_f32(vacc_lo_f, vscale);
        vacc_hi_f = vmulq_f32(vacc_hi_f, vscale);

        // 对于 AArch64 平台的特定处理
#if defined(__aarch64__)
        // 将浮点数累加器四舍五入为整数
        vacc_lo = vcvtnq_s32_f32(vacc_lo_f);
        vacc_hi = vcvtnq_s32_f32(vacc_hi_f);
        // 对结果进行饱和处理
        const int16x8_t vacc = vqaddq_s16(
            vqmovn_high_s32(vqmovn_s32(vacc_lo), vacc_hi), voutput_zero_point);
        // 转换为无符号字节向量
        uint8x8_t vout = vqmovun_s16(vacc);
        // 将结果限制在指定范围内
        vout = vmax_u8(vout, voutput_min);
        vout = vmin_u8(vout, voutput_max);
        vacc_lo_f = vminq_f32(vmaxq_f32(vacc_lo_f, vfmin), vfmax);
        // 将vacc_lo_f与vfmin比较取较大值，再与vfmax比较取较小值，结果存入vacc_lo_f

        vacc_hi_f = vminq_f32(vmaxq_f32(vacc_hi_f, vfmin), vfmax);
        // 将vacc_hi_f与vfmin比较取较大值，再与vfmax比较取较小值，结果存入vacc_hi_f

        vacc_lo = vsubq_s32(
            vreinterpretq_s32_f32(vaddq_f32(vacc_lo_f, vfmagic)), vimagic);
        // 将vacc_lo_f与vfmagic相加后转换为整型，再减去vimagic，结果存入vacc_lo

        vacc_hi = vsubq_s32(
            vreinterpretq_s32_f32(vaddq_f32(vacc_hi_f, vfmagic)), vimagic);
        // 将vacc_hi_f与vfmagic相加后转换为整型，再减去vimagic，结果存入vacc_hi

        const int16x8_t vacc =
            vcombine_s16(vqmovn_s32(vacc_lo), vqmovn_s32(vacc_hi));
        // 将vacc_lo和vacc_hi转换为16位整数，然后组合成一个int16x8_t向量

        uint8x8_t vout = vqmovun_s16(vacc);
        // 将int16x8_t类型的vacc向量转换为uint8x8_t类型的vout向量

#ifdef

        if (k & 4) {
          vst1_lane_u32(
              __builtin_assume_aligned(output, 1),
              vreinterpret_u32_u8(vout),
              0);
          // 如果k的最后两位是11，则将vout向量的第0个元素存储到output指向的地址，每次存储4个字节
          output += 4;
          // 更新output指针，使其指向下一个位置
          vout = vext_u8(vout, vout, 4);
          // 将vout向量向左移动4个字节，更新vout的值
        }
        if (k & 2) {
          vst1_lane_u16(
              __builtin_assume_aligned(output, 1),
              vreinterpret_u16_u8(vout),
              0);
          // 如果k的倒数第二位是1，则将vout向量的第0个元素存储到output指向的地址，每次存储2个字节
          output += 2;
          // 更新output指针，使其指向下一个位置
          vout = vext_u8(vout, vout, 2);
          // 将vout向量向左移动2个字节，更新vout的值
        }
        if (k & 1) {
          vst1_lane_u8(output, vout, 0);
          // 如果k的最后一位是1，则将vout向量的第0个元素存储到output指向的地址，每次存储1个字节
          output += 1;
          // 更新output指针，使其指向下一个位置
        }
#endif
      }
    }
    output = (uint8_t*)((uintptr_t)output + output_increment);
    // 更新output指针，使其跳过一个输出增量的大小

  } while (--n != 0);
  // 循环直到n减到0为止
}
```