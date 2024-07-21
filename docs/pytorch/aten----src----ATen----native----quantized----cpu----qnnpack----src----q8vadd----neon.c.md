# `.\pytorch\aten\src\ATen\native\quantized\cpu\qnnpack\src\q8vadd\neon.c`

```py
/*
 * 版权声明: Facebook, Inc. 及其关联公司保留所有权利。
 *
 * 此源代码使用 BSD 风格许可证授权，许可证可在源树根目录下的 LICENSE 文件中找到。
 */

#include <arm_neon.h>  // 包含 ARM NEON 指令集的头文件

#include <qnnpack/common.h>  // 包含 QNNPACK 公共头文件
#include <qnnpack/q8vadd.h>  // 包含 QNNPACK Q8VADD 函数声明

void pytorch_q8vadd_ukernel__neon(
    size_t n,
    const uint8_t* a,
    const uint8_t* b,
    uint8_t* y,
    const union pytorch_qnnp_add_quantization_params
        quantization_params[restrict static 1]) {
  const uint8x8_t va_zero_point =  // 加载 a 的零点值到 ARM NEON 寄存器
      vld1_dup_u8(&quantization_params->neon.a_zero_point);
  const uint8x8_t vb_zero_point =  // 加载 b 的零点值到 ARM NEON 寄存器
      vld1_dup_u8(&quantization_params->neon.b_zero_point);
  const int16x8_t vy_zero_point =  // 加载 y 的零点值到 ARM NEON 寄存器
      vld1q_dup_s16(&quantization_params->neon.y_zero_point);
  const int32x4_t va_multiplier =  // 加载 a 的乘法因子到 ARM NEON 寄存器
      vld1q_dup_s32(&quantization_params->neon.a_multiplier);
  const int32x4_t vb_multiplier =  // 加载 b 的乘法因子到 ARM NEON 寄存器
      vld1q_dup_s32(&quantization_params->neon.b_multiplier);
  const int32x4_t vright_shift =  // 加载右移位数到 ARM NEON 寄存器
      vld1q_dup_s32(&quantization_params->neon.right_shift);
  const int32x4_t vzero_shift_mask =  // 构造零移位掩码
      vreinterpretq_s32_u32(vceqq_s32(vright_shift, vmovq_n_s32(0)));
  const uint8x16_t vy_max =  // 加载 y 的最大值到 ARM NEON 寄存器
      vld1q_dup_u8(&quantization_params->neon.y_max);
  const uint8x16_t vy_min =  // 加载 y 的最小值到 ARM NEON 寄存器
      vld1q_dup_u8(&quantization_params->neon.y_min);
  if
    PYTORCH_QNNP_LIKELY(n >= 8) {  // 如果 n 大于等于 8，则执行以下操作
#else
      // 循环处理剩余长度大于等于16的情况
      for (; n >= 16; n -= 16) {
        // 加载并递增指针a指向的16个无符号8位整数到va01
        const uint8x16_t va01 = vld1q_u8(a);
        a += 16;
        // 加载并递增指针b指向的16个无符号8位整数到vb01
        const uint8x16_t vb01 = vld1q_u8(b);
        b += 16;

        /* Subtract zero point */
        // 将va01的低8位和高8位减去零点，转换为有符号16位整数存入vxa0、vxa1、vxb0、vxb1
        const int16x8_t vxa0 =
            vreinterpretq_s16_u16(vsubl_u8(vget_low_u8(va01), va_zero_point));
        const int16x8_t vxb0 =
            vreinterpretq_s16_u16(vsubl_u8(vget_low_u8(vb01), vb_zero_point));
        const int16x8_t vxa1 =
            vreinterpretq_s16_u16(vsubl_u8(vget_high_u8(va01), va_zero_point));
        const int16x8_t vxb1 =
            vreinterpretq_s16_u16(vsubl_u8(vget_high_u8(vb01), vb_zero_point));

        /* Multiply by factors and accumulate products */
        // 使用va_multiplier对vxa0、vxa1、vxb0、vxb1进行有符号32位整数乘法并累加到vacc0_lo、vacc1_lo、vacc0_hi、vacc1_hi
        int32x4_t vacc0_lo =
            vmulq_s32(vmovl_s16(vget_low_s16(vxa0)), va_multiplier);
        int32x4_t vacc1_lo =
            vmulq_s32(vmovl_s16(vget_low_s16(vxa1)), va_multiplier);
        int32x4_t vacc0_hi =
            vmulq_s32(vmovl_s16(vget_high_s16(vxa0)), va_multiplier);
        int32x4_t vacc1_hi =
            vmulq_s32(vmovl_s16(vget_high_s16(vxa1)), va_multiplier);

        // 预取下一个操作的数据到缓存，提高后续操作的访问速度
        __builtin_prefetch(a + 640);
        __builtin_prefetch(b + 640);

        // 使用vb_multiplier对vxb0、vxb1进行有符号32位整数乘法并累加到vacc0_lo、vacc1_lo、vacc0_hi、vacc1_hi
        vacc0_lo =
            vmlaq_s32(vacc0_lo, vmovl_s16(vget_low_s16(vxb0)), vb_multiplier);
        vacc1_lo =
            vmlaq_s32(vacc1_lo, vmovl_s16(vget_low_s16(vxb1)), vb_multiplier);
        vacc0_hi =
            vmlaq_s32(vacc0_hi, vmovl_s16(vget_high_s16(vxb0)), vb_multiplier);
        vacc1_hi =
            vmlaq_s32(vacc1_hi, vmovl_s16(vget_high_s16(vxb1)), vb_multiplier);

        /* Shift right and round */
        // 对vacc0_lo、vacc1_lo、vacc0_hi、vacc1_hi进行右移位并进行舍入操作
        vacc0_lo =
            vsraq_n_s32(vacc0_lo, vbicq_s32(vacc0_lo, vzero_shift_mask), 31);
        vacc1_lo =
            vsraq_n_s32(vacc1_lo, vbicq_s32(vacc1_lo, vzero_shift_mask), 31);
        vacc0_hi =
            vsraq_n_s32(vacc0_hi, vbicq_s32(vacc0_hi, vzero_shift_mask), 31);
        vacc1_hi =
            vsraq_n_s32(vacc1_hi, vbicq_s32(vacc1_hi, vzero_shift_mask), 31);

        vacc0_lo = vrshlq_s32(vacc0_lo, vright_shift);
        vacc1_lo = vrshlq_s32(vacc1_lo, vright_shift);
        vacc0_hi = vrshlq_s32(vacc0_hi, vright_shift);
        vacc1_hi = vrshlq_s32(vacc1_hi, vright_shift);

        /* Pack, saturate, and add output zero point */
        // 将vacc0、vacc1进行饱和转换成无符号16位整数并加上输出零点，存入vy01
        const int16x8_t vacc0 = vqaddq_s16(
            vcombine_s16(vqmovn_s32(vacc0_lo), vqmovn_s32(vacc0_hi)),
            vy_zero_point);
        const int16x8_t vacc1 = vqaddq_s16(
            vcombine_s16(vqmovn_s32(vacc1_lo), vqmovn_s32(vacc1_hi)),
            vy_zero_point);

        uint8x16_t vy01 = vcombine_u8(vqmovun_s16(vacc0), vqmovun_s16(vacc1));
        vy01 = vmaxq_u8(vy01, vy_min);
        vy01 = vminq_u8(vy01, vy_max);

        // 将vy01写回到y指向的地址，递增指针y
        vst1q_u8(y, vy01);
        y += 16;
      }
#endif
      // 循环处理每个8字节的数据
      for (; n >= 8; n -= 8) {
        // 从地址a加载8字节数据到va，然后a向后移动8字节
        const uint8x8_t va = vld1_u8(a);
        a += 8;
        // 从地址b加载8字节数据到vb，然后b向后移动8字节
        const uint8x8_t vb = vld1_u8(b);
        b += 8;

        /* Subtract zero point */
        // 将va和vb各元素减去零点，并转换为16位有符号整数
        const int16x8_t vxa =
            vreinterpretq_s16_u16(vsubl_u8(va, va_zero_point));
        const int16x8_t vxb =
            vreinterpretq_s16_u16(vsubl_u8(vb, vb_zero_point));

        /* Multiply by factors and accumulate products */
        // 计算va和vb的乘积，并用乘数系数进行累加
        int32x4_t vacc_lo =
            vmulq_s32(vmovl_s16(vget_low_s16(vxa)), va_multiplier);
#ifdef __aarch64__
        int32x4_t vacc_hi = vmulq_s32(vmovl_high_s16(vxa), va_multiplier);
#else
        int32x4_t vacc_hi =
            vmulq_s32(vmovl_s16(vget_high_s16(vxa)), va_multiplier);
#endif

        vacc_lo =
            vmlaq_s32(vacc_lo, vmovl_s16(vget_low_s16(vxb)), vb_multiplier);
#ifdef __aarch64__
        vacc_hi = vmlaq_s32(vacc_hi, vmovl_high_s16(vxb), vb_multiplier);
#else
        vacc_hi =
            vmlaq_s32(vacc_hi, vmovl_s16(vget_high_s16(vxb)), vb_multiplier);
#endif

        /* Shift right and round */
        // 对结果向右移动并进行舍入操作
        vacc_lo =
            vsraq_n_s32(vacc_lo, vbicq_s32(vacc_lo, vzero_shift_mask), 31);
        vacc_hi =
            vsraq_n_s32(vacc_hi, vbicq_s32(vacc_hi, vzero_shift_mask), 31);

        vacc_lo = vrshlq_s32(vacc_lo, vright_shift);
        vacc_hi = vrshlq_s32(vacc_hi, vright_shift);

        /* Pack, saturate, and add output zero point */
#ifdef __aarch64__
        // 打包、饱和化并添加输出零点到结果向量
        const int16x8_t vacc = vqaddq_s16(
            vqmovn_high_s32(vqmovn_s32(vacc_lo), vacc_hi), vy_zero_point);
#else
        const int16x8_t vacc = vqaddq_s16(
            vcombine_s16(vqmovn_s32(vacc_lo), vqmovn_s32(vacc_hi)),
            vy_zero_point);
#endif

        uint8x8_t vy = vqmovun_s16(vacc);
        vy = vmax_u8(vy, vget_low_u8(vy_min));
        vy = vmin_u8(vy, vget_low_u8(vy_max));

        // 将结果向量vy存储到地址y，然后y向后移动8字节
        vst1_u8(y, vy);
        y += 8;
      }
      // 处理剩余不足8字节的数据
      if (n != 0) {
        // 计算要增加的偏移量
        const size_t n_increment = n - 8;
        const int64x1_t vld_shift = vmov_n_s64(8 * n_increment);
        // 加载a和b中剩余数据，并根据增量偏移量进行左移
        const uint8x8_t va = vreinterpret_u8_u64(
            vshl_u64(vreinterpret_u64_u8(vld1_u8(a + n_increment)), vld_shift));
        const uint8x8_t vb = vreinterpret_u8_u64(
            vshl_u64(vreinterpret_u64_u8(vld1_u8(b + n_increment)), vld_shift));

        /* Subtract zero point */
        // 将va和vb各元素减去零点，并转换为16位有符号整数
        const int16x8_t vxa =
            vreinterpretq_s16_u16(vsubl_u8(va, va_zero_point));
        const int16x8_t vxb =
            vreinterpretq_s16_u16(vsubl_u8(vb, vb_zero_point));

        /* Multiply by factors and accumulate products */
        // 计算va和vb的乘积，并用乘数系数进行累加
        int32x4_t vacc_lo =
            vmulq_s32(vmovl_s16(vget_low_s16(vxa)), va_multiplier);
#ifdef __aarch64__
        int32x4_t vacc_hi = vmulq_s32(vmovl_high_s16(vxa), va_multiplier);
#else
        int32x4_t vacc_hi =
            vmulq_s32(vmovl_s16(vget_high_s16(vxa)), va_multiplier);
#endif

        vacc_lo =
            vmlaq_s32(vacc_lo, vmovl_s16(vget_low_s16(vxb)), vb_multiplier);
#ifdef __aarch64__
        // 如果目标平台是 aarch64 架构，则使用高级别函数对应的操作
        vacc_hi = vmlaq_s32(vacc_hi, vmovl_high_s16(vxb), vb_multiplier);
#else
        // 否则，使用普通级别函数对应的操作
        vacc_hi =
            vmlaq_s32(vacc_hi, vmovl_s16(vget_high_s16(vxb)), vb_multiplier);
#endif

        /* Shift right and round */
        // 对低位累加器进行右移和舍入操作
        vacc_lo =
            vsraq_n_s32(vacc_lo, vbicq_s32(vacc_lo, vzero_shift_mask), 31);
        // 对高位累加器进行右移和舍入操作
        vacc_hi =
            vsraq_n_s32(vacc_hi, vbicq_s32(vacc_hi, vzero_shift_mask), 31);

        // 对低位累加器进行左移操作
        vacc_lo = vrshlq_s32(vacc_lo, vright_shift);
        // 对高位累加器进行左移操作
        vacc_hi = vrshlq_s32(vacc_hi, vright_shift);

        /* Pack, saturate, and add output zero point */
#ifdef __aarch64__
        // 如果目标平台是 aarch64 架构，则使用高级别函数对应的操作
        const int16x8_t vacc = vqaddq_s16(
            vqmovn_high_s32(vqmovn_s32(vacc_lo), vacc_hi), vy_zero_point);
#else
        // 否则，使用普通级别函数对应的操作
        const int16x8_t vacc = vqaddq_s16(
            vcombine_s16(vqmovn_s32(vacc_lo), vqmovn_s32(vacc_hi)),
            vy_zero_point);
#endif

        // 将饱和和打包后的结果转换为无符号八位整数
        uint8x8_t vy = vqmovun_s16(vacc);
        // 取结果与最小值的低位比较，并取最大值
        vy = vmax_u8(vy, vget_low_u8(vy_min));
        // 取结果与最大值的低位比较，并取最小值
        vy = vmin_u8(vy, vget_low_u8(vy_max));

        // 根据 n 的最低两位，分别进行存储操作
        if (n & 4) {
          // 将 vy 转换为无符号 32 位整数，写入 y
          vst1_lane_u32(
              __builtin_assume_aligned(y, 1), vreinterpret_u32_u8(vy), 0);
          // y 指针向后移动 4 个字节
          y += 4;
          // vy 进行 4 位字节的扩展
          vy = vext_u8(vy, vy, 4);
        }
        if (n & 2) {
          // 将 vy 转换为无符号 16 位整数，写入 y
          vst1_lane_u16(
              __builtin_assume_aligned(y, 1), vreinterpret_u16_u8(vy), 0);
          // y 指针向后移动 2 个字节
          y += 2;
          // vy 进行 2 位字节的扩展
          vy = vext_u8(vy, vy, 2);
        }
        if (n & 1) {
          // 将 vy 转换为无符号 8 位整数，写入 y
          vst1_lane_u8(y, vy, 0);
        }
      }
    }
  else {
    for (; n != 0; n--) {
      const uint8x8_t va = vld1_dup_u8(a);
      a += 1;
      const uint8x8_t vb = vld1_dup_u8(b);

      // 从 va 和 vb 中减去零点
      const int16x4_t vxa =
          vreinterpret_s16_u16(vget_low_u16(vsubl_u8(va, va_zero_point)));
      const int16x4_t vxb =
          vreinterpret_s16_u16(vget_low_u16(vsubl_u8(vb, vb_zero_point)));

      // 将 vxa 和 vxb 乘以因子并累积乘积
      int32x2_t vacc =
          vmul_s32(vget_low_s32(vmovl_s16(vxa)), vget_low_s32(va_multiplier));
      vacc = vmla_s32(
          vacc, vget_low_s32(vmovl_s16(vxb)), vget_low_s32(vb_multiplier));

      // 对累加器进行右移和舍入操作
      vacc = vsra_n_s32(vacc, vbic_s32(vacc, vget_low_s32(vzero_shift_mask)), 31);

      // 对累加器进行左移操作
      vacc = vrshl_s32(vacc, vget_low_s32(vright_shift));

      // 将结果添加零点并进行饱和和打包
      const int16x4_t vacc16 = vqadd_s16(
          vqmovn_s32(vcombine_s32(vacc, vacc)), vget_low_s16(vy_zero_point));

      // 将饱和和打包后的结果转换为无符号八位整数
      uint8x8_t vy = vqmovun_s16(vcombine_s16(vacc16, vacc16));

      // 取结果与最大值的低位比较，并取最小值
      vy = vmin_u8(vy, vget_low_u8(vy_max));
      // 取结果与最小值的低位比较，并取最大值
      vy = vmax_u8(vy, vget_low_u8(vy_min));

      // 将结果写入 y，并将 y 指针向后移动
      vst1_lane_u8(y, vy, 0);
      y += 1;
    }
  }
}
```