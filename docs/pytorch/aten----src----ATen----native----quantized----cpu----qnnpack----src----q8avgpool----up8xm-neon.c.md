# `.\pytorch\aten\src\ATen\native\quantized\cpu\qnnpack\src\q8avgpool\up8xm-neon.c`

```py
/*
 * 版权声明和许可声明
 * 本源代码使用BSD风格许可证，详见根目录下的LICENSE文件
 */

#include <assert.h> // 引入断言库

#include <arm_neon.h> // 引入ARM NEON SIMD指令集

#include <qnnpack/q8avgpool.h> // 引入QNNPACK Q8平均池化函数头文件

void pytorch_q8avgpool_ukernel_up8xm__neon(
    size_t n, // n为输入数据批次大小
    size_t ks, // ks为内核尺寸
    size_t kc, // kc为通道数，需小于8
    const uint8_t** input, // 输入数据的指针数组
    const uint8_t* zero, // 零点数据的指针
    uint8_t* output, // 输出数据的指针
    size_t input_increment, // 输入数据递增量
    size_t output_increment, // 输出数据递增量
    const union pytorch_qnnp_avgpool_quantization_params
        quantization_params[restrict static 1]) { // 量化参数结构体数组

  assert(n != 0); // 断言确保批次大小不为零
  assert(ks != 0); // 断言确保内核尺寸不为零
  assert(kc < 8); // 断言确保通道数小于8

  // 加载量化参数到NEON寄存器
  const int32x4_t vbias = vld1q_dup_s32(&quantization_params->neon.bias);
  const float32x4_t vscale = vdupq_n_f32(quantization_params->neon.scale);
  const int16x8_t voutput_zero_point = vld1q_dup_s16(&quantization_params->neon.output_zero_point);
  const uint8x8_t voutput_min = vld1_dup_u8(&quantization_params->neon.output_min);
  const uint8x8_t voutput_max = vld1_dup_u8(&quantization_params->neon.output_max);

  do {
    int32x4_t vacc_lo = vbias; // 初始化低位累加器
    int32x4_t vacc_hi = vbias; // 初始化高位累加器
    const uint8_t** next_input = (const uint8_t**)((uintptr_t)input + input_increment);

    size_t m = ks;
    do {
      const uint8_t* i = *input++; // 指向当前输入数据
      i += kc; // 跳过通道数
      uint8x8_t vi = vmov_n_u8(0); // 初始化输入数据向量
      if (kc & 1) { // 如果通道数是奇数
        i -= 1; // 向前偏移一个字节
        vi = vld1_lane_u8(i, vi, 0); // 加载数据到向量中
      }
      if (kc & 2) { // 如果通道数为2
        vi = vext_u8(vi, vi, 6); // 向右扩展
        i -= 2; // 向前偏移两个字节
        vi = vreinterpret_u8_u16(vld1_lane_u16(__builtin_assume_aligned(i, 1), vreinterpret_u16_u8(vi), 0)); // 加载数据到向量中
      }
      if (kc & 4) { // 如果通道数为4
        vi = vext_u8(vi, vi, 4); // 向右扩展
        i -= 4; // 向前偏移四个字节
        vi = vreinterpret_u8_u32(vld1_lane_u32(__builtin_assume_aligned(i, 1), vreinterpret_u32_u8(vi), 0)); // 加载数据到向量中
      }

      const uint16x8_t vxi = vmovl_u8(vi); // 将uint8向量扩展为uint16向量
      vacc_lo = vaddw_s16(vacc_lo, vreinterpret_s16_u16(vget_low_u16(vxi))); // 低位累加
      vacc_hi = vaddw_s16(vacc_hi, vreinterpret_s16_u16(vget_high_u16(vxi))); // 高位累加
    } while (--m != 0); // 内核尺寸递减
    input = next_input; // 更新输入数据指针

    float32x4_t vacc_lo_f = vcvtq_f32_s32(vacc_lo); // 将低位累加转为浮点型
    float32x4_t vacc_hi_f = vcvtq_f32_s32(vacc_hi); // 将高位累加转为浮点型

    vacc_lo_f = vmulq_f32(vacc_lo_f, vscale); // 乘以量化比例
    vacc_hi_f = vmulq_f32(vacc_hi_f, vscale); // 乘以量化比例

#if defined(__aarch64__)
    vacc_lo = vcvtnq_s32_f32(vacc_lo_f); // 转换为整数
    vacc_hi = vcvtnq_s32_f32(vacc_hi_f); // 转换为整数
    const int16x8_t vacc = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc_lo), vacc_hi), voutput_zero_point); // 加上零点并饱和
    uint8x8_t vout = vqmovun_s16(vacc); // 转为无符号8位整数
    vout = vmax_u8(vout, voutput_min); // 取最大值
    vout = vmin_u8(vout, voutput_max); // 取最小值
#else
    const float32x4_t vfmin = vdupq_n_f32(quantization_params->neon.vfmin); // 加载量化参数的最小浮点数
    const float32x4_t vfmax = vdupq_n_f32(quantization_params->neon.vfmax); // 加载量化参数的最大浮点数
    const float32x4_t vfmagic = vdupq_n_f32(quantization_params->neon.vfmagic); // 加载量化参数的魔法数
    const int32x4_t vimagic = vdupq_n_s32(quantization_params->neon.vimagic); // 加载量化参数的魔法整数
#endif
    # 使用vmaxq_f32函数计算两个浮点向量的最大值，然后使用vminq_f32函数将结果与另一个浮点向量进行最小值比较和截断。
    vacc_lo_f = vminq_f32(vmaxq_f32(vacc_lo_f, vfmin), vfmax);
    
    # 使用vmaxq_f32函数计算两个浮点向量的最大值，然后使用vminq_f32函数将结果与另一个浮点向量进行最小值比较和截断。
    vacc_hi_f = vminq_f32(vmaxq_f32(vacc_hi_f, vfmin), vfmax);

    # 将vacc_lo_f浮点向量加上vfmagic向量，然后转换为有符号32位整数向量，最后减去vimagic向量。
    vacc_lo = vsubq_s32(
        vreinterpretq_s32_f32(vaddq_f32(vacc_lo_f, vfmagic)), vimagic);
    
    # 将vacc_hi_f浮点向量加上vfmagic向量，然后转换为有符号32位整数向量，最后减去vimagic向量。
    vacc_hi = vsubq_s32(
        vreinterpretq_s32_f32(vaddq_f32(vacc_hi_f, vfmagic)), vimagic);
    
    # 将两个有符号32位整数向量vacc_lo和vacc_hi转换为有符号16位整数向量，并合并成一个有符号16x8整数向量。
    const int16x8_t vacc =
        vcombine_s16(vqmovn_s32(vacc_lo), vqmovn_s32(vacc_hi));
    
    # 将有符号16位整数向量vacc转换为无符号8位整数向量vout。
    uint8x8_t vout = vqmovun_s16(vacc);
#endif

// 如果 kc 的最低位为1（即 kc & 1 != 0），执行以下代码块
if (kc & 4) {
  // 将 vout 中的数据转换为 uint32_t 类型并存储到 output 指向的内存地址中的第一个位置
  vst1_lane_u32(__builtin_assume_aligned(output, 1), vreinterpret_u32_u8(vout), 0);
  // output 指针向后移动4个字节
  output += 4;
  // 将 vout 向左循环移动4个字节
  vout = vext_u8(vout, vout, 4);
}
// 如果 kc 的第二低位为1（即 kc & 2 != 0），执行以下代码块
if (kc & 2) {
  // 将 vout 中的数据转换为 uint16_t 类型并存储到 output 指向的内存地址中的第一个位置
  vst1_lane_u16(__builtin_assume_aligned(output, 1), vreinterpret_u16_u8(vout), 0);
  // output 指针向后移动2个字节
  output += 2;
  // 将 vout 向左循环移动2个字节
  vout = vext_u8(vout, vout, 2);
}
// 如果 kc 的最高位为1（即 kc & 1 != 0），执行以下代码块
if (kc & 1) {
  // 将 vout 中的数据存储到 output 指向的内存地址中的第一个位置
  vst1_lane_u8(output, vout, 0);
  // output 指针向后移动1个字节
  output += 1;
}
// 将 output 指针转换为 uintptr_t 类型，加上 output_increment 的值，并重新转换为 uint8_t 指针类型
output = (uint8_t*)((uintptr_t)output + output_increment);

} while (--n != 0);
```