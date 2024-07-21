# `.\pytorch\aten\src\ATen\native\quantized\cpu\qnnpack\src\q8conv\8x8-neon.c`

```
/*
 * 从这里开始是一个用 NEON 指令集优化的卷积运算的内核函数
 */

#include <arm_neon.h>  // 包含 ARM NEON 指令集的头文件

#include <qnnpack/q8conv.h>  // 包含 QNNPACK 提供的 Q8 卷积函数头文件
#include <requantization/runtime-neon.h>  // 包含 NEON 下的重新量化函数头文件

void pytorch_q8conv_ukernel_8x8__neon(
    size_t mr,
    size_t nr,
    size_t kc,
    size_t ks,
    const uint8_t** restrict a,  // 输入数据矩阵的指针数组
    const void* restrict w,  // 卷积核参数的指针
    uint8_t* restrict c,  // 输出数据矩阵的指针
    size_t c_stride,
    size_t output_channel_index,
    const union pytorch_qnnp_conv_quantization_params
        quantization_params[restrict static 1]) {  // 量化参数结构体数组的指针
  const uint8x8_t va_zero_point =  // 加载输入数据的零点值
      vld1_dup_u8((const uint8_t*)&quantization_params->neon.input_zero_point);
  const uint8x8_t vb_zero_point =  // 加载卷积核的零点值
      vld1_u8((const uint8_t*)&quantization_params->neon.kernel_zero_points
          [output_channel_index]);

  // 加载累加器的初始值
  int32x4_t vacc0x0123 = vld1q_s32(w);
  w = (void*)((uintptr_t)w + sizeof(int32x4_t));
  int32x4_t vacc0x4567 = vld1q_s32(w);
  w = (void*)((uintptr_t)w + sizeof(int32x4_t));
  int32x4_t vacc1x0123 = vacc0x0123;
  int32x4_t vacc1x4567 = vacc0x4567;
  int32x4_t vacc2x0123 = vacc0x0123;
  int32x4_t vacc2x4567 = vacc0x4567;
  int32x4_t vacc3x0123 = vacc0x0123;
  int32x4_t vacc3x4567 = vacc0x4567;
  int32x4_t vacc4x0123 = vacc0x0123;
  int32x4_t vacc4x4567 = vacc0x4567;
  int32x4_t vacc5x0123 = vacc0x0123;
  int32x4_t vacc5x4567 = vacc0x4567;
  int32x4_t vacc6x0123 = vacc0x0123;
  int32x4_t vacc6x4567 = vacc0x4567;
  int32x4_t vacc7x0123 = vacc0x0123;
  int32x4_t vacc7x4567 = vacc0x4567;

  do {
    // 逐行加载输入数据
    const uint8_t* restrict a0 = *a++;
    const uint8_t* restrict a1 = *a++;
    const uint8_t* restrict a2 = *a++;
    const uint8_t* restrict a3 = *a++;
    const uint8_t* restrict a4 = *a++;
    const uint8_t* restrict a5 = *a++;
    const uint8_t* restrict a6 = *a++;
    const uint8_t* restrict a7 = *a++;

    size_t k = kc;
    // 执行卷积计算
    // (此处应该有具体的卷积计算步骤，但因为缺少代码，无法提供详细的注释)

  } while (--ks != 0);

  // 加载重新量化的缩放因子
  const float32x4_t requantization_scale_c0123 =
      vld1q_f32(
          &quantization_params->neon.requantization_scales[output_channel_index]);
  const float32x4_t requantization_scale_c4567 =
      vld1q_f32(
          &quantization_params->neon.requantization_scales[
              output_channel_index + 4]);

  // 对累加器进行重新量化和缩放
  const float32x4_t vacc0x0123_f =
    vmulq_f32(vcvtq_f32_s32(vacc0x0123), requantization_scale_c0123);
  const float32x4_t vacc1x0123_f =
    vmulq_f32(vcvtq_f32_s32(vacc1x0123), requantization_scale_c0123);
  const float32x4_t vacc2x0123_f =
    vmulq_f32(vcvtq_f32_s32(vacc2x0123), requantization_scale_c0123);
  const float32x4_t vacc3x0123_f =
    vmulq_f32(vcvtq_f32_s32(vacc3x0123), requantization_scale_c0123);
  const float32x4_t vacc0x4567_f =
    vmulq_f32(vcvtq_f32_s32(vacc0x4567), requantization_scale_c4567);
  const float32x4_t vacc1x4567_f =
    vmulq_f32(vcvtq_f32_s32(vacc1x4567), requantization_scale_c4567);
  const float32x4_t vacc2x4567_f =
    vmulq_f32(vcvtq_f32_s32(vacc2x4567), requantization_scale_c4567);


这段代码是一个使用 NEON 指令集进行优化的卷积运算的内核函数。
  # 将整型向量 vacc2x4567 转换为单精度浮点向量，并乘以 requantization_scale_c4567
  const float32x4_t vacc2x4567_f =
    vmulq_f32(vcvtq_f32_s32(vacc2x4567), requantization_scale_c4567);
  # 将整型向量 vacc3x4567 转换为单精度浮点向量，并乘以 requantization_scale_c4567
  const float32x4_t vacc3x4567_f =
    vmulq_f32(vcvtq_f32_s32(vacc3x4567), requantization_scale_c4567);
  # 将整型向量 vacc4x0123 转换为单精度浮点向量，并乘以 requantization_scale_c0123
  const float32x4_t vacc4x0123_f =
    vmulq_f32(vcvtq_f32_s32(vacc4x0123), requantization_scale_c0123);
  # 将整型向量 vacc5x0123 转换为单精度浮点向量，并乘以 requantization_scale_c0123
  const float32x4_t vacc5x0123_f =
    vmulq_f32(vcvtq_f32_s32(vacc5x0123), requantization_scale_c0123);
  # 将整型向量 vacc6x0123 转换为单精度浮点向量，并乘以 requantization_scale_c0123
  const float32x4_t vacc6x0123_f =
    vmulq_f32(vcvtq_f32_s32(vacc6x0123), requantization_scale_c0123);
  # 将整型向量 vacc7x0123 转换为单精度浮点向量，并乘以 requantization_scale_c0123
  const float32x4_t vacc7x0123_f =
    vmulq_f32(vcvtq_f32_s32(vacc7x0123), requantization_scale_c0123);
  # 将整型向量 vacc4x4567 转换为单精度浮点向量，并乘以 requantization_scale_c4567
  const float32x4_t vacc4x4567_f =
    vmulq_f32(vcvtq_f32_s32(vacc4x4567), requantization_scale_c4567);
  # 将整型向量 vacc5x4567 转换为单精度浮点向量，并乘以 requantization_scale_c4567
  const float32x4_t vacc5x4567_f =
    vmulq_f32(vcvtq_f32_s32(vacc5x4567), requantization_scale_c4567);
  # 将整型向量 vacc6x4567 转换为单精度浮点向量，并乘以 requantization_scale_c4567
  const float32x4_t vacc6x4567_f =
    vmulq_f32(vcvtq_f32_s32(vacc6x4567), requantization_scale_c4567);
  # 将整型向量 vacc7x4567 转换为单精度浮点向量，并乘以 requantization_scale_c4567
  const float32x4_t vacc7x4567_f =
    vmulq_f32(vcvtq_f32_s32(vacc7x4567), requantization_scale_c4567);
#endif

// 保存原始指针 c 的值
uint8_t* c0 = c;
// 计算下一个指针 c1，位移量为 c_stride
uint8_t* c1 = (uint8_t*)((uintptr_t)c0 + c_stride);
// 如果 mr 小于 2，则将 c1 设为 c0，即与 c0 相同
if (mr < 2) {
  c1 = c0;
}
// 计算下一个指针 c2
uint8_t* c2 = (uint8_t*)((uintptr_t)c1 + c_stride);
// 如果 mr 小于等于 2，则将 c2 设为 c1，即与 c1 相同
if (mr <= 2) {
  c2 = c1;
}
// 计算下一个指针 c3
uint8_t* c3 = (uint8_t*)((uintptr_t)c2 + c_stride);
// 如果 mr 小于 4，则将 c3 设为 c2，即与 c2 相同
if (mr < 4) {
  c3 = c2;
}
// 计算下一个指针 c4
uint8_t* c4 = (uint8_t*)((uintptr_t)c3 + c_stride);
// 如果 mr 小于等于 4，则将 c4 设为 c3，即与 c3 相同
if (mr <= 4) {
  c4 = c3;
}
// 计算下一个指针 c5
uint8_t* c5 = (uint8_t*)((uintptr_t)c4 + c_stride);
// 如果 mr 小于 6，则将 c5 设为 c4，即与 c4 相同
if (mr < 6) {
  c5 = c4;
}
// 计算下一个指针 c6
uint8_t* c6 = (uint8_t*)((uintptr_t)c5 + c_stride);
// 如果 mr 小于等于 6，则将 c6 设为 c5，即与 c5 相同
if (mr <= 6) {
  c6 = c5;
}
// 计算下一个指针 c7
uint8_t* c7 = (uint8_t*)((uintptr_t)c6 + c_stride);
// 如果 mr 不等于 8，则将 c7 设为 c6，即与 c6 相同
if (mr != 8) {
  c7 = c6;
}

// 如果 nr 等于 8，依次将向量数据存储到指针所指向的内存位置
if (nr == 8) {
  vst1_u8(c0, vget_low_u8(vout0x01234567_1x01234567));
  vst1_u8(c1, vget_high_u8(vout0x01234567_1x01234567));
  vst1_u8(c2, vget_low_u8(vout2x01234567_3x01234567));
  vst1_u8(c3, vget_high_u8(vout2x01234567_3x01234567));
  vst1_u8(c4, vget_low_u8(vout4x01234567_5x01234567));
  vst1_u8(c5, vget_high_u8(vout4x01234567_5x01234567));
  vst1_u8(c6, vget_low_u8(vout6x01234567_7x01234567));
  vst1_u8(c7, vget_high_u8(vout6x01234567_7x01234567));
} else {
  // 如果 nr 大于等于 4，则将每个向量中的两个 32 位数据存储到对应的指针位置
  vst1q_lane_u32(
      __builtin_assume_aligned(c0, 1),
      vreinterpretq_u32_u8(vout0x01234567_1x01234567),
      0);
  c0 += 4;
  vst1q_lane_u32(
      __builtin_assume_aligned(c1, 1),
      vreinterpretq_u32_u8(vout0x01234567_1x01234567),
      2);
  c1 += 4;
  vst1q_lane_u32(
      __builtin_assume_aligned(c2, 1),
      vreinterpretq_u32_u8(vout2x01234567_3x01234567),
      0);
  c2 += 4;
  vst1q_lane_u32(
      __builtin_assume_aligned(c3, 1),
      vreinterpretq_u32_u8(vout2x01234567_3x01234567),
      2);
  c3 += 4;
  vst1q_lane_u32(
      __builtin_assume_aligned(c4, 1),
      vreinterpretq_u32_u8(vout4x01234567_5x01234567),
      0);
  c4 += 4;
  vst1q_lane_u32(
      __builtin_assume_aligned(c5, 1),
      vreinterpretq_u32_u8(vout4x01234567_5x01234567),
      2);
  c5 += 4;
  vst1q_lane_u32(
      __builtin_assume_aligned(c6, 1),
      vreinterpretq_u32_u8(vout6x01234567_7x01234567),
      0);
  c6 += 4;
  vst1q_lane_u32(
      __builtin_assume_aligned(c7, 1),
      vreinterpretq_u32_u8(vout6x01234567_7x01234567),
      2);
  c7 += 4;

  // 更新向量数据，移动每个向量中的数据到左边 4 个字节
  vout0x01234567_1x01234567 =
      vextq_u8(vout0x01234567_1x01234567, vout0x01234567_1x01234567, 4);
  vout2x01234567_3x01234567 =
      vextq_u8(vout2x01234567_3x01234567, vout2x01234567_3x01234567, 4);
  vout4x01234567_5x01234567 =
      vextq_u8(vout4x01234567_5x01234567, vout4x01234567_5x01234567, 4);
  vout6x01234567_7x01234567 =
      vextq_u8(vout6x01234567_7x01234567, vout6x01234567_7x01234567, 4);
  // 减少 nr 计数
  nr -= 4;
}
    if (nr >= 2) {
      // 如果剩余处理的数量大于等于2
      vst1q_lane_u16(
          __builtin_assume_aligned(c0, 1),
          vreinterpretq_u16_u8(vout0x01234567_1x01234567),
          0);
      // 将 vout0x01234567_1x01234567 的低位16位按顺序写入 c0 所指向的内存地址
      c0 += 2;
      // c0 指针向后移动两个元素（16位每个元素）
      vst1q_lane_u16(
          __builtin_assume_aligned(c1, 1),
          vreinterpretq_u16_u8(vout0x01234567_1x01234567),
          4);
      // 将 vout0x01234567_1x01234567 的高位16位按顺序写入 c1 所指向的内存地址
      c1 += 2;
      // c1 指针向后移动两个元素（16位每个元素）
      vst1q_lane_u16(
          __builtin_assume_aligned(c2, 1),
          vreinterpretq_u16_u8(vout2x01234567_3x01234567),
          0);
      // 将 vout2x01234567_3x01234567 的低位16位按顺序写入 c2 所指向的内存地址
      c2 += 2;
      // c2 指针向后移动两个元素（16位每个元素）
      vst1q_lane_u16(
          __builtin_assume_aligned(c3, 1),
          vreinterpretq_u16_u8(vout2x01234567_3x01234567),
          4);
      // 将 vout2x01234567_3x01234567 的高位16位按顺序写入 c3 所指向的内存地址
      c3 += 2;
      // c3 指针向后移动两个元素（16位每个元素）
      vst1q_lane_u16(
          __builtin_assume_aligned(c4, 1),
          vreinterpretq_u16_u8(vout4x01234567_5x01234567),
          0);
      // 将 vout4x01234567_5x01234567 的低位16位按顺序写入 c4 所指向的内存地址
      c4 += 2;
      // c4 指针向后移动两个元素（16位每个元素）
      vst1q_lane_u16(
          __builtin_assume_aligned(c5, 1),
          vreinterpretq_u16_u8(vout4x01234567_5x01234567),
          4);
      // 将 vout4x01234567_5x01234567 的高位16位按顺序写入 c5 所指向的内存地址
      c5 += 2;
      // c5 指针向后移动两个元素（16位每个元素）
      vst1q_lane_u16(
          __builtin_assume_aligned(c6, 1),
          vreinterpretq_u16_u8(vout6x01234567_7x01234567),
          0);
      // 将 vout6x01234567_7x01234567 的低位16位按顺序写入 c6 所指向的内存地址
      c6 += 2;
      // c6 指针向后移动两个元素（16位每个元素）
      vst1q_lane_u16(
          __builtin_assume_aligned(c7, 1),
          vreinterpretq_u16_u8(vout6x01234567_7x01234567),
          4);
      // 将 vout6x01234567_7x01234567 的高位16位按顺序写入 c7 所指向的内存地址
      c7 += 2;
      // c7 指针向后移动两个元素（16位每个元素）
      vout0x01234567_1x01234567 =
          vextq_u8(vout0x01234567_1x01234567, vout0x01234567_1x01234567, 2);
      // 将 vout0x01234567_1x01234567 向左循环移动两个字节（8位）的数据
      vout2x01234567_3x01234567 =
          vextq_u8(vout2x01234567_3x01234567, vout2x01234567_3x01234567, 2);
      // 将 vout2x01234567_3x01234567 向左循环移动两个字节（8位）的数据
      vout4x01234567_5x01234567 =
          vextq_u8(vout4x01234567_5x01234567, vout4x01234567_5x01234567, 2);
      // 将 vout4x01234567_5x01234567 向左循环移动两个字节（8位）的数据
      vout6x01234567_7x01234567 =
          vextq_u8(vout6x01234567_7x01234567, vout6x01234567_7x01234567, 2);
      // 将 vout6x01234567_7x01234567 向左循环移动两个字节（8位）的数据
      nr -= 2;
      // 减少处理数量计数器的值
    }
    if (nr != 0) {
      // 如果剩余处理数量不为0
      vst1q_lane_u8(c0, vout0x01234567_1x01234567, 0);
      // 将 vout0x01234567_1x01234567 的第一个8位数据写入 c0 所指向的内存地址
      vst1q_lane_u8(c1, vout0x01234567_1x01234567, 8);
      // 将 vout0x01234567_1x01234567 的第九个8位数据写入 c1 所指向的内存地址
      vst1q_lane_u8(c2, vout2x01234567_3x01234567, 0);
      // 将 vout2x01234567_3x01234567 的第一个8位数据写入 c2 所指向的内存地址
      vst1q_lane_u8(c3, vout2x01234567_3x01234567, 8);
      // 将 vout2x01234567_3x01234567 的第九个8位数据写入 c3 所指向的内存地址
      vst1q_lane_u8(c4, vout4x01234567_5x01234567, 0);
      // 将 vout4x01234567_5x01234567 的第一个8位数据写入 c4 所指向的内存地址
      vst1q_lane_u8(c5, vout4x01234567_5x01234567, 8);
      // 将 vout4x01234567_5x01234567 的第九个8位数据写入 c5 所指向的内存地址
      vst1q_lane_u8(c6, vout6x01234567_7x01234567, 0);
      // 将 vout6x01234567_7x01234567 的第一个8位数据写入 c6 所指向的内存地址
      vst1q_lane_u8(c7, vout6x01234567_7x01234567, 8);
      // 将 vout6x01234567_7x01234567 的第九个8位数据写入 c7 所指向的内存地址
    }
}



# 这行代码表示一个代码块的结束，匹配前面的一个语句或结构体的开始
```