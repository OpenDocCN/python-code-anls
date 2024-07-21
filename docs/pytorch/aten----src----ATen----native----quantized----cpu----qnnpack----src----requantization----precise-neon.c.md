# `.\pytorch\aten\src\ATen\native\quantized\cpu\qnnpack\src\requantization\precise-neon.c`

```py
/*
 * 版权所有（c）Facebook公司及其附属公司。
 * 保留所有权利。
 *
 * 此源代码受BSD风格许可证的保护，许可证详见
 * 根目录下的LICENSE文件。
 */

#include <assert.h>
#include <stdint.h>

#include <arm_neon.h> // 包含ARM NEON指令集的头文件

#include <fp16/bitcasts.h> // 包含FP16格式转换的头文件
#include <qnnpack/requantization-stubs.h> // 包含重新量化（requantization）存根的头文件

void pytorch_qnnp_requantize_precise__neon(
    size_t n, // 输入数据的长度，必须是16的倍数
    const int32_t* input, // 输入的int32数组指针
    float scale, // 缩放因子
    uint8_t zero_point, // 零点
    uint8_t qmin, // 最小量化值
    uint8_t qmax, // 最大量化值
    uint8_t* output) { // 输出的uint8数组指针

  assert(n % 16 == 0); // 断言输入长度是16的倍数
  assert(scale < 1.0f); // 断言缩放因子小于1.0
  assert(scale >= 0x1.0p-32f); // 断言缩放因子大于等于2^-32

  // 将浮点数缩放因子转换成32位整数，并设置为带有偏置的乘法因子和右移量
  const uint32_t scale_bits = fp32_to_bits(scale);
  const int32_t multiplier =
      ((int32_t)scale_bits & INT32_C(0x007FFFFF)) | INT32_C(0x00800000);
  const int32_t shift = 127 + 23 - (scale_bits >> 23);
  assert(shift >= 24); // 断言右移量大于等于24
  assert(shift < 56); // 断言右移量小于56

#if defined(__aarch64__)
  // 使用NEON指令集在ARM64平台上复制乘法因子到向量寄存器
  const int32x4_t vmultiplier = vdupq_n_s32(multiplier);
#else
  // 在ARM32平台上复制乘法因子到向量寄存器
  const int32x2_t vmultiplier = vdup_n_s32(multiplier);
#endif
  // 复制零点值到向量寄存器
  const int16x8_t vzero_point = vdupq_n_s16((int16_t)(uint16_t)zero_point);
  // 复制右移量到向量寄存器
  const int64x2_t vshift = vdupq_n_s64(-shift);
  // 复制最小量化值到向量寄存器
  const uint8x16_t vqmin = vdupq_n_u8(qmin);
  // 复制最大量化值到向量寄存器
  const uint8x16_t vqmax = vdupq_n_u8(qmax);

  // 循环处理每16个元素的输入数据
  for (; n != 0; n -= 16) {
    // 加载四组int32输入数据到NEON向量寄存器
    const int32x4_t x = vld1q_s32(input);
    const int32x4_t y = vld1q_s32(input + 4);
    const int32x4_t z = vld1q_s32(input + 8);
    const int32x4_t w = vld1q_s32(input + 12);
    input += 16;

    // 根据输入数据的符号生成掩码
    const uint32x4_t x_neg_mask = vcltq_s32(x, vmovq_n_s32(0));
    const uint32x4_t y_neg_mask = vcltq_s32(y, vmovq_n_s32(0));
    const uint32x4_t z_neg_mask = vcltq_s32(z, vmovq_n_s32(0));
    const uint32x4_t w_neg_mask = vcltq_s32(w, vmovq_n_s32(0));

#if defined(__aarch64__)
    // 在ARM64平台上执行高效的64位乘法操作
    const int64x2_t x01_product =
        vmull_s32(vget_low_s32(x), vget_low_s32(vmultiplier));
    const int64x2_t x23_product = vmull_high_s32(x, vmultiplier);
    const int64x2_t y01_product =
        vmull_s32(vget_low_s32(y), vget_low_s32(vmultiplier));
    const int64x2_t y23_product = vmull_high_s32(y, vmultiplier);
    const int64x2_t z01_product =
        vmull_s32(vget_low_s32(z), vget_low_s32(vmultiplier));
    const int64x2_t z23_product = vmull_high_s32(z, vmultiplier);
    const int64x2_t w01_product =
        vmull_s32(vget_low_s32(w), vget_low_s32(vmultiplier));
    const int64x2_t w23_product = vmull_high_s32(w, vmultiplier);
#else
    // 在ARM32平台上执行32位乘法操作
    const int64x2_t x01_product = vmull_s32(vget_low_s32(x), vmultiplier);
    const int64x2_t x23_product = vmull_s32(vget_high_s32(x), vmultiplier);
    const int64x2_t y01_product = vmull_s32(vget_low_s32(y), vmultiplier);
    const int64x2_t y23_product = vmull_s32(vget_high_s32(y), vmultiplier);
    const int64x2_t z01_product = vmull_s32(vget_low_s32(z), vmultiplier);
    const int64x2_t z23_product = vmull_s32(vget_high_s32(z), vmultiplier);
    const int64x2_t w01_product = vmull_s32(vget_low_s32(w), vmultiplier);

        const int64x2_t w23_product = vmull_s32(vget_high_s32(w), vmultiplier);
#endif
    // 继续处理每组数据的量化和饱和操作
    # 使用 NEON 指令 vmull_s32 对两个 int32x2_t 类型的向量进行有符号整数乘法运算，得到一个 int64x2_t 类型的结果向量。
    const int64x2_t w23_product = vmull_s32(vget_high_s32(w), vmultiplier);
#ifdef __aarch64__
    // 如果目标平台是 aarch64 架构，则进行如下操作：

    // 计算 x01_product 和 x_neg_mask 的和，并将结果调整为 int64x2_t 类型
    const int64x2_t x01_adjusted_product =
        vaddw_s32(x01_product, vreinterpret_s32_u32(vget_low_u32(x_neg_mask)));
    // 计算 x23_product 和 x_neg_mask 的和，并将结果调整为 int64x2_t 类型
    const int64x2_t x23_adjusted_product =
        vaddw_high_s32(x23_product, vreinterpretq_s32_u32(x_neg_mask));
    // 计算 y01_product 和 y_neg_mask 的和，并将结果调整为 int64x2_t 类型
    const int64x2_t y01_adjusted_product =
        vaddw_s32(y01_product, vreinterpret_s32_u32(vget_low_u32(y_neg_mask)));
    // 计算 y23_product 和 y_neg_mask 的和，并将结果调整为 int64x2_t 类型
    const int64x2_t y23_adjusted_product =
        vaddw_high_s32(y23_product, vreinterpretq_s32_u32(y_neg_mask));
    // 计算 z01_product 和 z_neg_mask 的和，并将结果调整为 int64x2_t 类型
    const int64x2_t z01_adjusted_product =
        vaddw_s32(z01_product, vreinterpret_s32_u32(vget_low_u32(z_neg_mask)));
    // 计算 z23_product 和 z_neg_mask 的和，并将结果调整为 int64x2_t 类型
    const int64x2_t z23_adjusted_product =
        vaddw_high_s32(z23_product, vreinterpretq_s32_u32(z_neg_mask));
    // 计算 w01_product 和 w_neg_mask 的和，并将结果调整为 int64x2_t 类型
    const int64x2_t w01_adjusted_product =
        vaddw_s32(w01_product, vreinterpret_s32_u32(vget_low_u32(w_neg_mask)));
    // 计算 w23_product 和 w_neg_mask 的和，并将结果调整为 int64x2_t 类型
    const int64x2_t w23_adjusted_product =
        vaddw_high_s32(w23_product, vreinterpretq_s32_u32(w_neg_mask));
#else
    // 如果目标平台不是 aarch64 架构，则进行如下操作：

    // 计算 x01_product 和 x_neg_mask 的和，并将结果调整为 int64x2_t 类型
    const int64x2_t x01_adjusted_product =
        vaddw_s32(x01_product, vreinterpret_s32_u32(vget_low_u32(x_neg_mask)));
    // 计算 x23_product 和 x_neg_mask 的和，并将结果调整为 int64x2_t 类型
    const int64x2_t x23_adjusted_product =
        vaddw_s32(x23_product, vreinterpret_s32_u32(vget_high_u32(x_neg_mask)));
    // 计算 y01_product 和 y_neg_mask 的和，并将结果调整为 int64x2_t 类型
    const int64x2_t y01_adjusted_product =
        vaddw_s32(y01_product, vreinterpret_s32_u32(vget_low_u32(y_neg_mask)));
    // 计算 y23_product 和 y_neg_mask 的和，并将结果调整为 int64x2_t 类型
    const int64x2_t y23_adjusted_product =
        vaddw_s32(y23_product, vreinterpret_s32_u32(vget_high_u32(y_neg_mask)));
    // 计算 z01_product 和 z_neg_mask 的和，并将结果调整为 int64x2_t 类型
    const int64x2_t z01_adjusted_product =
        vaddw_s32(z01_product, vreinterpret_s32_u32(vget_low_u32(z_neg_mask)));
    // 计算 z23_product 和 z_neg_mask 的和，并将结果调整为 int64x2_t 类型
    const int64x2_t z23_adjusted_product =
        vaddw_s32(z23_product, vreinterpret_s32_u32(vget_high_u32(z_neg_mask)));
    // 计算 w01_product 和 w_neg_mask 的和，并将结果调整为 int64x2_t 类型
    const int64x2_t w01_adjusted_product =
        vaddw_s32(w01_product, vreinterpret_s32_u32(vget_low_u32(w_neg_mask)));
    // 计算 w23_product 和 w_neg_mask 的和，并将结果调整为 int64x2_t 类型
    const int64x2_t w23_adjusted_product =
        vaddw_s32(w23_product, vreinterpret_s32_u32(vget_high_u32(w_neg_mask)));
#endif

// 将调整后的产品向右移位，得到缩放后的结果
const int64x2_t x01_scaled = vrshlq_s64(x01_adjusted_product, vshift);
const int64x2_t x23_scaled = vrshlq_s64(x23_adjusted_product, vshift);
const int64x2_t y01_scaled = vrshlq_s64(y01_adjusted_product, vshift);
const int64x2_t y23_scaled = vrshlq_s64(y23_adjusted_product, vshift);
const int64x2_t z01_scaled = vrshlq_s64(z01_adjusted_product, vshift);
const int64x2_t z23_scaled = vrshlq_s64(z23_adjusted_product, vshift);
const int64x2_t w01_scaled = vrshlq_s64(w01_adjusted_product, vshift);
const int64x2_t w23_scaled = vrshlq_s64(w23_adjusted_product, vshift);

#ifdef __aarch64__
    // 如果目标平台是 aarch64 架构，则进行如下操作：

    // 按照顺序重新排列 x01_scaled 和 x23_scaled 中的元素，得到 x_scaled
    const int32x4_t x_scaled = vuzp1q_s32(
        vreinterpretq_s32_s64(x01_scaled), vreinterpretq_s32_s64(x23_scaled));
    // 按照顺序重新排列 y01_scaled 和 y23_scaled 中的元素，得到 y_scaled
    const int32x4_t y_scaled = vuzp1q_s32(
        vreinterpretq_s32_s64(y01_scaled), vreinterpretq_s32_s64(y23_scaled));
    // 按照顺序重新排列 z01_scaled 和 z23_scaled 中的元素，得到 z_scaled
    const int32x4_t z_scaled = vuzp1q_s32(
        vreinterpretq_s32_s64(z01_scaled), vreinterpretq_s32_s64(z23_scaled));
    # 将两个 64 位有符号整数向量解压缩成一个 128 位整数向量，取第一个部分
    const int32x4_t w_scaled = vuzp1q_s32(
        vreinterpretq_s32_s64(w01_scaled), vreinterpretq_s32_s64(w23_scaled));

    # 将两个 32 位整数向量分别移动至高位和低位，然后转换为 16 位整数向量，
    # 再与零点偏移量相加，得到 xy_packed 向量
    const int16x8_t xy_packed = vqaddq_s16(
        vqmovn_high_s32(vqmovn_s32(x_scaled), y_scaled), vzero_point);

    # 将两个 32 位整数向量分别移动至高位和低位，然后转换为 16 位整数向量，
    # 再与零点偏移量相加，得到 zw_packed 向量
    const int16x8_t zw_packed = vqaddq_s16(
        vqmovn_high_s32(vqmovn_s32(z_scaled), w_scaled), vzero_point);

    # 将两个 16 位整数向量分别移动至高位和低位，然后转换为 8 位无符号整数向量，
    # 合并成一个 128 位向量 xyzw_packed
    const uint8x16_t xyzw_packed =
        vqmovun_high_s16(vqmovun_s16(xy_packed), zw_packed);
#else
    // 创建 x_scaled，将 x01_scaled 和 x23_scaled 合并为一个 int32x4_t 类型的向量
    const int32x4_t x_scaled =
        vcombine_s32(vmovn_s64(x01_scaled), vmovn_s64(x23_scaled));
    // 创建 y_scaled，将 y01_scaled 和 y23_scaled 合并为一个 int32x4_t 类型的向量
    const int32x4_t y_scaled =
        vcombine_s32(vmovn_s64(y01_scaled), vmovn_s64(y23_scaled));
    // 创建 z_scaled，将 z01_scaled 和 z23_scaled 合并为一个 int32x4_t 类型的向量
    const int32x4_t z_scaled =
        vcombine_s32(vmovn_s64(z01_scaled), vmovn_s64(z23_scaled));
    // 创建 w_scaled，将 w01_scaled 和 w23_scaled 合并为一个 int32x4_t 类型的向量
    const int32x4_t w_scaled =
        vcombine_s32(vmovn_s64(w01_scaled), vmovn_s64(w23_scaled));

    // 将 x_scaled 和 y_scaled 合并为一个 int16x8_t 类型的向量，然后与 vzero_point 相加
    const int16x8_t xy_packed = vqaddq_s16(
        vcombine_s16(vqmovn_s32(x_scaled), vqmovn_s32(y_scaled)), vzero_point);
    // 将 z_scaled 和 w_scaled 合并为一个 int16x8_t 类型的向量，然后与 vzero_point 相加
    const int16x8_t zw_packed = vqaddq_s16(
        vcombine_s16(vqmovn_s32(z_scaled), vqmovn_s32(w_scaled)), vzero_point);
    // 将 xy_packed 和 zw_packed 合并为一个 uint8x16_t 类型的向量
    const uint8x16_t xyzw_packed =
        vcombine_u8(vqmovun_s16(xy_packed), vqmovun_s16(zw_packed));
#endif

    // 将 xyzw_packed 向量中的值限制在 vqmin 和 vqmax 之间，并返回结果
    const uint8x16_t xyzw_clamped =
        vmaxq_u8(vminq_u8(xyzw_packed, vqmax), vqmin);

    /*
     * AArch32 version:
     *   4x VCLT.S32 Qd, Qm, #0
     *   8x VMULL.S32 Qd, Dm, Dn
     *   8x VADDW.S32 Qd, Qm, Dn
     *   8x VRSHL.S32 Qd, Qm, Qn
     *   8x VMOVN.S64 Dd, Qm
     *   4x VQMOVN.S32 Dd, Qm
     *   2x VADD.S16 Qd, Qm, Qn
     *   2x VQMOVUN.S16 Dd, Qm
     *   1x VMAX.U8 Qd, Qm, Qn
     *   1x VMIN.U8 Qd, Qm, Qn
     * ---------------------
     * 46 instructions total
     *
     * AArch64 version:
     *   4x CMLT Vd.4S, Vn.4S, #0
     *   4x SMULL Vd.2D, Vn.2S, Vm.2S
     *   4x SMULL2 Vd.2D, Vn.4S, Vm.4S
     *   4x SADDW Vd.2D, Vn.2D, Vm.2S
     *   4x SADDW2 Vd.2D, Vn.2D, Vm.4S
     *   8x SRSHL Vd.2D, Vn.2D, Vm.2D
     *   4x UZP1 Vd.4S, Vn.4S, Vm.4S
     *   2x SQXTN Vd.4H, Vn.4S
     *   2x SQXTN2 Vd.8H, Vn.4S
     *   2x ADD Vd.8H, Vn.8H, Vm.8H
     *   1x SQXTUN Vd.8B, Vn.8H
     *   1x SQXTUN2 Vd.16B, Vn.8H
     *   1x UMIN Vd.16B, Vn.16B, Vm.16B
     *   1x UMAX Vd.16B, Vn.16B, Vm.16B
     * ---------------------
     * 42 instructions total
     */

    // 将 xyzw_clamped 向量中的值存储到 output 指针指向的内存地址
    vst1q_u8(output, xyzw_clamped);
    // 将 output 指针移动到下一个 16 字节的内存地址
    output += 16;
  }
}
```