# `.\pytorch\aten\src\ATen\native\quantized\cpu\qnnpack\src\requantization\precise-scalar.c`

```
/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <assert.h>
#include <stdint.h>

#include <fp16/bitcasts.h>
#include <qnnpack/requantization-stubs.h>
#include <qnnpack/scalar-utils.h>

void pytorch_qnnp_requantize_precise__scalar_unsigned32(
    size_t n,
    const int32_t* input,
    float scale,
    uint8_t zero_point,
    uint8_t qmin,
    uint8_t qmax,
    uint8_t* output) {
  assert(n % 4 == 0);  // 确保输入大小是4的倍数，用于处理4个元素的批次
  assert(scale < 1.0f);  // 确保缩放因子小于1.0
  assert(scale >= 0x1.0p-32f);  // 确保缩放因子大于等于2的-32次方

  const uint32_t scale_bits = fp32_to_bits(scale);  // 将浮点数缩放因子转换为32位整数
  const uint32_t multiplier = (scale_bits << 8) | UINT32_C(0x80000000);  // 构造乘法器，包括缩放因子和标志位
  const uint32_t shift = 127 + 31 - (scale_bits >> 23);  // 计算右移位数，用于乘法运算
  assert(shift >= 32);  // 确保右移位数大于等于32
  assert(shift < 64);  // 确保右移位数小于64

  const uint64_t rounding = UINT64_C(1) << (shift - 1);  // 计算舍入常数
  const uint32_t rounding_hi = (uint32_t)(rounding >> 32);  // 舍入常数的高32位
  const uint32_t rounding_lo = (uint32_t)rounding;  // 舍入常数的低32位
  const uint32_t shift_minus_32 = shift - 32;  // 右移位数减去32
  const int32_t smin = (int32_t)(uint32_t)qmin - (int32_t)(uint32_t)zero_point;  // 计算下限
  const int32_t smax = (int32_t)(uint32_t)qmax - (int32_t)(uint32_t)zero_point;  // 计算上限
  for (; n != 0; n -= 4) {  // 遍历输入数组，以处理每4个元素一次
    const int32_t x = input[0];  // 取出输入数组的第一个元素
    const int32_t y = input[1];  // 取出输入数组的第二个元素
    const int32_t z = input[2];  // 取出输入数组的第三个元素
    const int32_t w = input[3];  // 取出输入数组的第四个元素
    input += 4;  // 指针移动到下一个批次的开始位置

    /*
     * Compute absolute value of input as unsigned 32-bit int.
     * All further computations will work with unsigned values to avoid
     * undefined behaviour on signed operations.
     */
    const uint32_t x_abs = (x >= 0) ? (uint32_t)x : -(uint32_t)x;  // 计算输入的绝对值
    const uint32_t y_abs = (y >= 0) ? (uint32_t)y : -(uint32_t)y;  // 计算输入的绝对值
    const uint32_t z_abs = (z >= 0) ? (uint32_t)z : -(uint32_t)z;  // 计算输入的绝对值
    const uint32_t w_abs = (w >= 0) ? (uint32_t)w : -(uint32_t)w;  // 计算输入的绝对值

    /* Compute full 64-bit product of 32-bit factors */
    const uint64_t x_product = (uint64_t)x_abs * (uint64_t)multiplier;  // 计算乘积
    const uint64_t y_product = (uint64_t)y_abs * (uint64_t)multiplier;  // 计算乘积
    const uint64_t z_product = (uint64_t)z_abs * (uint64_t)multiplier;  // 计算乘积
    const uint64_t w_product = (uint64_t)w_abs * (uint64_t)multiplier;  // 计算乘积
    /*
     * 将完整的 64 位乘积进行右移并进行四舍五入。
     * 四舍五入时向最接近的整数进行，如果中间值则向上舍入（与远离零相同）。
     *
     * 通常，此操作需要同时进行 64 位加法和 64 位移位，
     * 但我们使用两种技巧用 32 位操作替换 64 位操作。
     *
     * 为了避免完整的 64 位加法，我们利用了以下三个事实：
     * - 在移位之前加的 64 位四舍五入值是 2 的幂，因此只有一个位设置。
     * - 当 0x1.0p-32f <= scale < 0x1.0p-31f 时，非零位在低 32 位中，
     *   并且四舍五入值恰好为 0x80000000（2**31），因为四舍五入为 2**(scale-1) 且 scale >= 32。
     *   在这种情况下，通过溢出，四舍五入的添加只会影响到乘积的高 32 位，
     *   如果低 32 位部分的乘积等于或大于 0x80000000，则会发生溢出。
     *   我们可以将后一条件重新表述为低 32 位乘积的部分具有位 31 设置，
     *   然后溢出发生在乘积的低 32 位部分和四舍五入值的低 32 位部分都具有位 31 设置时。
     *   由于带有位 31 设置的 32 位数字在作为有符号整数解释时是负数，
     *   我们可以检查溢出条件为 (int32_t)((uint32_t)x_product & rounding_lo) < 0
     * - 当 0x1.0p-31f <= scale < 1.0f 时，非零位在四舍五入的高 32 位中。
     *   我们只需对四舍五入的高 32 位和乘积的高 32 位进行 32 位加法。
     *   此加法永远不会溢出，因为乘积 <= 0x80000000 * 0xFFFFFF00 < 2**63，
     *   而四舍五入 = 2**(scale-1) <= 2**62。
     *
     * 为了避免完整的 64 位移位，我们利用了移位 >= 32 的事实，并分两步进行：
     * - 通过移位 32，可以在 32 位系统上提取高 32 位字。
     * - 通过 (shift - 32) 的移位，可以作为加法结果的高字的 32 位移位实现。
     */
    const uint32_t x_carry_lo =
        (uint32_t)((int32_t)((uint32_t)x_product & rounding_lo) < 0);
    const uint32_t y_carry_lo =
        (uint32_t)((int32_t)((uint32_t)y_product & rounding_lo) < 0);
    const uint32_t z_carry_lo =
        (uint32_t)((int32_t)((uint32_t)z_product & rounding_lo) < 0);
    const uint32_t w_carry_lo =
        (uint32_t)((int32_t)((uint32_t)w_product & rounding_lo) < 0);

    const uint32_t x_product_hi = (uint32_t)(x_product >> 32);
    const uint32_t y_product_hi = (uint32_t)(y_product >> 32);
    const uint32_t z_product_hi = (uint32_t)(z_product >> 32);
    const uint32_t w_product_hi = (uint32_t)(w_product >> 32);

    const uint32_t x_abs_scaled =
        (uint32_t)(x_product_hi + rounding_hi + x_carry_lo) >> shift_minus_32;
    const uint32_t y_abs_scaled =
        (uint32_t)(y_product_hi + rounding_hi + y_carry_lo) >> shift_minus_32;
    /*
     * 计算缩放后的绝对值，根据右移位数进行调整
     */
    const uint32_t z_abs_scaled =
        (uint32_t)(z_product_hi + rounding_hi + z_carry_lo) >> shift_minus_32;
    const uint32_t w_abs_scaled =
        (uint32_t)(w_product_hi + rounding_hi + w_carry_lo) >> shift_minus_32;

    /*
     * 将输入的符号复制到缩放后的绝对值输入值
     */
    const int32_t x_scaled = (int32_t)(x >= 0 ? x_abs_scaled : -x_abs_scaled);
    const int32_t y_scaled = (int32_t)(y >= 0 ? y_abs_scaled : -y_abs_scaled);
    const int32_t z_scaled = (int32_t)(z >= 0 ? z_abs_scaled : -z_abs_scaled);
    const int32_t w_scaled = (int32_t)(w >= 0 ? w_abs_scaled : -w_abs_scaled);

    /*
     * 将缩放后的值夹在零点（qmin - zero point）和（qmax - zero point）之间
     */
    const int32_t x_clamped =
        x_scaled < smin ? smin : x_scaled > smax ? smax : x_scaled;
    const int32_t y_clamped =
        y_scaled < smin ? smin : y_scaled > smax ? smax : y_scaled;
    const int32_t z_clamped =
        z_scaled < smin ? smin : z_scaled > smax ? smax : z_scaled;
    const int32_t w_clamped =
        w_scaled < smin ? smin : w_scaled > smax ? smax : w_scaled;

    /*
     * 将零点加到夹紧后的值。
     * 结果保证在 [qmin, qmax] 范围内。
     *
     * 这个加法不能在夹紧之前安全地进行，因为缩放后的值在 [-2147483520, 2147483519] 范围内，
     * 所以零点的加法（最多可达 255）可能会导致有符号32位整数溢出。
     */
    const int32_t x_biased = x_clamped + zero_point;
    const int32_t y_biased = y_clamped + zero_point;
    const int32_t z_biased = z_clamped + zero_point;
    const int32_t w_biased = w_clamped + zero_point;

    /*
     * 将偏置后的值写入输出数组，每个值转换为无符号8位整数
     */
    output[0] = (uint8_t)x_biased;
    output[1] = (uint8_t)y_biased;
    output[2] = (uint8_t)z_biased;
    output[3] = (uint8_t)w_biased;
    // 指向下一个输出块的指针位置
    output += 4;
    const int32_t w_scaled = (int32_t)(w >= 0 ? w_abs_scaled : -w_abs_scaled);

    /*
     * 计算输入的绝对值作为无符号的32位整数。
     * 所有后续计算将使用无符号值，以避免在有符号操作中出现未定义行为。
     */
    const uint32_t x_abs = (x >= 0) ? (uint32_t)x : -(uint32_t)x;
    const uint32_t y_abs = (y >= 0) ? (uint32_t)y : -(uint32_t)y;
    const uint32_t z_abs = (z >= 0) ? (uint32_t)z : -(uint32_t)z;
    const uint32_t w_abs = (w >= 0) ? (uint32_t)w : -(uint32_t)w;

    /* 计算两个32位因子的完整64位乘积 */
    const uint64_t x_product = (uint64_t)x_abs * (uint64_t)multiplier;
    const uint64_t y_product = (uint64_t)y_abs * (uint64_t)multiplier;
    const uint64_t z_product = (uint64_t)z_abs * (uint64_t)multiplier;
    const uint64_t w_product = (uint64_t)w_abs * (uint64_t)multiplier;

    /*
     * 使用舍入的方式将完整的64位乘积右移。
     * 舍入向最接近的整数执行，中间值向上舍入（与远离零的舍入相同）。
     *
     * 注意，尽管舍入是预先计算的，但它取决于移位值，
     * 在具有64位“带舍入右移”的处理器上，每行以下代码可以由一条指令完成
     * （例如ARM NEON上的VRSHL.U64，ARM64 Advanced SIMD中的URSHL）。
     */
    const uint32_t x_abs_scaled = (uint32_t)((x_product + rounding) >> shift);
    const uint32_t y_abs_scaled = (uint32_t)((y_product + rounding) >> shift);
    const uint32_t z_abs_scaled = (uint32_t)((z_product + rounding) >> shift);
    const uint32_t w_abs_scaled = (uint32_t)((w_product + rounding) >> shift);

    /*
     * 将输入的符号复制到缩放后的绝对值输入值中。
     *
     * 在带有SSSE3指令集的x86处理器上，此操作很好地映射到PSIGND指令。
     */
    const int32_t x_scaled = (int32_t)(x >= 0 ? x_abs_scaled : -x_abs_scaled);
    const int32_t y_scaled = (int32_t)(y >= 0 ? y_abs_scaled : -y_abs_scaled);
    const int32_t z_scaled = (int32_t)(z >= 0 ? z_abs_scaled : -z_abs_scaled);
    const int32_t w_scaled = (int32_t)(w >= 0 ? w_abs_scaled : -w_abs_scaled);
    // 计算带符号整数 w 的缩放值，根据 w 的正负选择缩放的绝对值
    const int32_t w_scaled = (int32_t)(w >= 0 ? w_abs_scaled : -w_abs_scaled);

    /*
     * 将缩放后的值限制在 (qmin - zero point) 和 (qmax - zero point) 之间。
     * 如果 x_scaled 超出了这个范围，则使用 smin 或 smax 来代替。
     */
    const int32_t x_clamped =
        x_scaled < smin ? smin : x_scaled > smax ? smax : x_scaled;
    const int32_t y_clamped =
        y_scaled < smin ? smin : y_scaled > smax ? smax : y_scaled;
    const int32_t z_clamped =
        z_scaled < smin ? smin : z_scaled > smax ? smax : z_scaled;
    const int32_t w_clamped =
        w_scaled < smin ? smin : w_scaled > smax ? smax : w_scaled;

    /*
     * 将零点加到限制后的值上。
     * 结果保证在 [qmin, qmax] 范围内。
     *
     * 这个加法不能在限制之前安全地执行，因为缩放后的值在 [-2147483520, 2147483519] 范围内，
     * 所以零点的加法（最大可以达到 255）可能会导致有符号 32 位整数溢出。
     */
    const int32_t x_biased = x_clamped + zero_point;
    const int32_t y_biased = y_clamped + zero_point;
    const int32_t z_biased = z_clamped + zero_point;
    const int32_t w_biased = w_clamped + zero_point;

    // 将偏置后的整数值转换为无符号 8 位整数，并存储到输出数组中
    output[0] = (uint8_t)x_biased;
    output[1] = (uint8_t)y_biased;
    output[2] = (uint8_t)z_biased;
    output[3] = (uint8_t)w_biased;
    // 更新输出指针，指向下一个输出位置
    output += 4;
}
  /*
   * 确保输入元素个数为4的倍数
   */
  assert(n % 4 == 0);
  
  /*
   * 确保缩放因子 scale 小于1.0
   */
  assert(scale < 1.0f);
  
  /*
   * 确保缩放因子 scale 大于或等于 2^-32
   */
  assert(scale >= 0x1.0p-32f);

  /*
   * 将浮点数 scale 转换为整数表示的二进制，并提取出其中的低23位
   */
  const uint32_t scale_bits = fp32_to_bits(scale);
  
  /*
   * 计算乘法因子 multiplier，将 scale_bits 的低23位与 0x00800000 按位或，确保乘法因子为正数
   */
  const int32_t multiplier =
      ((int32_t)scale_bits & INT32_C(0x007FFFFF)) | INT32_C(0x00800000);
  
  /*
   * 计算右移位数 shift，确保其在范围 [24, 55] 内
   */
  const uint32_t shift = 127 + 23 - (scale_bits >> 23);
  assert(shift >= 24);
  assert(shift < 56);

  /*
   * 计算舍入值 rounding，用于后续算术右移操作，以模拟向远离零的方向舍入
   */
  const int64_t rounding = INT64_C(1) << (shift - 1);
  
  /*
   * 计算输入数据在量化空间中的最小值和最大值与零点的差
   */
  const int32_t smin = (int32_t)(uint32_t)qmin - (int32_t)(uint32_t)zero_point;
  const int32_t smax = (int32_t)(uint32_t)qmax - (int32_t)(uint32_t)zero_point;
  
  /*
   * 对每四个输入元素进行量化和反量化的精确处理
   */
  for (; n != 0; n -= 4) {
    const int32_t x = input[0];
    const int32_t y = input[1];
    const int32_t z = input[2];
    const int32_t w = input[3];
    input += 4;

    /*
     * 计算每个输入元素与乘法因子 multiplier 的完整64位乘积
     */
    const int64_t x_product = (int64_t)x * (int64_t)multiplier;
    const int64_t y_product = (int64_t)y * (int64_t)multiplier;
    const int64_t z_product = (int64_t)z * (int64_t)multiplier;
    const int64_t w_product = (int64_t)w * (int64_t)multiplier;

    /*
     * 调整乘积值，以便在进行算术右移时模拟向远离零的舍入
     */
    const int64_t x_adjusted_product = x_product - (int64_t)(x < 0);
    const int64_t y_adjusted_product = y_product - (int64_t)(y < 0);
    const int64_t z_adjusted_product = z_product - (int64_t)(z < 0);
    const int64_t w_adjusted_product = w_product - (int64_t)(w < 0);

    /*
     * 执行算术右移64位乘积，并模拟舍入
     */
    const int32_t x_scaled =
        (int32_t)asr_s64(x_adjusted_product + rounding, shift);
    const int32_t y_scaled =
        (int32_t)asr_s64(y_adjusted_product + rounding, shift);
    const int32_t z_scaled =
        (int32_t)asr_s64(z_adjusted_product + rounding, shift);
    const int32_t w_scaled =
        (int32_t)asr_s64(w_adjusted_product + rounding, shift);

    /*
     * 将量化后的值限制在指定的范围内，确保不超过量化空间的边界
     */
    const int32_t x_clamped =
        x_scaled < smin ? smin : x_scaled > smax ? smax : x_scaled;
    const int32_t y_clamped =
        y_scaled < smin ? smin : y_scaled > smax ? smax : y_scaled;
    const int32_t z_clamped =
        z_scaled < smin ? smin : z_scaled > smax ? smax : z_scaled;
    // 对输入的 w_scaled 进行范围限制，确保其在 [smin, smax] 范围内
    const int32_t w_clamped =
        w_scaled < smin ? smin : w_scaled > smax ? smax : w_scaled;

    /*
     * 将偏置值 zero_point 添加到经过范围限制的值上。
     * 结果保证在 [qmin, qmax] 范围内。
     *
     * 这个加法操作不能在范围限制之前进行，因为经过缩放的值处于 [-2147483520, 2147483519] 范围内，
     * 因此将 zero_point（最大为 255）加上可能会导致有符号 32 位整数溢出。
     */
    const int32_t x_biased = x_clamped + zero_point;
    const int32_t y_biased = y_clamped + zero_point;
    const int32_t z_biased = z_clamped + zero_point;
    const int32_t w_biased = w_clamped + zero_point;

    // 将偏置后的结果转换为 uint8_t 类型，并存入输出数组中，每个值占用一个字节
    output[0] = (uint8_t)x_biased;
    output[1] = (uint8_t)y_biased;
    output[2] = (uint8_t)z_biased;
    output[3] = (uint8_t)w_biased;
    // 更新输出数组的指针，使其指向下一个四字节的位置
    output += 4;
}
}



# 这是代码中的一个闭合大括号，用于结束一个代码块或控制结构。
```