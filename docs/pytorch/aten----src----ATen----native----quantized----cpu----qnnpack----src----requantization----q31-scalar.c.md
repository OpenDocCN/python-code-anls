# `.\pytorch\aten\src\ATen\native\quantized\cpu\qnnpack\src\requantization\q31-scalar.c`

```py
/*
 * 版权声明：Facebook公司及其关联公司版权所有。
 * 保留所有权利。
 *
 * 此源代码在根目录下的LICENSE文件中以BSD风格许可证授权。
 */

#include <assert.h>
#include <stdint.h>

#include <fp16/bitcasts.h>
#include <qnnpack/requantization-stubs.h>
#include <qnnpack/scalar-utils.h>

void pytorch_qnnp_requantize_q31__scalar(
    size_t n,
    const int32_t* input,
    float scale,
    uint8_t zero_point,
    uint8_t qmin,
    uint8_t qmax,
    uint8_t* output) {
  // 断言：输入数据个数必须是4的倍数
  assert(n % 4 == 0);
  // 断言：缩放因子必须小于1.0
  assert(scale < 1.0f);
  // 断言：缩放因子必须大于等于2^-32
  assert(scale >= 0x1.0p-32f);

  /* 计算重新量化的参数 */
  const uint32_t scale_bits = fp32_to_bits(scale);

  /* 倍乘器在[0x40000000, 0x7FFFFF80]范围内 */
  const int32_t multiplier = (int32_t)(
      ((scale_bits & UINT32_C(0x007FFFFF)) | UINT32_C(0x00800000)) << 7);
  assert(multiplier >= INT32_C(0x40000000));
  assert(multiplier <= INT32_C(0x7FFFFF80));

  /* 移位值在[0, 31]范围内 */
  const int32_t shift = 127 + 31 - 32 - (fp32_to_bits(scale) >> 23);
  assert(shift >= 0);
  assert(shift < 32);

  const int64_t q31rounding = INT64_C(0x40000000);
  const int32_t remainder_mask =
      (int32_t)((UINT32_C(1) << shift) - UINT32_C(1));
  const int32_t threshold = (int32_t)((uint32_t)remainder_mask >> 1);
  const int32_t smin = (int32_t)(uint32_t)qmin - (int32_t)(uint32_t)zero_point;
  const int32_t smax = (int32_t)(uint32_t)qmax - (int32_t)(uint32_t)zero_point;
  for (; n != 0; n -= 4) {
    const int32_t x = input[0];
    const int32_t y = input[1];
    const int32_t z = input[2];
    const int32_t w = input[3];
    input += 4;

    /*
     * 计算带符号32位因子的完整64位乘积。
     *
     * 注意：倍乘器可以被视为有符号或无符号。
     */
    const int64_t x_product = (int64_t)x * (int64_t)multiplier;
    const int64_t y_product = (int64_t)y * (int64_t)multiplier;
    const int64_t z_product = (int64_t)z * (int64_t)multiplier;
    const int64_t w_product = (int64_t)w * (int64_t)multiplier;

    /*
     * 通过提取产品的31-62位进行Q31乘法结果计算，带舍入。
     * 添加舍入值（0x40000000），然后右移31位并提取低32位字。
     * 注意：需要将转换为无符号类型以避免未定义行为。
     * 鉴于倍乘器范围，Q31乘法的结果在[-2147483520, 2147483519]范围内。
     */
    const int32_t x_q31product =
        (int32_t)(uint32_t)((uint64_t)(x_product + q31rounding) >> 31);
    const int32_t y_q31product =
        (int32_t)(uint32_t)((uint64_t)(y_product + q31rounding) >> 31);
    const int32_t z_q31product =
        (int32_t)(uint32_t)((uint64_t)(z_product + q31rounding) >> 31);
    const int32_t w_q31product =
        (int32_t)(uint32_t)((uint64_t)(w_product + q31rounding) >> 31);
    /*
     * 对调整后的乘积进行算术右移并进行四舍五入。
     * 四舍五入向最接近的整数进行，若中点则向远离零的方向舍入。
     *
     * 使用正确的四舍五入可以通过预先添加四舍五入常量来有效实现，
     * 但由于输入在 [-2147483520, 2147483519] 范围内，并且四舍五入常量高达 2**30，
     * 我们无法排除溢出的可能性。这个限制给我们留下了三个选择：
     * 1. 将输入扩展到64位有符号整数，对64位整数执行加法和移位，然后截断结果为32位。
     * 2. 检测溢出并单独处理此情况。注意溢出仅在输入为正数时可能发生，即使四舍五入常量溢出32位有符号整数，
     *    它仍不会溢出32位无符号整数。因此，在有符号溢出的情况下，可以使用无符号算术来计算结果，特别是在算术右移的情况下。
     * 3. 如原样进行算术右移，这将产生向下舍入的除法结果。然后通过除以2的幂的余数来调整结果。当：
     *    - 输入为正数，移位非零，并且余数 >= 2**(shift - 1)，例如 10 >> 2 需要调整
     *    - 输入为负数，移位非零，并且余数 > 2**(shift - 1)，例如 -10 >> 2 不需要调整
     *    这些条件可以泛化为 remainder + (input <= 0) > 2**(shift - 1) 或等效地
     *    remainder - (input < 0) > ((2**shift - 1) >> 1)
     *    当 shift 为 0 时，余数也为 0，最后一个条件始终为假，并且不执行任何调整。
     *
     * 在这些选项中，选项 3 在各个方面的性能表现最佳，尽管对于64位指令集来说，选项 1 也很有前景。
     */
    const int32_t x_remainder =
        (x_q31product & remainder_mask) - (int32_t)(x_q31product < 0);
    const int32_t y_remainder =
        (y_q31product & remainder_mask) - (int32_t)(y_q31product < 0);
    const int32_t z_remainder =
        (z_q31product & remainder_mask) - (int32_t)(z_q31product < 0);
    const int32_t w_remainder =
        (w_q31product & remainder_mask) - (int32_t)(w_q31product < 0);

    // 对 x_q31product 进行算术右移并根据余数是否超过阈值进行调整
    const int32_t x_scaled =
        asr_s32(x_q31product, shift) + (int32_t)(x_remainder > threshold);
    // 对 y_q31product 进行算术右移并根据余数是否超过阈值进行调整
    const int32_t y_scaled =
        asr_s32(y_q31product, shift) + (int32_t)(y_remainder > threshold);
    // 对 z_q31product 进行算术右移并根据余数是否超过阈值进行调整
    const int32_t z_scaled =
        asr_s32(z_q31product, shift) + (int32_t)(z_remainder > threshold);
    // 对 w_q31product 进行算术右移并根据余数是否超过阈值进行调整
    const int32_t w_scaled =
        asr_s32(w_q31product, shift) + (int32_t)(w_remainder > threshold);

    /*
     * 将调整后的值夹紧到零点介于 (qmin - zero point) 和 (qmax - zero point) 之间。
     */
    // 将 x_scaled 值限制在 smin 和 smax 之间，并赋给 x_clamped
    const int32_t x_clamped =
        x_scaled < smin ? smin : x_scaled > smax ? smax : x_scaled;
    // 将 y_scaled 值限制在 smin 和 smax 之间，并赋给 y_clamped
    const int32_t y_clamped =
        y_scaled < smin ? smin : y_scaled > smax ? smax : y_scaled;
    // 将 z_scaled 值限制在 smin 和 smax 之间，并赋给 z_clamped
    const int32_t z_clamped =
        z_scaled < smin ? smin : z_scaled > smax ? smax : z_scaled;
    // 将 w_scaled 值限制在 smin 和 smax 之间，并赋给 w_clamped
    const int32_t w_clamped =
        w_scaled < smin ? smin : w_scaled > smax ? smax : w_scaled;

    /*
     * 将零点加到 clamped 值上。
     * 结果保证在 [qmin, qmax] 范围内。
     *
     * 在进行限制之前无法安全地执行此加法，因为缩放后的值在 [-2147483520, 2147483519] 范围内，
     * 因此零点的加法（最多可达 255）可能会导致有符号 32 位整数溢出。
     */
    // 将 zero_point 加到 x_clamped 上，并赋给 x_biased
    const int32_t x_biased = x_clamped + zero_point;
    // 将 zero_point 加到 y_clamped 上，并赋给 y_biased
    const int32_t y_biased = y_clamped + zero_point;
    // 将 zero_point 加到 z_clamped 上，并赋给 z_biased
    const int32_t z_biased = z_clamped + zero_point;
    // 将 zero_point 加到 w_clamped 上，并赋给 w_biased
    const int32_t w_biased = w_clamped + zero_point;

    // 将 x_biased 转换为 uint8_t 类型并存入 output 数组的第一个位置
    output[0] = (uint8_t)x_biased;
    // 将 y_biased 转换为 uint8_t 类型并存入 output 数组的第二个位置
    output[1] = (uint8_t)y_biased;
    // 将 z_biased 转换为 uint8_t 类型并存入 output 数组的第三个位置
    output[2] = (uint8_t)z_biased;
    // 将 w_biased 转换为 uint8_t 类型并存入 output 数组的第四个位置
    output[3] = (uint8_t)w_biased;
    // 更新 output 指针，使其指向下一个四个字节的位置
    output += 4;
}
}



# 这是一个单独的右大括号 '}'，用于结束一个代码块、函数或类定义的语法结构。
```