# `.\pytorch\aten\src\ATen\native\quantized\cpu\qnnpack\src\requantization\precise-psimd.c`

```
/*
 * 版权所有（C）Facebook公司及其关联公司。
 * 保留所有权利。
 *
 * 此源代码根据根目录中的LICENSE文件中的BSD风格许可证许可。
 */

#include <assert.h>
#include <stdint.h>

#include <psimd.h>

#include <fp16/bitcasts.h>
#include <qnnpack/requantization-stubs.h>

void pytorch_qnnp_requantize_precise__psimd(
    size_t n,
    const int32_t* input,
    float scale,
    uint8_t zero_point,
    uint8_t qmin,
    uint8_t qmax,
    uint8_t* output) {
  // 确保输入大小是16的倍数
  assert(n % 16 == 0);
  // 确保缩放因子小于1.0
  assert(scale < 1.0f);
  // 确保缩放因子大于等于2^-32
  assert(scale >= 0x1.0p-32f);

  // 将浮点数缩放因子转换为32位整数的位表示
  const uint32_t scale_bits = fp32_to_bits(scale);
  // 构造乘法因子，包括缩放因子的位表示和符号位
  const uint32_t multiplier = (scale_bits << 8) | UINT32_C(0x80000000);
  // 计算右移的位数，用于乘法的舍入
  const uint32_t shift = 127 + 31 - (scale_bits >> 23);
  // 确保右移的位数在合理范围内
  assert(shift >= 32);
  assert(shift < 64);
  // 计算舍入常量
  const uint64_t rounding = UINT64_C(1) << (shift - 1);

  // 使用SIMD加载乘法因子的低16位并扩展为全局
  const psimd_u32 vmultiplier_lo =
      psimd_splat_u32(multiplier & UINT32_C(0x0000FFFF));
  // 使用SIMD加载乘法因子的高16位并扩展为全局
  const psimd_u32 vmultiplier_hi = psimd_splat_u32(multiplier >> 16);
  // 使用SIMD加载零点值并扩展为全局
  const psimd_s32 vzero_point = psimd_splat_s32((int32_t)(uint32_t)zero_point);
  // 使用SIMD加载qmin值并将其与零点值相减后扩展为全局
  const psimd_s32 vsmin =
      psimd_splat_s32((int32_t)(uint32_t)qmin - (int32_t)(uint32_t)zero_point);
  // 使用SIMD加载qmax值并将其与零点值相减后扩展为全局
  const psimd_s32 vsmax =
      psimd_splat_s32((int32_t)(uint32_t)qmax - (int32_t)(uint32_t)zero_point);
  // 使用SIMD加载舍入常量的低32位并扩展为全局
  const psimd_u32 vrounding_lo = psimd_splat_u32((uint32_t)rounding);
  // 使用SIMD加载舍入常量的高32位并扩展为全局
  const psimd_u32 vrounding_hi = psimd_splat_u32((uint32_t)(rounding >> 32));
  // 使用SIMD加载右移位数并扩展为全局
  const psimd_u32 vshift = psimd_splat_u32(shift - 32);

  // 对每个16个元素的块进行处理
  for (; n != 0; n -= 16) {
    // 加载输入数据向量x
    const psimd_s32 x = psimd_load_s32(input);
    // 加载输入数据向量y
    const psimd_s32 y = psimd_load_s32(input + 4);
    // 加载输入数据向量z
    const psimd_s32 z = psimd_load_s32(input + 8);
    // 加载输入数据向量w
    const psimd_s32 w = psimd_load_s32(input + 12);
    // 更新输入指针以处理下一个块
    input += 16;

    // 计算x的负值掩码
    const psimd_s32 x_neg_mask = x >> psimd_splat_s32(31);
    // 计算y的负值掩码
    const psimd_s32 y_neg_mask = y >> psimd_splat_s32(31);
    // 计算z的负值掩码
    const psimd_s32 z_neg_mask = z >> psimd_splat_s32(31);
    // 计算w的负值掩码
    const psimd_s32 w_neg_mask = w >> psimd_splat_s32(31);

    // 计算x的绝对值
    const psimd_u32 x_abs = (psimd_u32)((x ^ x_neg_mask) - x_neg_mask);
    // 计算y的绝对值
    const psimd_u32 y_abs = (psimd_u32)((y ^ y_neg_mask) - y_neg_mask);
    // 计算z的绝对值
    const psimd_u32 z_abs = (psimd_u32)((z ^ z_neg_mask) - z_neg_mask);
    // 计算w的绝对值
    const psimd_u32 w_abs = (psimd_u32)((w ^ w_neg_mask) - w_neg_mask);

    // 提取x的低16位绝对值
    const psimd_u32 x_abs_lo = x_abs & psimd_splat_u32(UINT32_C(0x0000FFFF));
    // 提取x的高16位绝对值
    const psimd_u32 x_abs_hi = x_abs >> psimd_splat_u32(16);
    // 提取y的低16位绝对值
    const psimd_u32 y_abs_lo = y_abs & psimd_splat_u32(UINT32_C(0x0000FFFF));
    // 提取y的高16位绝对值
    const psimd_u32 y_abs_hi = y_abs >> psimd_splat_u32(16);
    // 提取z的低16位绝对值
    const psimd_u32 z_abs_lo = z_abs & psimd_splat_u32(UINT32_C(0x0000FFFF));
    // 提取z的高16位绝对值
    const psimd_u32 z_abs_hi = z_abs >> psimd_splat_u32(16);
    // 提取w的低16位绝对值
    const psimd_u32 w_abs_lo = w_abs & psimd_splat_u32(UINT32_C(0x0000FFFF));
    // 提取w的高16位绝对值
    const psimd_u32 w_abs_hi = w_abs >> psimd_splat_u32(16);

    // 计算x的低32位乘积
    const psimd_u32 x_product_ll = x_abs_lo * vmultiplier_lo;
    // 计算低位部分的乘积
    const psimd_u32 y_product_ll = y_abs_lo * vmultiplier_lo;
    const psimd_u32 z_product_ll = z_abs_lo * vmultiplier_lo;
    const psimd_u32 w_product_ll = w_abs_lo * vmultiplier_lo;

    // 计算第二个乘积部分的低位
    const psimd_u32 x_product_lh =
        x_abs_lo * vmultiplier_hi + (x_product_ll >> psimd_splat_u32(16));
    const psimd_u32 y_product_lh =
        y_abs_lo * vmultiplier_hi + (y_product_ll >> psimd_splat_u32(16));
    const psimd_u32 z_product_lh =
        z_abs_lo * vmultiplier_hi + (z_product_ll >> psimd_splat_u32(16));
    const psimd_u32 w_product_lh =
        w_abs_lo * vmultiplier_hi + (w_product_ll >> psimd_splat_u32(16));

    // 计算第三个乘积部分的低位
    const psimd_u32 x_product_hl = x_abs_hi * vmultiplier_lo +
        (x_product_lh & psimd_splat_u32(UINT32_C(0x0000FFFF)));
    const psimd_u32 y_product_hl = y_abs_hi * vmultiplier_lo +
        (y_product_lh & psimd_splat_u32(UINT32_C(0x0000FFFF)));
    const psimd_u32 z_product_hl = z_abs_hi * vmultiplier_lo +
        (z_product_lh & psimd_splat_u32(UINT32_C(0x0000FFFF)));
    const psimd_u32 w_product_hl = w_abs_hi * vmultiplier_lo +
        (w_product_lh & psimd_splat_u32(UINT32_C(0x0000FFFF)));

    // 计算低位部分的乘积
    const psimd_u32 x_product_lo = (x_product_hl << psimd_splat_u32(16)) +
        (x_product_ll & psimd_splat_u32(UINT32_C(0x0000FFFF)));
    const psimd_u32 y_product_lo = (y_product_hl << psimd_splat_u32(16)) +
        (y_product_ll & psimd_splat_u32(UINT32_C(0x0000FFFF)));
    const psimd_u32 z_product_lo = (z_product_hl << psimd_splat_u32(16)) +
        (z_product_ll & psimd_splat_u32(UINT32_C(0x0000FFFF)));
    const psimd_u32 w_product_lo = (w_product_hl << psimd_splat_u32(16)) +
        (w_product_ll & psimd_splat_u32(UINT32_C(0x0000FFFF)));

    // 计算高位部分的乘积
    const psimd_u32 x_product_hi = x_abs_hi * vmultiplier_hi +
        (x_product_lh >> psimd_splat_u32(16)) +
        (x_product_hl >> psimd_splat_u32(16));
    const psimd_u32 y_product_hi = y_abs_hi * vmultiplier_hi +
        (y_product_lh >> psimd_splat_u32(16)) +
        (y_product_hl >> psimd_splat_u32(16));
    const psimd_u32 z_product_hi = z_abs_hi * vmultiplier_hi +
        (z_product_lh >> psimd_splat_u32(16)) +
        (z_product_hl >> psimd_splat_u32(16));
    const psimd_u32 w_product_hi = w_abs_hi * vmultiplier_hi +
        (w_product_lh >> psimd_splat_u32(16)) +
        (w_product_hl >> psimd_splat_u32(16));

    // 调整乘积结果并应用舍入
    const psimd_u32 x_adjusted_product = (x_product_hi + vrounding_hi) -
        ((psimd_s32)(x_product_lo & vrounding_lo) >> psimd_splat_s32(31));
    const psimd_u32 y_adjusted_product = (y_product_hi + vrounding_hi) -
        ((psimd_s32)(y_product_lo & vrounding_lo) >> psimd_splat_s32(31));
    const psimd_u32 z_adjusted_product = (z_product_hi + vrounding_hi) -
        ((psimd_s32)(z_product_lo & vrounding_lo) >> psimd_splat_s32(31));
    const psimd_u32 w_adjusted_product = (w_product_hi + vrounding_hi) -
        ((psimd_s32)(w_product_lo & vrounding_lo) >> psimd_splat_s32(31));

    // 对调整后的乘积结果进行缩放
    const psimd_u32 x_abs_scaled = x_adjusted_product >> vshift;
    // 将 y_adjusted_product 右移 vshift 位，并存储结果到 y_abs_scaled
    const psimd_u32 y_abs_scaled = y_adjusted_product >> vshift;
    // 将 z_adjusted_product 右移 vshift 位，并存储结果到 z_abs_scaled
    const psimd_u32 z_abs_scaled = z_adjusted_product >> vshift;
    // 将 w_adjusted_product 右移 vshift 位，并存储结果到 w_abs_scaled
    const psimd_u32 w_abs_scaled = w_adjusted_product >> vshift;

    // 计算 x_scaled：将 x_abs_scaled 与 x_neg_mask 异或后强制类型转换为 psimd_s32，并减去 x_neg_mask
    const psimd_s32 x_scaled =
        (psimd_s32)(x_abs_scaled ^ x_neg_mask) - x_neg_mask;
    // 计算 y_scaled：将 y_abs_scaled 与 y_neg_mask 异或后强制类型转换为 psimd_s32，并减去 y_neg_mask
    const psimd_s32 y_scaled =
        (psimd_s32)(y_abs_scaled ^ y_neg_mask) - y_neg_mask;
    // 计算 z_scaled：将 z_abs_scaled 与 z_neg_mask 异或后强制类型转换为 psimd_s32，并减去 z_neg_mask
    const psimd_s32 z_scaled =
        (psimd_s32)(z_abs_scaled ^ z_neg_mask) - z_neg_mask;
    // 计算 w_scaled：将 w_abs_scaled 与 w_neg_mask 异或后强制类型转换为 psimd_s32，并减去 w_neg_mask
    const psimd_s32 w_scaled =
        (psimd_s32)(w_abs_scaled ^ w_neg_mask) - w_neg_mask;

    // 计算 x_clamped：对 x_scaled 进行上下限约束，再加上 vzero_point
    const psimd_u32 x_clamped =
        (psimd_u32)psimd_max_s32(psimd_min_s32(x_scaled, vsmax), vsmin) +
        vzero_point;
    // 计算 y_clamped：对 y_scaled 进行上下限约束，再加上 vzero_point
    const psimd_u32 y_clamped =
        (psimd_u32)psimd_max_s32(psimd_min_s32(y_scaled, vsmax), vsmin) +
        vzero_point;
    // 计算 z_clamped：对 z_scaled 进行上下限约束，再加上 vzero_point
    const psimd_u32 z_clamped =
        (psimd_u32)psimd_max_s32(psimd_min_s32(z_scaled, vsmax), vsmin) +
        vzero_point;
    // 计算 w_clamped：对 w_scaled 进行上下限约束，再加上 vzero_point
    const psimd_u32 w_clamped =
        (psimd_u32)psimd_max_s32(psimd_min_s32(w_scaled, vsmax), vsmin) +
        vzero_point;

    // 将 x_clamped 和 y_clamped 合并成 psimd_u16 类型的 xy_clamped
    const psimd_u16 xy_clamped =
        psimd_concat_even_u16((psimd_u16)x_clamped, (psimd_u16)y_clamped);
    // 将 z_clamped 和 w_clamped 合并成 psimd_u16 类型的 zw_clamped
    const psimd_u16 zw_clamped =
        psimd_concat_even_u16((psimd_u16)z_clamped, (psimd_u16)w_clamped);

    // 将 xy_clamped 和 zw_clamped 合并成 psimd_u8 类型的 xyzw_clamped
    const psimd_u8 xyzw_clamped =
        psimd_concat_even_u8((psimd_u8)xy_clamped, (psimd_u8)zw_clamped);

    // 将 xyzw_clamped 存储到 output 指向的内存位置
    psimd_store_u8(output, xyzw_clamped);
    // 将 output 指针移动到下一个 16 字节对齐的位置
    output += 16;
}
}



# 这行代码是一个单独的右括号 '}'，用于结束某个代码块（例如函数、循环或条件语句）的定义。
```