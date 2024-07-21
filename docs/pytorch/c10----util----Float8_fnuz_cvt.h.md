# `.\pytorch\c10\util\Float8_fnuz_cvt.h`

```py
/*
 * 在 c10::detail 命名空间中定义了一个模板函数，用于将以 f8 E4M3FNUZ 或 bf8 E5M2FNUZ 格式表示的
 * 8位浮点数转换为32位浮点数的值。
 * 参数 we 和 wm 分别指定了输入格式的位宽，we 表示指数部分的位数，wm 表示尾数部分的位数。
 */
template <uint32_t we, uint32_t wm>
inline C10_HOST_DEVICE float fp8_fnuz_to_fp32_value(uint8_t x) {
  // 确保输入的格式符合 f8 E4M3FNUZ 或 bf8 E5M2FNUZ 格式之一
  static_assert((we == 4 && wm == 3) || (we == 5 && wm == 2));

  // 定义常量 weo 和 wmo 分别表示目标 32 位浮点数的指数和尾数的位数
  constexpr uint32_t weo = 8;
  constexpr uint32_t wmo = 23;

  // 处理特殊值情况
  if (x == 0) {
    return 0;  // 若输入 x 为 0，则返回浮点数 0
  }

  if (x == 0x80) {
    // 若输入 x 为 0x80，则表示 NaN，返回对应的浮点数 NaN
    constexpr uint32_t ifNaN = 0x7F800001;
    return fp32_from_bits(ifNaN);
  }

  // 从输入 x 中提取尾数和指数部分
  uint32_t mantissa = x & ((1 << wm) - 1);  // 提取尾数部分
  uint32_t exponent = (x & 0x7F) >> wm;     // 提取指数部分

  // 处理 subnormal 输入情况
  if (exponent == 0) {
    // 对于 subnormal 输入，确保尾数部分不为 0，因为 0x0 和 0x80 的情况在上面已经处理了
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    uint32_t renorm_shift = __clz(mantissa);  // CUDA 或 HIP 编译器的处理方式
#elif defined(__SYCL_DEVICE_ONLY__)
    uint32_t renorm_shift = sycl::clz(mantissa);  // SYCL 设备编译器的处理方式
#elif defined(_MSC_VER)
    unsigned long nonsign_bsr;
    _BitScanReverse(&nonsign_bsr, (unsigned long)mantissa);
    uint32_t renorm_shift = (uint32_t)nonsign_bsr ^ 31;  // Microsoft 编译器的处理方式
#else
    uint32_t renorm_shift = __builtin_clz(mantissa);  // 默认情况下的处理方式
#endif
    uint32_t sh = 1 + renorm_shift - (32 - wm);
    mantissa <<= sh;
    exponent += 1 - sh;
    mantissa &= ((1 << wm) - 1);
  }

  // 计算指数的低位截止值，并更新指数和尾数部分
  const uint32_t exp_low_cutoff = (1 << (weo - 1)) - (1 << (we - 1));
  exponent += exp_low_cutoff - 1;
  mantissa <<= wmo - wm;

  // 构建 32 位浮点数的位表示
  uint32_t sign = x >> 7;
  uint32_t retval = (sign << 31) | (exponent << 23) | mantissa;

  // 返回最终的 32 位浮点数值
  return fp32_from_bits(retval);
}
```