# `.\pytorch\c10\test\util\Half_test.cpp`

```py
namespace {

// 将半精度浮点数的二进制表示转换为单精度浮点数
float halfbits2float(unsigned short h) {
  // 提取符号位
  unsigned sign = ((h >> 15) & 1);
  // 提取指数位
  unsigned exponent = ((h >> 10) & 0x1f);
  // 提取尾数位并左移13位
  unsigned mantissa = ((h & 0x3ff) << 13);

  // 处理特殊情况：NaN 或 Inf
  if (exponent == 0x1f) {
    mantissa = (mantissa ? (sign = 0, 0x7fffff) : 0);
    exponent = 0xff;
  } 
  // 处理特殊情况：非规格化数或零
  else if (!exponent) {
    if (mantissa) {
      // 找到尾数的最高有效位并进行规范化
      unsigned int msb;
      exponent = 0x71;
      do {
        msb = (mantissa & 0x400000);
        mantissa <<= 1; /* normalize */
        --exponent;
      } while (!msb);
      mantissa &= 0x7fffff; /* 1.mantissa is implicit */
    }
  } 
  // 普通情况：调整指数位
  else {
    exponent += 0x70;
  }

  // 组合结果的单精度浮点数位
  unsigned result_bit = (sign << 31) | (exponent << 23) | mantissa;

  // 调用 C10 库函数将二进制位转换为单精度浮点数
  return c10::detail::fp32_from_bits(result_bit);
}

// 将单精度浮点数转换为半精度浮点数的二进制表示
unsigned short float2halfbits(float src) {
  // 调用 C10 库函数将单精度浮点数转换为其二进制表示
  unsigned x = c10::detail::fp32_to_bits(src);

  // NOLINTNEXTLINE(cppcoreguidelines-init-variables,cppcoreguidelines-avoid-magic-numbers)
  unsigned u = (x & 0x7fffffff), remainder, shift, lsb, lsb_s1, lsb_m1;
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  unsigned sign, exponent, mantissa;

  // 处理特殊情况：NaN 或 Inf
  if (u > 0x7f800000) {
    return 0x7fffU;
  }

  sign = ((x >> 16) & 0x8000);

  // 处理特殊情况：Inf 或 0
  if (u > 0x477fefff) {
    return sign | 0x7c00U;
  }
  if (u < 0x33000001) {
    return (sign | 0x0000);
  }

  exponent = ((u >> 23) & 0xff);
  mantissa = (u & 0x7fffff);

  // 调整指数位和尾数位，以便转换为半精度浮点数的二进制表示
  if (exponent > 0x70) {
    shift = 13;
    exponent -= 0x70;
  } else {
    shift = 0x7e - exponent;
    exponent = 0;
    mantissa |= 0x800000;
  }
  lsb = (1 << shift);
  lsb_s1 = (lsb >> 1);
  lsb_m1 = (lsb - 1);

  // 进行最近偶数舍入
  remainder = (mantissa & lsb_m1);
  mantissa >>= shift;
  if (remainder > lsb_s1 || (remainder == lsb_s1 && (mantissa & 0x1))) {
    ++mantissa;
    if (!(mantissa & 0x3ff)) {
      ++exponent;
      mantissa = 0;
    }
  }

  // 组合结果的半精度浮点数的二进制表示
  return (sign | (exponent << 10) | mantissa);
}

// 测试半精度浮点数与单精度浮点数之间的互转
TEST(HalfConversionTest, TestPorableConversion) {
  // 定义测试用例：半精度浮点数的二进制表示
  std::vector<uint16_t> inputs = {
      0,
      0xfbff, // 1111 1011 1111 1111
      (1 << 15 | 1),
      0x7bff // 0111 1011 1111 1111
  };
  // 遍历测试用例
  for (auto x : inputs) {
    // 将半精度浮点数的二进制表示转换为单精度浮点数，期望得到的值
    auto target = c10::detail::fp16_ieee_to_fp32_value(x);
    // 断言：转换后的单精度浮点数与期望值相等
    EXPECT_EQ(halfbits2float(x), target)
        << "Test failed for uint16 to float " << x << "\n";
    // 将单精度浮点数转换为半精度浮点数的二进制表示，期望得到的值
    EXPECT_EQ(
        float2halfbits(target), c10::detail::fp16_ieee_from_fp32_value(target))
        << "Test failed for float to uint16" << target << "\n";
  }
}

// 测试从本地半精度浮点数到单精度浮点数的转换
TEST(HalfConversion, TestNativeConversionToFloat) {
  // 遍历所有可能的半精度浮点数的二进制表示
  for (auto x : c10::irange(std::numeric_limits<uint16_t>::max() + 1)) {
    // 创建半精度浮点数对象
    auto h = c10::Half(x, c10::Half::from_bits());
    // 将输入的半精度浮点数转换为单精度浮点数
    auto f = halfbits2float(x);
    // NaN 不等于任何其他 NaN，这里检查是否都是 NaN
    if (std::isnan(f) && std::isnan(static_cast<float>(h))) {
      // 如果都是 NaN，则继续下一次循环
      continue;
    }
    // 使用单精度浮点数进行期望值断言比较，检查转换是否正确
    EXPECT_EQ(f, static_cast<float>(h)) << "Conversion error using " << x;
  }
}

TEST(HalfConversion, TestNativeConversionToHalf) {
  auto check_conversion = [](float f) {
    // 使用c10命名空间中的Half构造函数将float转换为半精度浮点数
    auto h = c10::Half(f);
    // 使用float2halfbits函数将float转换为半精度浮点数表示的比特位
    auto h_bits = float2halfbits(f);
    // NaN不等于其他NaN，只需检查h是否为NaN即可
    if (std::isnan(f)) {
      EXPECT_TRUE(std::isnan(static_cast<float>(h)));
    } else {
      // 检查转换后的半精度浮点数h的比特位是否与float2halfbits函数转换的一致
      EXPECT_EQ(h.x, h_bits) << "Conversion error using " << f;
    }
  };

  // 遍历从0到最大uint16_t的所有整数，即遍历半精度浮点数的所有可能表示
  for (auto x : c10::irange(std::numeric_limits<uint16_t>::max() + 1)) {
    // 使用halfbits2float函数将半精度浮点数的比特位转换回float，然后检查转换
    check_conversion(halfbits2float(x));
  }
  // 检查一些超出半精度浮点数范围的值的转换情况
  check_conversion(std::numeric_limits<float>::max());
  check_conversion(std::numeric_limits<float>::min());
  check_conversion(std::numeric_limits<float>::epsilon());
  check_conversion(std::numeric_limits<float>::lowest());
}

} // namespace
```