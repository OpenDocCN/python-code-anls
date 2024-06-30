# `D:\src\scipysrc\scipy\scipy\io\_fast_matrix_market\fast_matrix_market\dependencies\ryu\ryu\common.h`

```
// 定义条件编译宏，检测当前平台是否为32位平台，包括x86和ARM架构
#if defined(_M_IX86) || defined(_M_ARM)
#define RYU_32_BIT_PLATFORM
#endif

// 返回一个32位整数v的十进制表示的位数，v必须不超过9位数字
static inline uint32_t decimalLength9(const uint32_t v) {
  // 函数前提条件：v不是一个10位数
  // (f2s: 9位数字足以保证往返转换的准确性)
  // (d2fixed: 我们以9位数字块的形式打印)
  assert(v < 1000000000);
  if (v >= 100000000) { return 9; }
  if (v >= 10000000) { return 8; }
  if (v >= 1000000) { return 7; }
  if (v >= 100000) { return 6; }
  if (v >= 10000) { return 5; }
  if (v >= 1000) { return 4; }
  if (v >= 100) { return 3; }
  if (v >= 10) { return 2; }
  return 1;
}

// 返回 e == 0 ? 1 : [log_2(5^e)]; 要求 0 <= e <= 3528
static inline int32_t log2pow5(const int32_t e) {
  // 这个近似方法在乘法运算溢出之前有效，当 e = 3529 时乘法运算会溢出
  // 如果乘法运算使用64位，则会在 5^4004 处失败，这个值略大于 2^9297
  assert(e >= 0);
  assert(e <= 3528);
  return (int32_t) ((((uint32_t) e) * 1217359) >> 19);
}

// 返回 e == 0 ? 1 : ceil(log_2(5^e)); 要求 0 <= e <= 3528
static inline int32_t pow5bits(const int32_t e) {
  // 这个近似方法在乘法运算溢出之前有效，当 e = 3529 时乘法运算会溢出
  // 如果乘法运算使用64位，则会在 5^4004 处失败，这个值略大于 2^9297
  assert(e >= 0);
  assert(e <= 3528);
  return (int32_t) (((((uint32_t) e) * 1217359) >> 19) + 1);
}

// 返回 e == 0 ? 1 : ceil(log_2(5^e)); 要求 0 <= e <= 3528
static inline int32_t ceil_log2pow5(const int32_t e) {
  return log2pow5(e) + 1;
}

// 返回 floor(log_10(2^e)); 要求 0 <= e <= 1650
static inline uint32_t log10Pow2(const int32_t e) {
  // 这个近似方法在乘法运算溢出之前有效，当 e = 1651 时乘法运算会溢出
  // 第一个失败的值是 2^1651，略大于 10^297
  assert(e >= 0);
  assert(e <= 1650);
  return (((uint32_t) e) * 78913) >> 18;
}

// 返回 floor(log_10(5^e)); 要求 0 <= e <= 2620
static inline uint32_t log10Pow5(const int32_t e) {
  // 这个近似方法在乘法运算溢出之前有效，当 e = 2621 时乘法运算会溢出
  // 第一个失败的值是 5^2621，略大于 10^1832
  assert(e >= 0);
  assert(e <= 2620);
  return (((uint32_t) e) * 732923) >> 20;
}
# 复制特殊字符串到目标数组中，并返回复制的字符数
static inline int copy_special_str(char * const result, const bool sign, const bool exponent, const bool mantissa) {
    # 如果需要复制的字符串是 NaN，则将 "NaN" 复制到结果数组中
    if (mantissa) {
        memcpy(result, "NaN", 3);
        return 3;
    }
    # 如果需要复制负号，则将 '-' 放入结果数组的第一个位置
    if (sign) {
        result[0] = '-';
    }
    # 如果需要复制指数部分，则将 "Infinity" 复制到结果数组中
    if (exponent) {
        memcpy(result + sign, "Infinity", 8);
        return sign + 8;
    }
    # 如果没有特殊要求，则将 "0E0" 复制到结果数组中
    memcpy(result + sign, "0E0", 3);
    return sign + 3;
}

# 将浮点数转换为其对应的位表示（32位）
static inline uint32_t float_to_bits(const float f) {
    uint32_t bits = 0;
    # 将浮点数的二进制表示复制到 bits 变量中
    memcpy(&bits, &f, sizeof(float));
    return bits;
}

# 将双精度浮点数转换为其对应的位表示（64位）
static inline uint64_t double_to_bits(const double d) {
    uint64_t bits = 0;
    # 将双精度浮点数的二进制表示复制到 bits 变量中
    memcpy(&bits, &d, sizeof(double));
    return bits;
}

# 结束定义 RYU_COMMON_H 的条件编译
#endif // RYU_COMMON_H
```