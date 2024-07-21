# `.\pytorch\c10\util\int128.cpp`

```
// 包含日志记录和整数128位操作的必要头文件
#include <c10/util/Logging.h>
#include <c10/util/int128.h>
// 包含输出流和NOLINT指令的必要头文件
#include <iomanip>
#include <ostream> // NOLINT(readability/streams)

// 定义命名空间c10，用于组织代码
namespace c10 {

// 定义一个常量uint128_pod，表示128位无符号整数的最大值
const uint128_pod kuint128max = {
    uint64_t{0xFFFFFFFFFFFFFFFFu},  // 高64位全为1
    uint64_t{0xFFFFFFFFFFFFFFFFu}}; // 低64位全为1

// 返回给定uint64类型数的最高位（即最高有效位）的索引，从0开始计数
// 参数n不能为0
//
// 例如：
//   给定：5（十进制）== 101（二进制）
//   返回：2
#define STEP(T, n, pos, sh)                   \
  do {                                        \
    if ((n) >= (static_cast<T>(1) << (sh))) { \
      (n) = (n) >> (sh);                      \
      (pos) |= (sh);                          \
    }                                         \
  } while (0)

// 定义静态内联函数Fls64，实现返回给定uint64_t数的最高位索引
static inline int Fls64(uint64_t n) {
  // GOOGLE_DCHECK_NE(0, n); // 该行被注释掉，应为用于断言n不为0
  uint64_t pos = 0;
  // 分步检查n中的高位，设置pos来记录最高位索引
  STEP(uint64_t, n, pos, 0x20); // 检查64位中的高32位
  uint32_t n32 = n;
  STEP(uint32_t, n32, pos, 0x10); // 检查32位中的高16位
  STEP(uint32_t, n32, pos, 0x08); // 检查16位中的高8位
  STEP(uint32_t, n32, pos, 0x04); // 检查8位中的高4位
  // 返回最高位的索引，使用位运算和预先定义的常量来计算
  return static_cast<int>(
      pos + ((uint64_t{0x3333333322221100u} >> (n32 << 2)) & 0x3));
}
#undef STEP
// 返回给定 uint128 中最高位（即最重要位）的位置，从 0 开始计数，类似于上面的 Fls64()。
// 参数 n 不能为 0。
static inline int Fls128(uint128 n) {
  // 检查 uint128 的高 64 位是否非零，如果是，调用 Uint128High64() 函数获取最高位的位置并加上 64
  if (uint64_t hi = Uint128High64(n)) {
    return Fls64(hi) + 64;
  }
  // 否则，调用 Uint128Low64() 函数获取最低位的位置
  return Fls64(Uint128Low64(n));
}

// 实现 uint128 类的除法和取模运算
void uint128::DivModImpl(
    uint128 dividend,
    uint128 divisor,
    uint128* quotient_ret,
    uint128* remainder_ret) {
  // 如果除数为 0，记录错误日志并中止程序
  if (divisor == 0) {
    LOG(FATAL) << "Division or mod by zero: dividend.hi=" << dividend.hi_
               << ", lo=" << dividend.lo_;
  } else if (dividend < divisor) {
    // 如果被除数小于除数，设置商为 0，余数为被除数，并返回
    *quotient_ret = 0;
    *remainder_ret = dividend;
    return;
  } else {
    // 计算被除数和除数的位长度
    int dividend_bit_length = Fls128(dividend);
    int divisor_bit_length = Fls128(divisor);
    int difference = dividend_bit_length - divisor_bit_length;
    uint128 quotient = 0;
    // 循环直到被除数减去移位后的除数小于等于 0
    while (difference >= 0) {
      quotient <<= 1;
      uint128 shifted_divisor = divisor << difference;
      if (shifted_divisor <= dividend) {
        dividend -= shifted_divisor;
        quotient += 1;
      }
      difference -= 1;
    }
    // 记录最终的商和余数
    *quotient_ret = quotient;
    *remainder_ret = dividend;
  }
}

// 实现 uint128 类的除法赋值运算符
uint128& uint128::operator/=(const uint128& divisor) {
  uint128 quotient = 0;
  uint128 remainder = 0;
  DivModImpl(*this, divisor, &quotient, &remainder);
  *this = quotient;
  return *this;
}

// 实现 uint128 类的取模赋值运算符
uint128& uint128::operator%=(const uint128& divisor) {
  uint128 quotient = 0;
  uint128 remainder = 0;
  DivModImpl(*this, divisor, &quotient, &remainder);
  *this = remainder;
  return *this;
}

// 重载流输出运算符以便将 uint128 类对象输出到流中
std::ostream& operator<<(std::ostream& o, const uint128& b) {
  std::ios_base::fmtflags flags = o.flags();

  // 根据流的格式选项选择适当的除数，该除数是小于 2^64 的最大的基数的幂次方
  uint128 div;
  int div_base_log = 0;
  switch (flags & std::ios::basefield) {
    case std::ios::hex:
      div = (uint64_t)0x1000000000000000u; // 16^15
      div_base_log = 15;
      break;
    case std::ios::oct:
      div = (uint64_t)01000000000000000000000u; // 8^21
      div_base_log = 21;
      break;
    default: // std::ios::dec
      div = (uint64_t)10000000000000000000u; // 10^19
      div_base_log = 19;
      break;
  }

  // 将 uint128 表示分为三个小于 "div" 的部分，并分别转换成 uint64
  std::ostringstream os;
  std::ios_base::fmtflags copy_mask =
      std::ios::basefield | std::ios::showbase | std::ios::uppercase;
  os.setf(flags & copy_mask, copy_mask);
  uint128 high = b;
  uint128 low;
  uint128::DivModImpl(high, div, &high, &low);
  uint128 mid;
  uint128::DivModImpl(high, div, &high, &mid);
  if (high.lo_ != 0) {
    os << high.lo_;
    os << std::noshowbase << std::setfill('0') << std::setw(div_base_log);
    os << mid.lo_;
    os << std::setw(div_base_log);
  } else if (mid.lo_ != 0) {
    os << mid.lo_;
``
    // 将输出流设置为不显示基数，并填充到指定的位数
    os << std::noshowbase << std::setfill('0') << std::setw(div_base_log);
  }
  // 将低位数值追加到输出流中
  os << low.lo_;
  // 获取输出流的字符串表示
  std::string rep = os.str();

  // 添加必要的填充
  std::streamsize width = o.width(0);
  // 如果需要的宽度大于当前字符串长度，则进行填充
  if (width > static_cast<std::streamsize>(rep.size())) {
    if ((flags & std::ios::adjustfield) == std::ios::left) {
      // 左对齐情况下，在字符串末尾添加填充字符
      rep.append(width - rep.size(), o.fill());
    } else {
      // 右对齐情况下，在字符串开头插入填充字符
      rep.insert(
          static_cast<std::string::size_type>(0), width - rep.size(), o.fill());
    }
  }

  // 以单次 "<<" 调用流式输出最终的表示形式
  // 将最终字符串输出到流 o 中
  return o << rep;
}

} // namespace c10
```