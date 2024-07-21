# `.\pytorch\c10\util\int128.h`

```py
// 这个文件基于 protobuf 的 uint128 实现，源自以下链接：
// https://github.com/protocolbuffers/protobuf/blob/1e88936fce10cf773cb72b44c6a7f48b38c7578b/src/google/protobuf/stubs/int128.h
//
// Protocol Buffers - Google's data interchange format
// 版权所有 2008 Google Inc. 保留所有权利。
// https://developers.google.com/protocol-buffers/
//
// 在源代码和二进制形式的重新发布和使用时，需符合以下条件：
//
//     * 必须保留上述版权声明、此条件列表和以下免责声明。
//     * 在文档和/或其他提供的材料中，必须复制上述版权声明、此条件列表和以下免责声明。
//     * 未经特定的书面许可，不得使用 Google Inc. 或其贡献者的名称，来认可或推广从本软件派生的产品。
//
// 本软件由版权持有者和贡献者“按现状”提供，任何明示或暗示的担保，
// 包括但不限于对适销性和特定用途的适用性的暗示担保，均被拒绝。
// 在任何情况下，版权持有者或贡献者对任何直接的、间接的、偶然的、
// 特殊的、惩罚性的或后果性的损害赔偿（包括但不限于替代商品或服务的
// 采购、使用损失、数据或利润的损失或业务中断）概不负责，无论是在合同、
// 严格责任或侵权行为（包括疏忽或其他方式）引起的，即使事先已告知可能性。
#pragma once

#include <c10/macros/Export.h>
#include <cstdint>
#include <iosfwd>

namespace c10 {

// 声明一个结构体 uint128_pod，具体定义未在此处给出
struct uint128_pod;

// 当 constexpr 可用时，定义 UINT128_CONSTEXPR 为 constexpr
// 否则定义为普通的空宏
#ifdef GOOGLE_PROTOBUF_HAS_CONSTEXPR
#define UINT128_CONSTEXPR constexpr
#else
#define UINT128_CONSTEXPR
#endif

// 声明一个类 uint128
class uint128;

// 声明 uint128 类的左移操作符重载函数
inline uint128& operator<<=(uint128& self, int amount);

// 无符号的 128 位整数类型，线程兼容
class C10_API uint128 {
 public:
  // 默认构造函数，设置为 0，但不能依赖这种行为
  UINT128_CONSTEXPR uint128();
  // 构造函数，初始化为给定的 top 和 bottom 的值
  UINT128_CONSTEXPR uint128(uint64_t top, uint64_t bottom);
#ifndef SWIG
  // 构造函数，将 int 类型的 bottom 转换为 uint128，top 96 bits = 0
  UINT128_CONSTEXPR uint128(int bottom);
  // 构造函数，将 uint32_t 类型的 bottom 转换为 uint128，top 96 bits = 0
  UINT128_CONSTEXPR uint128(uint32_t bottom);
#endif
#endif
// 结束条件编译指令，用于关闭之前开启的条件编译块

UINT128_CONSTEXPR uint128(uint64_t bottom); // hi_ = 0
// 构造函数声明：使用底部 64 位初始化 uint128 对象，将高 64 位初始化为 0

UINT128_CONSTEXPR uint128(const uint128_pod& val);
// 构造函数声明：从 uint128_pod 类型的 val 对象初始化 uint128 对象

// Trivial copy constructor, assignment operator and destructor.

void Initialize(uint64_t top, uint64_t bottom);
// 初始化函数声明：用给定的 top 和 bottom 分别设置 uint128 对象的高 64 位和低 64 位

// Arithmetic operators.
uint128& operator+=(const uint128& b);
// 加法赋值运算符重载声明：将当前对象与另一个 uint128 对象 b 相加并赋值给当前对象

uint128& operator-=(const uint128& b);
// 减法赋值运算符重载声明：将当前对象减去另一个 uint128 对象 b 并赋值给当前对象

uint128& operator*=(const uint128& b);
// 乘法赋值运算符重载声明：将当前对象乘以另一个 uint128 对象 b 并赋值给当前对象

// Long division/modulo for uint128.
uint128& operator/=(const uint128& b);
// 除法赋值运算符重载声明：将当前对象除以另一个 uint128 对象 b 并赋值给当前对象

uint128& operator%=(const uint128& b);
// 取模赋值运算符重载声明：将当前对象对另一个 uint128 对象 b 取模并赋值给当前对象

uint128 operator++(int);
// 后置递增运算符重载声明：返回当前对象的副本，然后将当前对象加一

uint128 operator--(int);
// 后置递减运算符重载声明：返回当前对象的副本，然后将当前对象减一

// Make msvc happy with using operator<<= from DivModImpl
// which is a static function, and linker complained about missing
// static version of this overload
friend uint128& operator<<=(uint128&, int);
// 左移赋值运算符重载声明：友元函数，用于左移操作，帮助解决 MSVC 缺少静态版本的问题

uint128& operator>>=(int);
// 右移赋值运算符重载声明：将当前对象右移指定位数并赋值给当前对象

uint128& operator&=(const uint128& b);
// 按位与赋值运算符重载声明：将当前对象与另一个 uint128 对象 b 按位与并赋值给当前对象

uint128& operator|=(const uint128& b);
// 按位或赋值运算符重载声明：将当前对象与另一个 uint128 对象 b 按位或并赋值给当前对象

uint128& operator^=(const uint128& b);
// 按位异或赋值运算符重载声明：将当前对象与另一个 uint128 对象 b 按位异或并赋值给当前对象

uint128& operator++();
// 前置递增运算符重载声明：将当前对象加一并返回当前对象的引用

uint128& operator--();
// 前置递减运算符重载声明：将当前对象减一并返回当前对象的引用

friend uint64_t Uint128Low64(const uint128& v);
// 友元函数声明：返回 uint128 对象 v 的低 64 位值

friend uint64_t Uint128High64(const uint128& v);
// 友元函数声明：返回 uint128 对象 v 的高 64 位值

// We add "std::" to avoid including all of port.h.
C10_API friend std::ostream& operator<<(std::ostream& o, const uint128& b);
// 友元函数声明：重载输出流运算符，以便将 uint128 对象 b 输出到流 o 中

private:
static void DivModImpl(
    uint128 dividend,
    uint128 divisor,
    uint128* quotient_ret,
    uint128* remainder_ret);
// 静态私有方法声明：执行 uint128 的长除法和取模运算，计算商和余数并存储在指定的变量中

// Little-endian memory order optimizations can benefit from
// having lo_ first, hi_ last.
// See util/endian/endian.h and Load128/Store128 for storing a uint128.
uint64_t lo_;
uint64_t hi_;
// 私有成员变量声明：分别存储 uint128 对象的低 64 位和高 64 位

// Not implemented, just declared for catching automatic type conversions.
uint128(uint8_t);
uint128(uint16_t);
uint128(float v);
uint128(double v);
// 构造函数声明：禁止的自动类型转换，但未实现，仅用于捕获不正确的类型转换

};

// This is a POD form of uint128 which can be used for static variables which
// need to be operated on as uint128.
struct uint128_pod {
// Note: The ordering of fields is different than 'class uint128' but the
// same as its 2-arg constructor.  This enables more obvious initialization
// of static instances, which is the primary reason for this struct in the
// first place.  This does not seem to defeat any optimizations wrt
// operations involving this struct.
uint64_t hi;
uint64_t lo;
};
// POD 结构体声明：uint128 的 POD（平凡数据类型）形式，用于需要作为 uint128 操作的静态变量的场合

C10_API extern const uint128_pod kuint128max;
// 声明：外部链接的常量，表示最大的 uint128_pod 对象

// allow uint128 to be logged
C10_API extern std::ostream& operator<<(std::ostream& o, const uint128& b);
// 声明：允许 uint128 对象 b 被记录到输出流 o 中

// Methods to access low and high pieces of 128-bit value.
// Defined externally from uint128 to facilitate conversion
// to native 128-bit types when compilers support them.
inline uint64_t Uint128Low64(const uint128& v) {
return v.lo_;
}
// 内联函数定义：返回 uint128 对象 v 的低 64 位值

inline uint64_t Uint128High64(const uint128& v) {
return v.hi_;
}
// 内联函数定义：返回 uint128 对象 v 的高 64 位值

// TODO: perhaps it would be nice to have int128, a signed 128-bit type?

// --------------------------------------------------------------------------
// Implementation details follow
// --------------------------------------------------------------------------
// 定义比较运算符 ==，比较两个 uint128 类型对象是否相等
inline bool operator==(const uint128& lhs, const uint128& rhs) {
  // 比较两个 uint128 对象的低 64 位是否相等，以及高 64 位是否相等
  return (
      Uint128Low64(lhs) == Uint128Low64(rhs) &&
      Uint128High64(lhs) == Uint128High64(rhs));
}

// 定义比较运算符 !=，比较两个 uint128 类型对象是否不相等
inline bool operator!=(const uint128& lhs, const uint128& rhs) {
  // 利用已定义的 == 运算符判断两个对象是否相等，然后取反
  return !(lhs == rhs);
}

// 默认构造函数 uint128()，初始化为 0
C10_API inline UINT128_CONSTEXPR uint128::uint128() : lo_(0), hi_(0) {}

// 构造函数 uint128(top, bottom)，使用给定的 top 和 bottom 初始化对象
C10_API inline UINT128_CONSTEXPR uint128::uint128(uint64_t top, uint64_t bottom)
    : lo_(bottom), hi_(top) {}

// 构造函数 uint128(v)，使用 uint128_pod 结构的值 v 进行初始化
C10_API inline UINT128_CONSTEXPR uint128::uint128(const uint128_pod& v)
    : lo_(v.lo), hi_(v.hi) {}

// 构造函数 uint128(bottom)，使用一个 uint64_t 类型的 bottom 初始化对象
C10_API inline UINT128_CONSTEXPR uint128::uint128(uint64_t bottom)
    : lo_(bottom), hi_(0) {}

#ifndef SWIG
// 预处理器条件编译，构造函数 uint128(bottom)，使用一个 uint32_t 类型的 bottom 初始化对象
C10_API inline UINT128_CONSTEXPR uint128::uint128(uint32_t bottom)
    : lo_(bottom), hi_(0) {}

// 预处理器条件编译，构造函数 uint128(bottom)，使用一个 int 类型的 bottom 初始化对象
C10_API inline UINT128_CONSTEXPR uint128::uint128(int bottom)
    : lo_(bottom), hi_(static_cast<int64_t>((bottom < 0) ? -1 : 0)) {}
#endif

// 取消定义 UINT128_CONSTEXPR 宏

// 初始化函数 Initialize(top, bottom)，用给定的 top 和 bottom 设置对象的值
C10_API inline void uint128::Initialize(uint64_t top, uint64_t bottom) {
  hi_ = top;
  lo_ = bottom;
}

// 比较运算符的宏定义，用于生成 <、>、>=、<= 比较运算符
#define CMP128(op)                                                  \
  inline bool operator op(const uint128& lhs, const uint128& rhs) { \
    // 首先比较两个对象的高 64 位，如果相等，则比较低 64 位
    return (Uint128High64(lhs) == Uint128High64(rhs))               \
        ? (Uint128Low64(lhs) op Uint128Low64(rhs))                  \
        : (Uint128High64(lhs) op Uint128High64(rhs));               \
  }

// 生成 < 运算符
CMP128(<)
// 生成 > 运算符
CMP128(>)
// 生成 >= 运算符
CMP128(>=)
// 生成 <= 运算符
CMP128(<=)

// 取消定义 CMP128 宏

// 一元操作符 -，返回对象的补码
inline uint128 operator-(const uint128& val) {
  // 对 val 的高 64 位和低 64 位取反，然后低 64 位加 1
  const uint64_t hi_flip = ~Uint128High64(val);
  const uint64_t lo_flip = ~Uint128Low64(val);
  const uint64_t lo_add = lo_flip + 1;
  // 如果低 64 位加 1 后小于原来的低 64 位，则高 64 位加 1
  if (lo_add < lo_flip) {
    return uint128(hi_flip + 1, lo_add);
  }
  // 否则返回新的 uint128 对象
  return uint128(hi_flip, lo_add);
}

// 逻辑非操作符 !，判断对象是否为 0
inline bool operator!(const uint128& val) {
  // 如果对象的高 64 位和低 64 位都为 0，则返回 true，否则返回 false
  return !Uint128High64(val) && !Uint128Low64(val);
}

// 按位取反操作符 ~，对对象的每一位取反
inline uint128 operator~(const uint128& val) {
  // 对对象的高 64 位和低 64 位取反，返回新的 uint128 对象
  return uint128(~Uint128High64(val), ~Uint128Low64(val));
}

// 逻辑操作符的宏定义，用于生成 |、&、^ 逻辑运算符
#define LOGIC128(op)                                                   \
  inline uint128 operator op(const uint128& lhs, const uint128& rhs) { \
    // 分别对 lhs 和 rhs 的高 64 位和低 64 位进行 op 运算，并返回新的 uint128 对象
    return uint128(                                                    \
        Uint128High64(lhs) op Uint128High64(rhs),                      \
        Uint128Low64(lhs) op Uint128Low64(rhs));                       \
  }

// 生成 | 运算符
LOGIC128(|)
// 生成 & 运算符
LOGIC128(&)
// 生成 ^ 运算符
LOGIC128(^)

// 取消定义 LOGIC128 宏

// 逻辑赋值运算符的宏定义，用于生成 |=、&=、^= 逻辑赋值运算符
#define LOGICASSIGN128(op)                                              \
  C10_API inline uint128& uint128::operator op(const uint128 & other) { \
    // 对对象的高 64 位和低 64 位分别执行 op 运算，并返回对象的引用
    hi_ op other.hi_;                                                   \
    lo_ op other.lo_;                                                   \
    return *this;                                                       \
  }

// 生成 |= 运算符
LOGICASSIGN128(|=)
// 生成 &= 运算符
LOGICASSIGN128(&=)
// 生成 ^= 运算符
LOGICASSIGN128(^=)

// 取消定义 LOGICASSIGN128 宏

// 移位运算符
inline uint128 operator<<(const uint128& val, int amount) {
  // 如果移位量小于64，执行普通的位移操作
  if (amount < 64) {
    // 如果移位量为0，直接返回原值
    if (amount == 0) {
      return val;
    }
    // 计算新的高64位和低64位值
    uint64_t new_hi =
        (Uint128High64(val) << amount) | (Uint128Low64(val) >> (64 - amount));
    uint64_t new_lo = Uint128Low64(val) << amount;
    return uint128(new_hi, new_lo);
  } else if (amount < 128) {  // 如果移位量在64到127之间，高64位清零，低64位左移相应位数
    return uint128(Uint128Low64(val) << (amount - 64), 0);
  } else {  // 移位量超过127，结果为全零
    return uint128(0, 0);
  }
}

inline uint128 operator>>(const uint128& val, int amount) {
  // 如果移位量小于64，执行普通的位移操作
  if (amount < 64) {
    // 如果移位量为0，直接返回原值
    if (amount == 0) {
      return val;
    }
    // 计算新的高64位和低64位值
    uint64_t new_hi = Uint128High64(val) >> amount;
    uint64_t new_lo =
        (Uint128Low64(val) >> amount) | (Uint128High64(val) << (64 - amount));
    return uint128(new_hi, new_lo);
  } else if (amount < 128) {  // 如果移位量在64到127之间，低64位清零，高64位右移相应位数
    return uint128(0, Uint128High64(val) >> (amount - 64));
  } else {  // 移位量超过127，结果为全零
    return uint128(0, 0);
  }
}

inline uint128& operator<<=(uint128& self, int amount) {
  // 如果移位量小于64，执行普通的位移操作
  if (amount < 64) {
    // 如果移位量不为0，更新高64位和低64位的值
    if (amount != 0) {
      self.hi_ = (self.hi_ << amount) | (self.lo_ >> (64 - amount));
      self.lo_ = self.lo_ << amount;
    }
  } else if (amount < 128) {  // 如果移位量在64到127之间，高64位清零，低64位左移相应位数
    self.hi_ = self.lo_ << (amount - 64);
    self.lo_ = 0;
  } else {  // 移位量超过127，结果为全零
    self.hi_ = 0;
    self.lo_ = 0;
  }
  return self;
}

C10_API inline uint128& uint128::operator>>=(int amount) {
  // 如果移位量小于64，执行普通的位移操作
  if (amount < 64) {
    // 如果移位量不为0，更新低64位和高64位的值
    if (amount != 0) {
      lo_ = (lo_ >> amount) | (hi_ << (64 - amount));
      hi_ = hi_ >> amount;
    }
  } else if (amount < 128) {  // 如果移位量在64到127之间，低64位清零，高64位右移相应位数
    lo_ = hi_ >> (amount - 64);
    hi_ = 0;
  } else {  // 移位量超过127，结果为全零
    lo_ = 0;
    hi_ = 0;
  }
  return *this;
}

inline uint128 operator+(const uint128& lhs, const uint128& rhs) {
  // 返回两个uint128数值的和
  return uint128(lhs) += rhs;
}

inline uint128 operator-(const uint128& lhs, const uint128& rhs) {
  // 返回两个uint128数值的差
  return uint128(lhs) -= rhs;
}

inline uint128 operator*(const uint128& lhs, const uint128& rhs) {
  // 返回两个uint128数值的积
  return uint128(lhs) *= rhs;
}

inline uint128 operator/(const uint128& lhs, const uint128& rhs) {
  // 返回两个uint128数值的商
  return uint128(lhs) /= rhs;
}

inline uint128 operator%(const uint128& lhs, const uint128& rhs) {
  // 返回两个uint128数值的模
  return uint128(lhs) %= rhs;
}

C10_API inline uint128& uint128::operator+=(const uint128& b) {
  // 计算并更新高位和低位的和，处理进位
  hi_ += b.hi_;
  uint64_t lolo = lo_ + b.lo_;
  if (lolo < lo_)
    ++hi_;
  lo_ = lolo;
  return *this;
}

C10_API inline uint128& uint128::operator-=(const uint128& b) {
  // 计算并更新高位和低位的差，处理借位
  hi_ -= b.hi_;
  if (b.lo_ > lo_)
    --hi_;
  lo_ -= b.lo_;
  return *this;
}
// 实现 uint128 类的乘法赋值运算符 *=
C10_API inline uint128& uint128::operator*=(const uint128& b) {
    // 将当前对象的高 64 位分解为两个 32 位部分
    uint64_t a96 = hi_ >> 32;
    uint64_t a64 = hi_ & 0xffffffffu;
    // 将当前对象的低 64 位分解为两个 32 位部分
    uint64_t a32 = lo_ >> 32;
    uint64_t a00 = lo_ & 0xffffffffu;
    // 将参数对象的高 64 位分解为两个 32 位部分
    uint64_t b96 = b.hi_ >> 32;
    uint64_t b64 = b.hi_ & 0xffffffffu;
    // 将参数对象的低 64 位分解为两个 32 位部分
    uint64_t b32 = b.lo_ >> 32;
    uint64_t b00 = b.lo_ & 0xffffffffu;
    
    // 计算乘法结果的高 64 位，可能会产生进位
    uint64_t c96 = a96 * b00 + a64 * b32 + a32 * b64 + a00 * b96;
    // 计算乘法结果的次高 64 位，忽略进位
    uint64_t c64 = a64 * b00 + a32 * b32 + a00 * b64;
    
    // 将结果赋值给当前对象的高 128 位
    this->hi_ = (c96 << 32) + c64;
    // 将当前对象的低 128 位清零
    this->lo_ = 0;
    
    // 逐个添加乘法结果的剩余部分，以捕获进位
    *this += uint128(a32 * b00) << 32;
    *this += uint128(a00 * b32) << 32;
    *this += a00 * b00;
    
    // 返回乘法赋值后的当前对象引用
    return *this;
}

// 实现 uint128 类的后置自增运算符 ++
C10_API inline uint128 uint128::operator++(int) {
    // 复制当前对象
    uint128 tmp(*this);
    // 使用前置自增运算符增加当前对象
    *this += 1;
    // 返回复制的对象
    return tmp;
}

// 实现 uint128 类的后置自减运算符 --
C10_API inline uint128 uint128::operator--(int) {
    // 复制当前对象
    uint128 tmp(*this);
    // 使用前置自减运算符减少当前对象
    *this -= 1;
    // 返回复制的对象
    return tmp;
}

// 实现 uint128 类的前置自增运算符 ++
C10_API inline uint128& uint128::operator++() {
    // 使用前置自增运算符增加当前对象
    *this += 1;
    // 返回当前对象引用
    return *this;
}

// 实现 uint128 类的前置自减运算符 --
C10_API inline uint128& uint128::operator--() {
    // 使用前置自减运算符减少当前对象
    *this -= 1;
    // 返回当前对象引用
    return *this;
}
```