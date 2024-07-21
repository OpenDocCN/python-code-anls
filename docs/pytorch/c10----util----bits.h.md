# `.\pytorch\c10\util\bits.h`

```py
#pragma once
#include <cstdint>  // 包含标准整数类型的头文件

#include <c10/macros/Macros.h>  // 包含 C10 宏定义的头文件

namespace c10 {

/**
 * bits1x8 is an uninterpreted dtype of a tensor with 1 bit (packed to byte
 * boundary), without any semantics defined.
 */
struct alignas(1) bits1x8 {
  using underlying = uint8_t;  // 定义底层类型为 uint8_t
  uint8_t val_;  // 存储 1 位数据的 uint8_t 类型变量
  bits1x8() = default;  // 默认构造函数
  C10_HOST_DEVICE explicit bits1x8(uint8_t val) : val_(val) {}  // 显式构造函数，接受一个 uint8_t 类型参数
};

/**
 * bits2x4 is an uninterpreted dtype of a tensor with 2 bits (packed to byte
 * boundary), without any semantics defined.
 */
struct alignas(1) bits2x4 {
  using underlying = uint8_t;  // 定义底层类型为 uint8_t
  uint8_t val_;  // 存储 2 位数据的 uint8_t 类型变量
  bits2x4() = default;  // 默认构造函数
  C10_HOST_DEVICE explicit bits2x4(uint8_t val) : val_(val) {}  // 显式构造函数，接受一个 uint8_t 类型参数
};

/**
 * bits4x2 is an uninterpreted dtype of a tensor with 4 bits (packed to byte
 * boundary), without any semantics defined.
 */
struct alignas(1) bits4x2 {
  using underlying = uint8_t;  // 定义底层类型为 uint8_t
  uint8_t val_;  // 存储 4 位数据的 uint8_t 类型变量
  bits4x2() = default;  // 默认构造函数
  C10_HOST_DEVICE explicit bits4x2(uint8_t val) : val_(val) {}  // 显式构造函数，接受一个 uint8_t 类型参数
};

/**
 * bits8 is an uninterpreted dtype of a tensor with 8 bits, without any
 * semantics defined.
 */
struct alignas(1) bits8 {
  uint8_t val_;  // 存储 8 位数据的 uint8_t 类型变量
  bits8() = default;  // 默认构造函数
  C10_HOST_DEVICE explicit bits8(uint8_t val) : val_(val) {}  // 显式构造函数，接受一个 uint8_t 类型参数
};

/**
 * bits16 is an uninterpreted dtype of a tensor with 16 bits, without any
 * semantics defined.
 */
struct alignas(2) bits16 {
  uint16_t val_;  // 存储 16 位数据的 uint16_t 类型变量
  bits16() = default;  // 默认构造函数
  C10_HOST_DEVICE explicit bits16(uint16_t val) : val_(val) {}  // 显式构造函数，接受一个 uint16_t 类型参数
};

} // namespace c10
```