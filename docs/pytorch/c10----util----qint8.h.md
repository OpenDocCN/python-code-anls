# `.\pytorch\c10\util\qint8.h`

```py
#pragma once
#include <cstdint> // 包含 C++ 标准库中的cstdint，定义了标准整数类型
#include <c10/macros/Macros.h> // 包含 c10 库中的宏定义

namespace c10 {

/**
 * This is the data type for quantized Tensors. Right now we only have
 * qint8 which is for 8 bit Tensors, and qint32 for 32 bit int Tensors,
 * we might have 4 bit, 2 bit or 1 bit data types in the future.
 */
struct alignas(1) qint8 {
  using underlying = int8_t; // 定义了 underlying 别名为 int8_t，表示 qint8 的基础类型为 int8_t
  int8_t val_; // qint8 结构体中的成员变量，存储 8 位整数值
  qint8() = default; // 默认构造函数，使用默认方式初始化 qint8 对象
  C10_HOST_DEVICE explicit qint8(int8_t val) : val_(val) {} // 显式构造函数，用于直接赋值初始化 qint8 对象的 val_ 成员
};

} // namespace c10
```