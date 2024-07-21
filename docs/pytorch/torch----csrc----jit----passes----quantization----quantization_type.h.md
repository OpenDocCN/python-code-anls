# `.\pytorch\torch\csrc\jit\passes\quantization\quantization_type.h`

```py
#pragma once
// 预处理指令，确保头文件只被编译一次

#include <cstdint>
// 包含 C++ 标准库中的 cstdint 头文件，定义了标准整数类型

#include <ostream>
// 包含 C++ 标准库中的 ostream 头文件，提供了输出流类

namespace torch {
namespace jit {

// torch::jit 命名空间，用于封装 JIT 相关的功能和数据结构

// Quantization type (dynamic quantization, static quantization).
// Should match the Python enum in quantize_jit.py
// 枚举类型 QuantType 表示量化类型（动态量化，静态量化），
// 应与 quantize_jit.py 中的 Python 枚举保持一致
enum QuantType : std::uint8_t { DYNAMIC = 0, STATIC };
// 定义一个枚举类型 QuantType，基础类型为 std::uint8_t，包括 DYNAMIC 和 STATIC 两个枚举值

std::ostream& operator<<(std::ostream& os, QuantType t);
// 重载流插入运算符 << ，用于将 QuantType 类型对象 t 输出到 ostream 对象 os 中

} // namespace jit
} // namespace torch
// 结束 torch::jit 命名空间和 torch 命名空间
```