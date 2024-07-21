# `.\pytorch\torch\csrc\jit\serialization\import_export_constants.h`

```
#pragma once
#include <cstddef>

namespace torch::jit {
// 指令字节码的索引位置
constexpr size_t BYTECODE_INDEX_INSTRUCTION = 0;
// 运算符字节码的索引位置
constexpr size_t BYTECODE_INDEX_OPERATOR = 1;
// 常量字节码的索引位置
constexpr size_t BYTECODE_INDEX_CONSTANT = 2;
// 类型字节码的索引位置
constexpr size_t BYTECODE_INDEX_TYPE = 3;
// 寄存器大小字节码的索引位置
constexpr size_t BYTECODE_INDEX_REGISTER_SIZE = 4;

// 模块模式中模式架构参数的索引位置
constexpr size_t BYTECODE_INDEX_SCHEMA_ARGUMENTS = 0;
// 模块模式中模式架构返回值的索引位置
constexpr size_t BYTECODE_INDEX_SCHEMA_RETURNS = 1;

// 模块模式中参数的名称的索引位置
constexpr size_t BYTECODE_INDEX_ARGUMENT_NAME = 0;
// 模块模式中参数的类型的索引位置
constexpr size_t BYTECODE_INDEX_ARGUMENT_TYPE = 1;
// 模块模式中参数的默认值的索引位置
constexpr size_t BYTECODE_INDEX_ARGUMENT_DEFAULT_VALUE = 2;

// 调试处理模块中的调试句柄的索引位置
constexpr size_t BYTECODE_INDEX_MODULE_DEBUG_HANDLES = 0;
} // namespace torch::jit
```