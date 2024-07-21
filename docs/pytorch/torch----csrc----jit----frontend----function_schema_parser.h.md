# `.\pytorch\torch\csrc\jit\frontend\function_schema_parser.h`

```py
#pragma once

#include <ATen/core/function_schema.h>
#include <c10/macros/Macros.h>
#include <string>
#include <variant>

namespace torch {
namespace jit {

// 如果 allow_typevars 为 true，则假设我们不理解的小写类型是类型变量。这仅适用于 TorchScript（而不适用于自定义操作）。
// 如果为 false，则除了某些情况下为了向后兼容性（即您的操作位于 aten 或 prim 命名空间中）之外，我们禁止使用类型变量。
TORCH_API std::variant<c10::OperatorName, c10::FunctionSchema> parseSchemaOrName(
    const std::string& schemaOrName,
    bool allow_typevars = true);
TORCH_API c10::FunctionSchema parseSchema(
    const std::string& schema,
    bool allow_typevars = true);
TORCH_API c10::OperatorName parseName(const std::string& name);

} // namespace jit
} // namespace torch
```