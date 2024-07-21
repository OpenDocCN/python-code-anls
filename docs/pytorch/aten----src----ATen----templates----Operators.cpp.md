# `.\pytorch\aten\src\ATen\templates\Operators.cpp`

```
// 包含 ATen 库的 Tensor 类和 Dispatcher 类的头文件
#include <ATen/Tensor.h>
#include <ATen/core/dispatch/Dispatcher.h>

// ${generated_comment}
// NOTE See [Sharded File] comment in VariableType
// 自动生成的注释内容，可能是由生成工具自动生成的说明
// 关于 VariableType 的 [Sharded File] 注释，请参见相应文档

// 如果未定义 AT_PER_OPERATOR_HEADERS 宏，则包含 Operators.h 头文件
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Operators.h>
// 如果定义了 AT_PER_OPERATOR_HEADERS 宏，则插入 operator_headers 变量所指定的头文件内容
#else
${operator_headers}
#endif

${static_dispatch_extra_headers}
// 命名空间开始：at::_ops
namespace at { namespace _ops {

${definitions}
// 命名空间结束：at::_ops

}} // namespace at::_ops
// 命名空间结束：at
```