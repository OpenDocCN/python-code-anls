# `.\pytorch\aten\src\ATen\templates\Operators.h`

```py
#pragma once

// ${generated_comment}  // 插入生成的注释

#ifdef TORCH_ASSERT_NO_OPERATORS
#error This change adds a dependency on native_functions.yaml,             \
  meaning the file will need to be re-compiled every time an operator      \
  is changed or added. Consider if your change would be better placed in   \
  another file, or if a more specific header might achieve the same goal.  \
  See NOTE: [Tensor vs. TensorBase]
#endif

#if defined(AT_PER_OPERATOR_HEADERS) && defined(TORCH_ASSERT_ONLY_METHOD_OPERATORS)
#error This change adds a dependency on all pytorch operators, meaning the     \
  file will need to be re-compiled every time an operator is changed or added. \
  Consider including a specific operator from <ATen/ops/{my_operator}_ops.h>   \
  and see NOTE [TORCH_ASSERT_ONLY_METHOD_OPERATORS].
#endif

#include <c10/core/SymInt.h>               // 包含对称整数定义
#include <c10/core/SymIntArrayRef.h>       // 包含对称整数数组引用定义
#include <c10/core/Scalar.h>               // 包含标量定义
#include <c10/core/TensorOptions.h>        // 包含张量选项定义
#include <c10/core/QScheme.h>              // 包含量化方案定义
#include <c10/util/OptionalArrayRef.h>     // 包含可选数组引用定义
#include <tuple>                           // 包含元组定义
#include <vector>                          // 包含向量定义

${Operators_includes}                     // 插入运算符的包含文件

// Extension writers: do you write wrapper functions? Are you frustrated with
// resolving overloads of operators? Are you frustrated with dealing with
// pointer-to-methods and resolving overloads of pointer-to-methods?? Look no
// further, this is the utility for you.
//
// Given an operator schema: aten::op.overload(...
//
// Use ATEN_FN2(op, overload) to get a *function* version of the operator
// that is guaranteed to not be overloaded. This means that you can safely
// decltype(&ATEN_FN2(op, overload)) it. NB: the 2 means this macro takes 2 args.
//
// Given an operator schema without an overload name: aten::op(...
//
// Use ATEN_FN(op) to get an unambiguous *function* version of the operator.
//
// There is some interesting behavior for out= operations.
// ATEN_FN2(sin, out) gives a function that is *faithful* to the schema;
// that is, the order of arguments is exactly what it looks like in the schema.

#define ATEN_FN2(op_name, overload) at::_ops::op_name##_##overload::call   // 定义获取特定操作和重载的函数版本
#define ATEN_FN(op_name) at::_ops::op_name::call                          // 定义获取特定操作的函数版本

// Separately, ATEN_OP(op) and ATEN_OP2(op, overload) define a class containing compile-time
// metadata about a given aten operator.
// Notable data on the class includes:
// - ATEN_OP2(add, Tensor)::name // returns the string name: "add"
// - ATEN_OP2(add, Tensor)::overload_name // returns the string overload name: "Tensor"
// - ATEN_OP2(add, Tensor)::schema // returns the C++ schema type: at::Tensor (const at::Tensor &, const at::Tensor &, const at::Scalar &)
// - ATEN_OP2(add, Tensor)::schema_str // returns the string jit type: "add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor"

#define ATEN_OP2(op_name, overload) at::_ops::op_name##_##overload        // 定义获取特定操作和重载的元数据类
#define ATEN_OP(op_name) at::_ops::op_name                                // 定义获取特定操作的元数据类

// WARNING: Please do not call any of the ops in the _ops namespace directly.
// Use the ATEN_FN macros. We do not guarantee stability of the naming
// 定义了 at::_ops 命名空间中的函数的结构

// 详细参见注释 [The ATen Operators API]，了解 at::_ops 命名空间的API细节

// 在 at 命名空间内声明了 _ops 命名空间，用于存放 ATen 操作符的函数声明

namespace at {
namespace _ops {
${Operators_declarations}  // ${Operators_declarations} 是一个占位符，用于插入实际的操作符声明
} // namespace _ops
} // namespace at
```