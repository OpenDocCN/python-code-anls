# `.\pytorch\aten\src\ATen\templates\NativeFunction.h`

```
#pragma once
// 声明指令，指示编译器仅包含本文件一次

// ${generated_comment}
// 自动生成的注释内容，可能包含文件生成相关的信息

#include <c10/core/Scalar.h>
// 包含标量类型相关的头文件

#include <c10/core/Storage.h>
// 包含存储相关的头文件

#include <c10/core/TensorOptions.h>
// 包含张量选项相关的头文件

#include <c10/util/Deprecated.h>
// 包含已废弃特性相关的头文件

#include <c10/util/Optional.h>
// 包含可选值相关的头文件

#include <c10/core/QScheme.h>
// 包含量化方案相关的头文件

#include <ATen/core/Reduction.h>
// 包含张量缩减相关的头文件

#include <ATen/core/Tensor.h>
// 包含张量操作相关的头文件

#include <tuple>
// 包含元组相关的头文件

#include <vector>
// 包含向量（数组）相关的头文件

${extra_includes}
// 可能包含额外用户定义的头文件，根据生成的具体内容

${native_function_declarations}
// 可能包含本地（native）函数声明，根据生成的具体内容
```