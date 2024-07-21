# `.\pytorch\aten\src\ATen\functorch\BatchRulesIndexing.cpp`

```
// 版权声明和许可信息
// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

// 包含必要的头文件
#include <ATen/Operators.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/functorch/BatchRulesHelper.h>

// 定义 at 命名空间下的 functorch 命名空间
namespace at { namespace functorch {

// 定义宏 OP_DECOMPOSE，用于注册操作和其对应的函数指针
#define OP_DECOMPOSE(op)  m.impl(#op, static_cast<decltype(&ATEN_FN(op))>(native::op));
// 定义宏 OP_DECOMPOSE2，用于注册具有重载版本的操作和其对应的函数指针
#define OP_DECOMPOSE2(op, overload)  m.impl(#op"."#overload, static_cast<decltype(&ATEN_FN2(op, overload))>(native::op));

// 定义 ATen 库中 aten 命名空间下的 FuncTorchBatched 库的实现
TORCH_LIBRARY_IMPL(aten, FuncTorchBatched, m) {
  // 使用 OP_DECOMPOSE2 宏注册 _unsafe_index 操作的 Tensor 重载版本
  OP_DECOMPOSE2(_unsafe_index, Tensor);
  // 使用 OP_DECOMPOSE 宏注册 _unsafe_masked_index 操作
  OP_DECOMPOSE(_unsafe_masked_index);
  // 使用 OP_DECOMPOSE 宏注册 _unsafe_index_put 操作
  OP_DECOMPOSE(_unsafe_index_put);
  // 使用 OP_DECOMPOSE 宏注册 _unsafe_masked_index_put_accumulate 操作
  OP_DECOMPOSE(_unsafe_masked_index_put_accumulate);
}

}} // namespace at::functorch
```