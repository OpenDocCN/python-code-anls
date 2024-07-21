# `.\pytorch\test\edge\templates\RegisterKernels.h`

```
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// ${generated_comment}
// 以上是版权声明和许可信息

// 包含必要的头文件，用于注册所有内核
#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/kernel/operator_registry.h>
#include <executorch/runtime/platform/profiler.h>

// 命名空间定义：torch::executor
namespace torch {
namespace executor {

// 声明一个函数 register_all_kernels，用于注册所有的内核
Error register_all_kernels();

} // namespace executor
} // namespace torch
```