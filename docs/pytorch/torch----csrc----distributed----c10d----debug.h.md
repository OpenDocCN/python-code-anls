# `.\pytorch\torch\csrc\distributed\c10d\debug.h`

```
// Copyright (c) Meta Platforms, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <c10/macros/Macros.h>  // 包含 C10 库的宏定义

namespace c10d {

enum class DebugLevel { Off = 0, Info = 1, Detail = 2 };  // 定义枚举类型 DebugLevel，表示调试级别

TORCH_API void setDebugLevel(DebugLevel level);  // 设置全局调试级别的函数声明

// Sets the debug level based on the value of the `TORCH_DISTRIBUTED_DEBUG`
// environment variable.
TORCH_API void setDebugLevelFromEnvironment();  // 根据环境变量 `TORCH_DISTRIBUTED_DEBUG` 设置调试级别的函数声明

TORCH_API DebugLevel debug_level() noexcept;  // 获取当前调试级别的函数声明

} // namespace c10d
```