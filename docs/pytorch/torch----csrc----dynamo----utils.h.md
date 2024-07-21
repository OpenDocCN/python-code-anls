# `.\pytorch\torch\csrc\dynamo\utils.h`

```
// 预处理指令：定义了一个条件编译宏，用于在不同操作系统下设置符号的可见性
#pragma once

// 根据操作系统设置宏VISIBILITY_HIDDEN，用于处理存储在结构体中的字段的可见性警告
// 在 Windows 下，VISIBILITY_HIDDEN 被定义为空
#ifdef _WIN32
#define VISIBILITY_HIDDEN
// 在非 Windows 系统下，使用__attribute__((visibility("hidden")))设置字段隐藏属性
#else
#define VISIBILITY_HIDDEN __attribute__((visibility("hidden")))
#endif
```