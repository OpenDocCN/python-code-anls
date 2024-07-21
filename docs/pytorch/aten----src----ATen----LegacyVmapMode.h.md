# `.\pytorch\aten\src\ATen\LegacyVmapMode.h`

```
#pragma once
// 预处理指令：确保头文件只被包含一次

#include <c10/core/impl/LocalDispatchKeySet.h>
// 包含本地调度键集头文件

namespace at::impl {
// 进入 at::impl 命名空间

// VmapMode 结构体定义
// 包含一个线程局部变量，用于记录当前嵌套的 vmap 层数（称为 vmap 级别）。
// VmapMode 在实现 Python 的 `torch.vmap` API 时使用。

struct TORCH_API VmapMode {
  // 静态方法：返回当前 vmap 级别，即当前嵌套 vmap 的数量。
  static int64_t current_vmap_level();

  // 静态方法：增加嵌套的 vmap 数量。
  // 如果这导致 vmap 级别大于 0，则在所有张量上启用 DispatchKey::VmapMode。
  static int64_t increment_nesting();

  // 静态方法：减少嵌套的 vmap 数量。
  // 如果这导致 vmap 级别等于 0，则在所有张量上禁用 DispatchKey::VmapMode。
  static int64_t decrement_nesting();
};

} // namespace at::impl
// 结束 at::impl 命名空间
```