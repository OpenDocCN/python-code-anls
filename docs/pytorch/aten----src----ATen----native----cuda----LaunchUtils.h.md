# `.\pytorch\aten\src\ATen\native\cuda\LaunchUtils.h`

```
// 声明宏，确保头文件仅被编译一次
#pragma once

// 包含算法标准库，以便使用 std::max
#include<algorithm>

// 定义 at 命名空间
namespace at {
namespace native {

// 返回不大于 n 的最大的 2 的幂
static int lastPow2(unsigned int n) {
  // n 与右移 1 位后的 n 按位或运算，确保 n 及其右侧的所有位都被设置为 1
  n |= (n >> 1);
  n |= (n >> 2);
  n |= (n >> 4);
  n |= (n >> 8);
  n |= (n >> 16);
  // 返回 1 和 n 与其右移 1 位后的差值的最大值
  return std::max<int>(1, n - (n >> 1));
}

} // namespace native
} // namespace at
```