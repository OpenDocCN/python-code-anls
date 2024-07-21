# `.\pytorch\aten\src\ATen\test\dispatch_key_set_test.cpp`

```py
#include <gtest/gtest.h>  // 包含 Google Test 框架的头文件

#include <ATen/ATen.h>  // 包含 PyTorch ATen 库的头文件
#include <c10/core/DispatchKeySet.h>  // 包含 ATen 中 DispatchKeySet 相关的头文件

#include <vector>  // 包含标准库中的 vector 头文件

using at::DispatchKey;  // 使用 ATen 命名空间中的 DispatchKey 类型
using at::DispatchKeySet;  // 使用 ATen 命名空间中的 DispatchKeySet 类型

TEST(DispatchKeySetTest, TestGetRuntimeDispatchKeySet) {
  // 对于 DispatchKeySet(DispatchKeySet::FULL) 中的每一个 DispatchKey dk1 进行迭代
  for (auto dk1: DispatchKeySet(DispatchKeySet::FULL)) {
    // 调用 getRuntimeDispatchKeySet 函数获取 dk1 对应的运行时 DispatchKeySet
    auto dks = getRuntimeDispatchKeySet(dk1);
    // 对于 DispatchKeySet(DispatchKeySet::FULL) 中的每一个 DispatchKey dk2 进行迭代
    for (auto dk2: DispatchKeySet(DispatchKeySet::FULL)) {
      // 使用 ASSERT_EQ 断言：验证 dks 中是否包含 dk2，并与 runtimeDispatchKeySetHas(dk1, dk2) 的返回值进行比较
      ASSERT_EQ(dks.has(dk2), runtimeDispatchKeySetHas(dk1, dk2));
    }
  }
}
```