# `.\pytorch\aten\src\ATen\test\weakref_test.cpp`

```
// 包含 Google Test 框架的头文件
#include <gtest/gtest.h>

// 包含 PyTorch C++ API 的头文件
#include <ATen/ATen.h>
#include <ATen/core/ivalue.h>

// 包含标准输入输出流的头文件
#include <iostream>
// 包含时间处理相关的头文件
#include <chrono>
// 包含字符串流的头文件
#include <sstream>

// 使用声明：定义 Tensor 类型别名
using at::Tensor;
// 使用声明：定义 WeakIValue 类型别名
using c10::WeakIValue;
// 使用声明：定义 IValue 类型别名
using c10::IValue;

// 弱指针测试
// 当 IValue 被无效化时，弱指针失效
TEST(TestWeakPointer, WeakPointerGetsInvalidated) {
  // 创建一个包含全为 1 的 2x2 Tensor
  IValue a = at::ones({2, 2});
  // 创建指向 a 的弱指针 b
  WeakIValue b = a;
  // 清空 a
  a = IValue();
  // 断言 b 通过 lock 后是 None
  ASSERT_TRUE(b.lock().isNone());
}

// 成功进行弱指针加锁操作
TEST(TestWeakPointer, WeakPointerLock) {
  // 创建一个包含全为 1 的 2x2 Tensor
  IValue a = at::ones({2, 2});
  // 创建指向 a 的弱指针 b
  WeakIValue b = a;
  // 使用 lock 方法加锁并赋值给 c
  auto c = b.lock();
  // 断言 c 是一个 Tensor
  ASSERT_TRUE(c.isTensor());

  // 清空 a
  a = IValue();
  // 断言 b 通过 lock 后不是 None
  ASSERT_TRUE(!b.lock().isNone());
  // 再次清空 c
  c = IValue();
  // 断言 b 通过 lock 后是 None
  ASSERT_TRUE(b.lock().isNone());
}

// 正确更新引用计数
TEST(TestWeakPointer, WeakUpdatesRefcountsTest) {
  // 创建一个包含全为 1 的 2x2 Tensor
  at::Tensor a = at::ones({2, 2});
  // 断言 a 的引用计数为 1，弱引用计数也为 1
  ASSERT_EQ(a.use_count(), 1);
  ASSERT_EQ(a.weak_use_count(), 1);

  {
    // 创建一个通过 IValue 包装的弱指针 b
    WeakIValue b = IValue(a);
    // 断言 a 的引用计数为 1，弱引用计数增加到 2
    ASSERT_EQ(a.use_count(), 1);
    ASSERT_EQ(a.weak_use_count(), 2);
  }

  // 再次断言 a 的引用计数为 1，弱引用计数为 1
  ASSERT_EQ(a.use_count(), 1);
  ASSERT_EQ(a.weak_use_count(), 1);

  {
    // 创建一个通过 IValue 包装的弱指针 b
    WeakIValue b = IValue(a);
    // 断言 a 的引用计数为 1
    ASSERT_EQ(a.use_count(), 1);
    // 锁定 b 并断言不是 None
    auto locked = b.lock();
    ASSERT_FALSE(locked.isNone());
    // 断言 a 的引用计数增加到 2
    ASSERT_EQ(a.use_count(), 2);
  }

  // 再次断言 a 的引用计数为 1，弱引用计数为 1
  ASSERT_EQ(a.use_count(), 1);
  ASSERT_EQ(a.weak_use_count(), 1);

  {
    // 创建一个通过 IValue 包装的弱指针 b
    WeakIValue b = IValue(a);
    // 断言 a 的引用计数为 1，弱引用计数增加到 2
    ASSERT_EQ(a.use_count(), 1);
    ASSERT_EQ(a.weak_use_count(), 2);
    // 重置 a
    a.reset();
    // 断言 b 的引用计数为 0，弱引用计数为 1
    ASSERT_EQ(b.use_count(), 0);
    ASSERT_EQ(b.weak_use_count(), 1);
  }
}
```