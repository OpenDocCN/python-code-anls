# `.\pytorch\c10\test\util\DeadlockDetection_test.cpp`

```py
#include <c10/util/DeadlockDetection.h>  // 包含 DeadlockDetection.h 头文件，提供死锁检测相关功能

#include <gtest/gtest.h>  // 包含 Google Test 框架的头文件

#include <cstdlib>  // 包含标准库的头文件，提供对环境变量的操作函数

using namespace ::testing;  // 使用 testing 命名空间，使得测试框架的符号可见
using namespace c10::impl;  // 使用 c10::impl 命名空间，访问实现细节

struct DummyPythonGILHooks : public PythonGILHooks {
  bool check_python_gil() const override {
    return true;  // 覆盖虚函数，始终返回 true，模拟 Python GIL 的检测函数
  }
};

TEST(DeadlockDetection, basic) {  // 定义测试用例 DeadlockDetection.basic
  ASSERT_FALSE(check_python_gil());  // 断言 Python GIL 不被持有
  DummyPythonGILHooks hooks;  // 创建 DummyPythonGILHooks 对象 hooks
  SetPythonGILHooks(&hooks);  // 设置 Python GIL 钩子为 hooks 对象
  ASSERT_TRUE(check_python_gil());  // 断言 Python GIL 被持有
  SetPythonGILHooks(nullptr);  // 恢复 Python GIL 钩子为空指针
}

#ifndef _WIN32
TEST(DeadlockDetection, disable) {  // 定义测试用例 DeadlockDetection.disable（仅限非 Windows 平台）
  setenv("TORCH_DISABLE_DEADLOCK_DETECTION", "1", 1);  // 设置环境变量 TORCH_DISABLE_DEADLOCK_DETECTION 为 "1"
  DummyPythonGILHooks hooks;  // 创建 DummyPythonGILHooks 对象 hooks
  SetPythonGILHooks(&hooks);  // 设置 Python GIL 钩子为 hooks 对象
  SetPythonGILHooks(&hooks);  // 再次设置 Python GIL 钩子为 hooks 对象
}
#endif
```