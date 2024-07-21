# `.\pytorch\c10\test\util\Synchronized_test.cpp`

```
// 包含C++的头文件：c10/util/Synchronized.h，用于多线程同步操作
#include <c10/util/Synchronized.h>
// 包含 Google Test 的头文件，用于编写和运行测试
#include <gtest/gtest.h>

// 匿名命名空间，用于定义测试用例
namespace {

// 定义单线程执行的测试用例 Synchronized.TestSingleThreadExecution
TEST(Synchronized, TestSingleThreadExecution) {
  // 创建一个同步整数对象 iv，初始值为 0
  c10::Synchronized<int> iv(0);
  // 定义常量 kMaxValue，值为 100
  const int kMaxValue = 100;
  // 循环执行 0 到 kMaxValue-1
  for (int i = 0; i < kMaxValue; ++i) {
    // 在锁定状态下执行 lambda 表达式，递增 iv 的值，并返回递增后的值
    auto ret = iv.withLock([](int& iv) { return ++iv; });
    // 断言 ret 的值等于 i + 1
    EXPECT_EQ(ret, i + 1);
  }

  // 在锁定状态下执行 lambda 表达式，验证 iv 的值是否等于 kMaxValue
  iv.withLock([kMaxValue](int& iv) { EXPECT_EQ(iv, kMaxValue); });
}

// 定义多线程执行的测试用例 Synchronized.TestMultiThreadedExecution
TEST(Synchronized, TestMultiThreadedExecution) {
  // 创建一个同步整数对象 iv，初始值为 0
  c10::Synchronized<int> iv(0);
  // 定义常量 NUM_LOOP_INCREMENTS，值为 10000
#define NUM_LOOP_INCREMENTS 10000

  // 定义线程执行的回调函数 thread_cb
  auto thread_cb = [&iv]() {
    // 循环执行 NUM_LOOP_INCREMENTS 次
    for (int i = 0; i < NUM_LOOP_INCREMENTS; ++i) {
      // 在锁定状态下执行 lambda 表达式，递增 iv 的值
      iv.withLock([](int& iv) { ++iv; });
    }
  };

  // 创建包含 10 个线程对象的数组 threads
  std::array<std::thread, 10> threads;
  // 循环遍历线程数组，为每个线程分配线程执行的回调函数
  for (auto& t : threads) {
    t = std::thread(thread_cb);
  }

  // 循环遍历线程数组，等待每个线程执行结束
  for (auto& t : threads) {
    t.join();
  }

  // 在锁定状态下执行 lambda 表达式，验证 iv 的值是否等于 NUM_LOOP_INCREMENTS * 10
  iv.withLock([](int& iv) { EXPECT_EQ(iv, NUM_LOOP_INCREMENTS * 10); });
#undef NUM_LOOP_INCREMENTS
}

} // namespace
```