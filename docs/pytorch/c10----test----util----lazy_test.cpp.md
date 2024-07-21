# `.\pytorch\c10\test\util\lazy_test.cpp`

```py
#include <atomic>  // 引入原子操作相关的头文件
#include <thread>  // 引入线程相关的头文件
#include <vector>  // 引入向量容器相关的头文件

#include <c10/util/Lazy.h>  // 引入Lazy相关的头文件
#include <gtest/gtest.h>    // 引入Google测试框架的头文件

namespace c10_test {

// Long enough not to fit in typical SSO.
const std::string kLongString = "I am a long enough string";  // 定义一个足够长的字符串常量

TEST(LazyTest, OptimisticLazy) {
  std::atomic<size_t> invocations = 0;  // 定义一个原子类型的计数器，记录调用次数
  auto factory = [&] {
    ++invocations;  // 工厂函数，每次调用增加调用计数
    return kLongString;  // 返回预定义的长字符串
  };

  c10::OptimisticLazy<std::string> s;  // 创建一个OptimisticLazy对象，存储字符串

  constexpr size_t kNumThreads = 16;  // 定义线程数量
  std::vector<std::thread> threads;   // 创建线程容器
  std::atomic<std::string*> address = nullptr;  // 原子指针，存储地址

  for (size_t i = 0; i < kNumThreads; ++i) {
    threads.emplace_back([&] {
      auto* p = &s.ensure(factory);  // 获取确保对象的指针
      auto old = address.exchange(p);  // 原子操作，交换地址，保证线程安全性
      if (old != nullptr) {
        // Even racing ensure()s should return a stable reference.
        EXPECT_EQ(old, p);  // 断言：即使是竞争条件下的ensure()也应返回稳定的引用
      }
    });
  }

  for (auto& t : threads) {
    t.join();  // 等待线程结束
  }

  EXPECT_GE(invocations.load(), 1);  // 断言：工厂函数至少被调用一次
  EXPECT_EQ(*address.load(), kLongString);  // 断言：地址中存储的字符串与预定义字符串相等

  invocations = 0;  // 重置调用计数
  s.reset();  // 重置OptimisticLazy对象
  s.ensure(factory);  // 确保对象中包含工厂函数生成的字符串
  EXPECT_EQ(invocations.load(), 1);  // 断言：工厂函数仅被调用一次

  invocations = 0;  // 重置调用计数

  auto sCopy = s;  // 复制OptimisticLazy对象
  EXPECT_EQ(sCopy.ensure(factory), kLongString);  // 断言：复制对象中确保包含工厂函数生成的字符串
  EXPECT_EQ(invocations.load(), 0);  // 断言：工厂函数未被调用

  auto sMove = std::move(s);  // 移动OptimisticLazy对象
  EXPECT_EQ(sMove.ensure(factory), kLongString);  // 断言：移动后对象确保包含工厂函数生成的字符串
  EXPECT_EQ(invocations.load(), 0);  // 断言：工厂函数未被调用
  // NOLINTNEXTLINE(bugprone-use-after-move)
  EXPECT_EQ(s.ensure(factory), kLongString);  // 断言：移动后再次使用对象，确保包含工厂函数生成的字符串
  EXPECT_EQ(invocations.load(), 1);  // 断言：工厂函数仅被调用一次

  invocations = 0;  // 重置调用计数

  s = sCopy;  // 赋值给OptimisticLazy对象
  EXPECT_EQ(s.ensure(factory), kLongString);  // 断言：赋值后对象确保包含工厂函数生成的字符串
  EXPECT_EQ(invocations.load(), 0);  // 断言：工厂函数未被调用

  s = std::move(sCopy);  // 移动赋值给OptimisticLazy对象
  EXPECT_EQ(s.ensure(factory), kLongString);  // 断言：移动赋值后对象确保包含工厂函数生成的字符串
  EXPECT_EQ(invocations.load(), 0);  // 断言：工厂函数未被调用
}

TEST(LazyTest, PrecomputedLazyValue) {
  static const std::string kLongString = "I am a string";  // 静态常量字符串
  EXPECT_EQ(
      std::make_shared<c10::PrecomputedLazyValue<std::string>>(kLongString)
          ->get(),
      kLongString);  // 断言：预先计算的Lazy值与预定义的字符串相等
}

TEST(LazyTest, OptimisticLazyValue) {
  static const std::string kLongString = "I am a string";  // 静态常量字符串

  class LazyString : public c10::OptimisticLazyValue<std::string> {
    std::string compute() const override {
      return kLongString;  // 实现compute方法，返回预定义的字符串
    }
  };

  auto ls = std::make_shared<LazyString>();  // 创建LazyString对象
  EXPECT_EQ(ls->get(), kLongString);  // 断言：获取的字符串与预定义的字符串相等

  // Returned reference should be stable.
  EXPECT_EQ(&ls->get(), &ls->get());  // 断言：返回的引用应是稳定的
}

} // namespace c10_test
```