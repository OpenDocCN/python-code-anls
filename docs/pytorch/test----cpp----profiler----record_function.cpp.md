# `.\pytorch\test\cpp\profiler\record_function.cpp`

```py
// 包含标准库头文件
#include <array>
#include <atomic>
#include <condition_variable>
#include <iostream>
#include <memory>
#include <random>
#include <utility>
#include <vector>

// 包含第三方库头文件
#include <fmt/format.h>        // 格式化输出库
#include <gtest/gtest.h>       // Google 测试框架

// 包含 PyTorch 的头文件
#include <ATen/Parallel.h>     // 并行处理支持
#include <ATen/record_function.h>  // 记录函数调用支持
#include <c10/util/irange.h>   // 区间操作工具

// 测试函数：验证可以添加和移除回调函数（全局和线程本地）
TEST(RecordFunctionTest, AddRemove) {
  // 清除所有回调函数
  at::clearCallbacks();
  ASSERT_FALSE(at::hasCallbacks());  // 确保没有注册回调函数

  // 定义开始回调函数（返回空指针）
  auto start_callback =
      [](const at::RecordFunction& fn) -> std::unique_ptr<at::ObserverContext> {
    return nullptr;
  };
  // 定义结束回调函数（空实现）
  auto end_callback = [](const at::RecordFunction& fn, at::ObserverContext*) {};

  // 添加线程本地回调函数
  auto handle = at::addThreadLocalCallback(
      at::RecordFunctionCallback(start_callback, end_callback));

  ASSERT_TRUE(at::hasCallbacks());   // 确保有注册的回调函数
  ASSERT_TRUE(at::hasThreadLocalCallbacks());  // 确保线程本地有回调函数
  ASSERT_FALSE(at::hasGlobalCallbacks());  // 确保全局没有回调函数

  // 移除回调函数
  at::removeCallback(handle);
  ASSERT_FALSE(at::hasCallbacks());   // 确保没有注册回调函数

  // 再次添加全局回调函数
  handle = at::addGlobalCallback(
      at::RecordFunctionCallback(start_callback, end_callback));

  ASSERT_TRUE(at::hasCallbacks());   // 确保有注册的回调函数
  ASSERT_FALSE(at::hasThreadLocalCallbacks());  // 确保线程本地没有回调函数
  ASSERT_TRUE(at::hasGlobalCallbacks());   // 确保全局有回调函数

  // 再次移除回调函数
  at::removeCallback(handle);
  ASSERT_FALSE(at::hasCallbacks());   // 确保没有注册回调函数
}

// 测试函数：验证注册的回调函数确实被执行
TEST(RecordFunctionTest, ThreadLocalState) {
  // 清除所有回调函数
  at::clearCallbacks();
  ASSERT_FALSE(at::hasCallbacks());  // 确保没有注册回调函数

  // 定义静态变量来统计回调函数执行次数
  static int tls_test_start_counter;
  static int tls_test_end_counter;
  tls_test_start_counter = 0;
  tls_test_end_counter = 0;

  // 定义开始回调函数（增加计数）
  auto start_callback =
      [](const at::RecordFunction&) -> std::unique_ptr<at::ObserverContext> {
    ++tls_test_start_counter;
    return nullptr;
  };
  // 定义结束回调函数（增加计数）
  auto end_callback = [](const at::RecordFunction&, at::ObserverContext*) {
    ++tls_test_end_counter;
  };

  // 添加线程本地回调函数
  auto handle = at::addThreadLocalCallback(
      at::RecordFunctionCallback(start_callback, end_callback));

  {
    // 创建 RecordFunction 守卫
    at::RecordFunction guard(at::RecordScope::USER_SCOPE);
    guard.before("Test");
    // 验证开始回调函数执行次数为1，结束回调函数未执行
    EXPECT_EQ(tls_test_start_counter, 1);
    EXPECT_EQ(tls_test_end_counter, 0);
  }
  // 验证开始和结束回调函数都执行了一次
  EXPECT_EQ(tls_test_start_counter, 1);
  EXPECT_EQ(tls_test_end_counter, 1);

  {
    // 创建 RecordFunction 守卫，但不进行 profile
    tls_test_start_counter = 0;
    tls_test_end_counter = 0;
    at::DisableRecordFunctionGuard no_profile_guard;
    at::RecordFunction guard(at::RecordScope::USER_SCOPE);
    guard.before("Test");
    // 验证开始和结束回调函数都未执行
    EXPECT_EQ(tls_test_start_counter, 0);
    EXPECT_EQ(tls_test_end_counter, 0);
  }
  // 验证开始和结束回调函数都未执行
  EXPECT_EQ(tls_test_start_counter, 0);
  EXPECT_EQ(tls_test_end_counter, 0);

  {
    // 创建带有宏的 RecordFunction 守卫
    tls_test_start_counter = 0;
    tls_test_end_counter = 0;
    RECORD_FUNCTION("Test", {});
    // 验证开始回调函数执行次数为1，结束回调函数未执行
    EXPECT_EQ(tls_test_start_counter, 1);
    EXPECT_EQ(tls_test_end_counter, 0);
  }
  // 验证开始和结束回调函数都执行了一次
  EXPECT_EQ(tls_test_start_counter, 1);
  EXPECT_EQ(tls_test_end_counter, 1);

  // 移除回调函数
  at::removeCallback(handle);
  ASSERT_FALSE(at::hasCallbacks());  // 确保没有注册回调函数
}
// 定义测试用例 RecordFunctionTest 的 CallOrder 测试
TEST(RecordFunctionTest, CallOrder) {
  // 清除所有回调函数
  at::clearCallbacks();
  // 断言当前没有注册的回调函数
  ASSERT_FALSE(at::hasCallbacks());

  // 静态变量，用于记录当前的索引
  static int current_index;
  current_index = 0;

  // 预期的回调顺序数组
  static std::array<std::string, 8> expected_order = {
      "Start Callback 0 Outer",
      "Start Callback 1 Outer",
      "Start Callback 0 Inner",
      "Start Callback 1 Inner",
      "End Callback 0 Inner",
      "End Callback 1 Inner",
      "End Callback 0 Outer",
      "End Callback 1 Outer",
  };

  // 定义宏，用于注册回调函数
#define REGISTER_CALLBACK(index)                                       \
  at::addThreadLocalCallback(                                          \
      at::RecordFunctionCallback(                                      \
          [](const at::RecordFunction& fn)                             \
              -> std::unique_ptr<at::ObserverContext> {                \
            // 检查开始回调的格式是否与预期一致
            EXPECT_EQ(                                                 \
                fmt::format("Start Callback {} {}", index, fn.name()), \
                expected_order[current_index++]);                      \
            return nullptr;                                            \
          },                                                           \
          [](const at::RecordFunction& fn, at::ObserverContext*) {     \
            // 检查结束回调的格式是否与预期一致
            EXPECT_EQ(                                                 \
                fmt::format("End Callback {} {}", index, fn.name()),   \
                expected_order[current_index++]);                      \
          })                                                           \
          .scopes({at::RecordScope::FUNCTION}))

  // 注册回调函数 0 和 1
  REGISTER_CALLBACK(0);
  REGISTER_CALLBACK(1);
#undef REGISTER_CALLBACK

  // 记录函数调用 "Outer"
  RECORD_FUNCTION("Outer", {});
  // 记录函数调用 "Inner"
  { RECORD_FUNCTION("Inner", {}); }

  // 清除所有回调函数
  at::clearCallbacks();
  // 断言当前没有注册的回调函数
  ASSERT_FALSE(at::hasCallbacks());
}

// 确保线程本地存储在线程迁移时能够正确地工作
TEST(RecordFunctionTest, ThreadMigration) {
  // 清除所有回调函数
  at::clearCallbacks();
  // 断言当前没有注册的回调函数
  ASSERT_FALSE(at::hasCallbacks());

  // 静态变量，用于记录调用次数
  static int call_count;
  call_count = 0;

  // 添加线程本地回调函数，并计数回调次数
  auto handle = at::addThreadLocalCallback(
      at::RecordFunctionCallback(
          [](const at::RecordFunction&)
              -> std::unique_ptr<at::ObserverContext> { return nullptr; },
          [](const at::RecordFunction&, at::ObserverContext*) { ++call_count; })
          .scopes({at::RecordScope::FUNCTION}));

  // 验证回调次数为 0
  EXPECT_EQ(call_count, 0);

  // 创建条件变量和互斥锁
  std::condition_variable cv;
  std::mutex lock;
  // 在线程中启动任务
  at::launch([&cv]() {
    // 记录函数调用 "Test"
    RECORD_FUNCTION("Test", {});
    cv.notify_all();
  });
  auto guard = std::unique_lock<std::mutex>(lock);
  // 等待条件变量，直到回调次数大于 0
  cv.wait(guard, [] { return call_count > 0; });

  // 验证回调次数为 1
  EXPECT_EQ(call_count, 1);

  // 移除回调函数处理句柄
  at::removeCallback(handle);
  // 断言当前没有注册的回调函数
  ASSERT_FALSE(at::hasCallbacks());
}
TEST(RecordFunctionTest, Sampling) {
  // 清除所有回调函数，确保没有回调函数存在
  at::clearCallbacks();
  // 断言确认没有任何回调函数存在
  ASSERT_FALSE(at::hasCallbacks());

  // 定义静态变量，用于计数
  static int sample_test_counter;
  sample_test_counter = 0;

  // 设置随机数种子和几何分布的概率
  uint32_t seed = 12345;
  double p = 0.25;

  // 为测试设置记录函数的随机数种子
  at::set_record_function_seed_for_testing(seed);
  // 创建 Mersenne Twister 伪随机数生成器，并设置种子
  std::mt19937 generator;
  generator.seed(seed);
  // 创建几何分布对象
  auto dist = std::geometric_distribution<int>(p);

  // 确定应该触发的步骤
  auto outcomes = std::array<int, 5>{7, 0, 0, 6, 2};
  // 遍历 outcomes 数组，检查生成的几何分布值是否符合预期
  for (const auto i : c10::irange(outcomes.size())) {
    ASSERT_EQ(dist(generator), outcomes[i]);
  }

  // 用于存储期望的计数值的向量
  std::vector<int> expected_counts;
  // 初始化运行计数
  int running_count = 0;
  // 构建期望计数向量，根据 outcomes 中的值
  for (const auto i : c10::irange(outcomes.size())) {
    for (const auto j : c10::irange(outcomes[i])) {
      expected_counts.push_back(running_count);
    }
    expected_counts.push_back(++running_count);
  }

  // 定义开始回调函数，增加 sample_test_counter 计数
  auto start_callback =
      [](const at::RecordFunction& fn) -> std::unique_ptr<at::ObserverContext> {
    ++sample_test_counter;
    return nullptr;
  };
  // 定义结束回调函数，什么也不做
  auto end_callback = [](const at::RecordFunction& fn, at::ObserverContext*) {};

  // 添加线程本地回调函数，用于记录函数回调
  auto handle = at::addThreadLocalCallback(
      at::RecordFunctionCallback(start_callback, end_callback)
          .samplingProb(p)
          .scopes({at::RecordScope::FUNCTION}));

  // 遍历期望计数向量，测试记录函数的执行和回调函数的触发
  for (const auto i : c10::irange(expected_counts.size())) {
    // 记录函数执行
    RECORD_FUNCTION("Test", {});
    // 检查 sample_test_counter 是否等于期望的计数值
    EXPECT_EQ(sample_test_counter, expected_counts[i]);
  }

  // 移除回调函数句柄
  at::removeCallback(handle);
  // 断言确认没有任何回调函数存在
  ASSERT_FALSE(at::hasCallbacks());
}

// 针对注册的一组复杂回调函数验证采样功能
TEST(RecordFunctionTest, MultipleCallbacks) {
  // 清除所有回调函数，确保没有回调函数存在
  at::clearCallbacks();
  // 断言确认没有任何回调函数存在
  ASSERT_FALSE(at::hasCallbacks());

  // 设置随机数种子
  uint32_t seed = 54321;

  // 创建 Mersenne Twister 伪随机数生成器，并设置种子
  std::mt19937 generator;
  generator.seed(seed);

  // 定义一个 lambda 函数用于生成样本数，根据概率 p
  auto sample = [&](double p) {
    return (p < 1.0 ? std::geometric_distribution<int>(p)(generator) : 0) + 1;
  };

  // 定义一组概率值和下一次调用的数组
  std::array<double, 4> probabilities{0.1, 1.0, 1.0, 0.3};
  std::array<int, 4> next_call;
  std::array<int, 4> counts;
  static std::array<int, 4> counts_from_rec_fn;
  counts_from_rec_fn.fill(0);

  // 定义开始回调函数，增加 counts_from_rec_fn[0] 计数
  auto start_callback_0 =
      [](const at::RecordFunction& fn) -> std::unique_ptr<at::ObserverContext> {
    ++counts_from_rec_fn[0];
    return nullptr;
  };

  // 定义结束回调函数，什么也不做
  auto end_callback = [](const at::RecordFunction& fn, at::ObserverContext*) {};


这样，每行代码都得到了适当的注释，描述了其功能和目的。
#define REGISTER_CALLBACK(register_fn, index)                   \
  // 定义一个宏，用于注册回调函数，接受注册函数和索引作为参数
  register_fn(at::RecordFunctionCallback(                       \
                  // 使用 RecordFunctionCallback 创建回调函数，该函数接受 RecordFunction 作为参数
                  [](const at::RecordFunction& fn)              \
                      -> std::unique_ptr<at::ObserverContext> { \
                    // 匿名函数，每次调用增加特定索引处的回调计数
                    ++counts_from_rec_fn[index];                \
                    // 返回空指针，表示不使用观察器上下文
                    return nullptr;                             \
                  },                                            \
                  end_callback)                                 \
                  // 设置采样概率为 probabilities[index]
                  .samplingProb(probabilities[index])           \
                  // 设置作用域为 FUNCTION
                  .scopes({at::RecordScope::FUNCTION}))

  // 注册全局回调函数，索引为 0
  REGISTER_CALLBACK(at::addGlobalCallback, 0);
  // 注册全局回调函数，索引为 1
  REGISTER_CALLBACK(at::addGlobalCallback, 1);
  // 注册线程本地回调函数，索引为 2
  REGISTER_CALLBACK(at::addThreadLocalCallback, 2);

  // RecordFunction 机制会在注册新观察器时重建回调函数，因此需要等到最后一个回调来设置随机数种子
  // 用于测试目的设置 RecordFunction 的随机数种子
  at::set_record_function_seed_for_testing(seed);
  // 注册线程本地回调函数，索引为 3
  REGISTER_CALLBACK(at::addThreadLocalCallback, 3);
#undef REGISTER_CALLBACK

// 对于 probabilities 大小范围内的每个索引，设置下一个调用的采样次数
for (const auto i : c10::irange(probabilities.size())) {
  next_call[i] = sample(probabilities[i]);
}

// 执行 50 次迭代
for (const auto i : c10::irange(50)) {
  RECORD_FUNCTION("Test", {});  // 记录函数执行，名称为 "Test"
  // 对于 next_call 的每个索引进行迭代
  for (const auto j : c10::irange(next_call.size())) {
    // 如果 next_call[j] 为零，增加 counts[j]，重新设置 next_call[j]
    if (!(--next_call[j])) {
      ++counts[j];
      next_call[j] = sample(probabilities[j]);
    }
    // 断言 counts[j] 等于 counts_from_rec_fn[j]
    EXPECT_EQ(counts[j], counts_from_rec_fn[j]);
  }
}

// 清除所有回调函数
at::clearCallbacks();
// 断言不再有任何回调函数
ASSERT_FALSE(at::hasCallbacks());
```