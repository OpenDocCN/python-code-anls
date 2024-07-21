# `.\pytorch\test\cpp\c10d\BackoffTest.cpp`

```
// 包含c10/util/irange.h头文件，用于范围迭代器
#include <c10/util/irange.h>
// 包含测试所需的StoreTestCommon.hpp头文件
#include "StoreTestCommon.hpp"

// 包含标准输入输出流库
#include <iostream>
// 包含线程库
#include <thread>

// 包含torch/csrc/distributed/c10d/Backoff.hpp头文件，定义了退避策略相关类
#include <torch/csrc/distributed/c10d/Backoff.hpp>

// 测试用例1：测试指数退避策略默认参数
TEST(BackoffTest, exponentialBackoffDefaults) {
  // 创建指数退避策略对象
  c10d::ExponentialBackoffWithJitter backoff;
  // 验证初始间隔为500毫秒
  EXPECT_EQ(backoff.initialInterval, std::chrono::milliseconds(500));
  // 验证最大间隔为60000毫秒
  EXPECT_EQ(backoff.maxInterval, std::chrono::milliseconds(60000));
  // 验证倍增因子为1.5
  EXPECT_EQ(backoff.multiplier, 1.5);
  // 验证随机因子为0.5
  EXPECT_EQ(backoff.randomizationFactor, 0.5);
}

// 测试用例2：测试自定义指数退避策略参数
TEST(BackoffTest, exponentialBackoff) {
  // 创建指数退避策略对象
  c10d::ExponentialBackoffWithJitter backoff;
  // 设置随机因子为0.0
  backoff.randomizationFactor = 0.0;
  // 设置倍增因子为2.0
  backoff.multiplier = 2.0;
  // 设置最大间隔为5000毫秒
  backoff.maxInterval = std::chrono::milliseconds(5000);

  // 验证不同步骤的下一个退避时间
  EXPECT_EQ(backoff.nextBackoff(), std::chrono::milliseconds(500));
  EXPECT_EQ(backoff.nextBackoff(), std::chrono::milliseconds(1000));
  EXPECT_EQ(backoff.nextBackoff(), std::chrono::milliseconds(2000));
  EXPECT_EQ(backoff.nextBackoff(), std::chrono::milliseconds(4000));
  EXPECT_EQ(backoff.nextBackoff(), std::chrono::milliseconds(5000));
  EXPECT_EQ(backoff.nextBackoff(), std::chrono::milliseconds(5000));

  // 重置退避策略对象
  backoff.reset();
  // 验证重置后的下一个退避时间
  EXPECT_EQ(backoff.nextBackoff(), std::chrono::milliseconds(500));
  EXPECT_EQ(backoff.nextBackoff(), std::chrono::milliseconds(1000));
}

// 测试用例3：测试带随机因子的指数退避策略
TEST(BackoffTest, expontentialBackoffRandomization) {
  // 创建指数退避策略对象
  c10d::ExponentialBackoffWithJitter backoff;
  // 设置初始间隔为1000毫秒
  backoff.initialInterval = std::chrono::milliseconds(1000);
  // 设置随机因子为0.5
  backoff.randomizationFactor = 0.5;
  // 设置倍增因子为1.0
  backoff.multiplier = 1.0;
  // 设置最大间隔为5000毫秒
  backoff.maxInterval = std::chrono::milliseconds(5000);

  // 进行100次退避操作，验证退避时间在指定范围内
  for (int i = 0; i < 100; i++) {
    auto backoffDur = backoff.nextBackoff();
    EXPECT_GE(backoffDur, std::chrono::milliseconds(500));
    EXPECT_LE(backoffDur, std::chrono::milliseconds(1500));
  }
}

// 测试用例4：测试固定间隔退避策略
TEST(BackoffTest, fixedBackoff) {
  // 创建固定间隔退避策略对象，间隔为1000毫秒
  c10d::FixedBackoff backoff{std::chrono::milliseconds(1000)};

  // 验证固定间隔下一个退避时间
  EXPECT_EQ(backoff.nextBackoff(), std::chrono::milliseconds(1000));
  EXPECT_EQ(backoff.nextBackoff(), std::chrono::milliseconds(1000));

  // 重置退避策略对象
  backoff.reset();
  // 验证重置后的下一个退避时间
  EXPECT_EQ(backoff.nextBackoff(), std::chrono::milliseconds(1000));
}

// 测试用例5：测试休眠操作
TEST(BackoffTest, sleep) {
  // 设置休眠时间为10毫秒
  std::chrono::milliseconds sleepTime{10};
  // 创建固定间隔退避策略对象，间隔为设定的休眠时间
  c10d::FixedBackoff backoff{sleepTime};

  // 验证固定间隔下一个退避时间
  EXPECT_EQ(backoff.nextBackoff(), sleepTime);

  // 记录休眠操作开始时间
  auto start = std::chrono::high_resolution_clock::now();
  // 执行退避策略的休眠操作
  backoff.sleepBackoff();
  // 计算实际休眠时间
  auto dur = std::chrono::high_resolution_clock::now() - start;
  // 验证实际休眠时间不少于设定的休眠时间
  EXPECT_GE(dur, sleepTime);
}
```