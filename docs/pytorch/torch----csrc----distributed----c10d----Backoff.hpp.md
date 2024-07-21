# `.\pytorch\torch\csrc\distributed\c10d\Backoff.hpp`

```py
#pragma once
// 预处理指令，确保头文件只被包含一次

#include <chrono>
// 包含时间相关的标准库

#include <random>
// 包含随机数生成器的标准库

#include <thread>
// 包含线程操作相关的标准库

#include <c10/macros/Macros.h>
// 包含 c10 库的宏定义

namespace c10d {
// 命名空间 c10d，用于组织代码，避免命名冲突

class TORCH_API Backoff {
 public:
  virtual ~Backoff() = default;
  // 虚析构函数，用于多态释放资源

  virtual std::chrono::milliseconds nextBackoff() = 0;
  // 纯虚函数，子类必须实现，返回下一个退避时长的毫秒数

  virtual void reset() = 0;
  // 纯虚函数，子类必须实现，重置退避状态

  void sleepBackoff() {
    // 休眠退避时间
    std::this_thread::sleep_for(nextBackoff());
  }
};

class TORCH_API ExponentialBackoffWithJitter : public Backoff {
 public:
  ExponentialBackoffWithJitter();
  // 构造函数，初始化指数退避算法带抖动的实例

  std::chrono::milliseconds nextBackoff() override;
  // 实现基类纯虚函数，计算下一个指数退避时间

  void reset() override;
  // 实现基类纯虚函数，重置指数退避状态

 public:
  std::chrono::milliseconds initialInterval{500};
  // 初始退避时间，单位毫秒，默认为 500 毫秒

  double randomizationFactor{0.5};
  // 随机因子，用于在计算退避时间时引入随机性，默认为 0.5

  double multiplier{1.5};
  // 乘数，用于计算下一个退避时间，默认为 1.5

  std::chrono::milliseconds maxInterval{60000};
  // 最大退避时间，单位毫秒，默认为 60000 毫秒

 private:
  std::mt19937 gen_;
  // 随机数生成器引擎

  std::chrono::milliseconds currentInterval_{0};
  // 当前退避时间，单位毫秒，默认为 0
};

class TORCH_API FixedBackoff : public Backoff {
 public:
  FixedBackoff(std::chrono::milliseconds interval);
  // 构造函数，初始化固定退避时间的实例

  std::chrono::milliseconds nextBackoff() override;
  // 实现基类纯虚函数，返回固定的退避时间

  void reset() override;
  // 实现基类纯虚函数，重置固定退避状态

 private:
  std::chrono::milliseconds interval_;
  // 固定的退避时间，单位毫秒
};

} // namespace c10d
// 命名空间 c10d 结束
```