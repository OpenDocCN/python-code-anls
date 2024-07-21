# `.\pytorch\torch\csrc\distributed\c10d\Backoff.cpp`

```
#include <torch/csrc/distributed/c10d/Backoff.hpp>

#include <exception>
#include <stdexcept>

// 定义 c10d 命名空间，包含分布式通信相关功能
namespace c10d {

// 匿名命名空间，用于私有函数或常量的定义
namespace {
// 初始间隔常量，设定为零毫秒
constexpr std::chrono::milliseconds kZeroInterval{0};

// 生成并返回一个随机种子值
int32_t randSeed() {
  std::random_device rd;
  return rd();
}
} // namespace

// ExponentialBackoffWithJitter 类的构造函数实现
ExponentialBackoffWithJitter::ExponentialBackoffWithJitter()
    : gen_(randSeed()) {}

// 计算并返回下一个退避间隔时间
std::chrono::milliseconds ExponentialBackoffWithJitter::nextBackoff() {
  // 检查初始间隔是否为零，如果是则抛出异常
  if (initialInterval == kZeroInterval) {
    throw std::out_of_range(
        "ExponentialBackoffWithJitter requires non-zero initial interval");
  }
  // 检查初始间隔是否超过最大间隔，如果是则抛出异常
  if (initialInterval > maxInterval) {
    throw std::out_of_range(
        "ExponentialBackoffWithJitter requires initialInterval <= maxInterval");
  }
  // 检查随机化因子是否在有效范围内，如果不是则抛出异常
  if (randomizationFactor >= 1 || randomizationFactor < 0) {
    throw std::out_of_range(
        "ExponentialBackoffWithJitter requires randomization factor (0,1]");
  }
  // 检查乘数是否大于等于 1，如果不是则抛出异常
  if (multiplier < 1.0) {
    throw std::out_of_range(
        "ExponentialBackoffWithJitter requires multiplier >=1");
  }

  // 检测初始设置阶段，将当前间隔设置为初始间隔
  if (currentInterval_ == kZeroInterval) {
    currentInterval_ = initialInterval;
  }

  // 根据随机化因子计算当前间隔的随机化范围
  std::chrono::milliseconds randomization{static_cast<int64_t>(
      randomizationFactor * static_cast<double>(currentInterval_.count()))};
  std::chrono::milliseconds minSampleInterval =
      currentInterval_ - randomization;
  std::chrono::milliseconds maxSampleInterval =
      currentInterval_ + randomization;

  // 在随机化范围内生成一个均匀分布的随机数作为退避间隔
  std::uniform_int_distribution<> dist(
      minSampleInterval.count(), maxSampleInterval.count());
  std::chrono::milliseconds backoffInterval{dist(gen_)};

  // 更新当前间隔为下一个周期的间隔，按乘数进行更新
  currentInterval_ = std::chrono::milliseconds(static_cast<int64_t>(
      static_cast<double>(currentInterval_.count()) * multiplier));

  // 如果更新后的当前间隔超过最大间隔，则设置为最大间隔
  if (currentInterval_ > maxInterval) {
    currentInterval_ = maxInterval;
  }

  // 返回计算得到的退避间隔
  return backoffInterval;
}

// 重置当前间隔为零毫秒，用于重新开始退避过程
void ExponentialBackoffWithJitter::reset() {
  currentInterval_ = kZeroInterval;
}

// FixedBackoff 类的构造函数，设定固定的退避间隔
FixedBackoff::FixedBackoff(std::chrono::milliseconds interval)
    : interval_(interval) {}

// 返回固定的退避间隔时间
std::chrono::milliseconds FixedBackoff::nextBackoff() {
  return interval_;
}

// 空实现，用于 FixedBackoff 类的重置操作
void FixedBackoff::reset() {}

} // namespace c10d
```