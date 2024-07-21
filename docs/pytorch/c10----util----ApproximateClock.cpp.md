# `.\pytorch\c10\util\ApproximateClock.cpp`

```
// 包含所需的头文件
#include <c10/util/ApproximateClock.h>
#include <c10/util/ArrayRef.h>
#include <c10/util/irange.h>
#include <fmt/format.h>

// 进入c10命名空间
namespace c10 {

// 构造函数，初始化开始时间数组
ApproximateClockToUnixTimeConverter::ApproximateClockToUnixTimeConverter()
    : start_times_(measurePairs()) {}

// 测量精确时间和Unix时间的对
ApproximateClockToUnixTimeConverter::UnixAndApproximateTimePair
ApproximateClockToUnixTimeConverter::measurePair() {
  // 获取两次近似时间的测量，以避免排序偏差
  auto fast_0 = getApproximateTime();
  auto wall = std::chrono::system_clock::now();
  auto fast_1 = getApproximateTime();

  // 断言确保近似时间的单调性
  TORCH_INTERNAL_ASSERT(fast_1 >= fast_0, "getCount is non-monotonic.");

  // 计算当前系统时间的纳秒数
  auto t = std::chrono::duration_cast<std::chrono::nanoseconds>(
      wall.time_since_epoch());

  // 使用更稳定的方法计算两次近似时间的平均值
  return {t.count(), fast_0 + (fast_1 - fast_0) / 2};
}

// 执行多次测量并返回测量对数组
ApproximateClockToUnixTimeConverter::time_pairs
ApproximateClockToUnixTimeConverter::measurePairs() {
  // 静态变量，定义热身次数
  static constexpr auto n_warmup = 5;
  // 进行n_warmup次循环，进行热身操作
  for (C10_UNUSED const auto _ : c10::irange(n_warmup)) {
    getApproximateTime();
    static_cast<void>(steady_clock_t::now());
  }

  // 创建时间对数组
  time_pairs out;
  // 遍历时间对数组并测量每一对的值
  for (const auto i : c10::irange(out.size())) {
    out[i] = measurePair();
  }
  return out;
}

// 创建并返回一个将近似时间转换为Unix时间的函数对象
std::function<time_t(approx_time_t)> ApproximateClockToUnixTimeConverter::
    makeConverter() {
  // 获取结束时间对
  auto end_times = measurePairs();

  // 计算每个近似时钟滴答对应的实际经过时间的比例因子
  std::array<long double, replicates> scale_factors{};
  for (const auto i : c10::irange(replicates)) {
    auto delta_ns = end_times[i].t_ - start_times_[i].t_;
    auto delta_approx = end_times[i].approx_t_ - start_times_[i].approx_t_;
    scale_factors[i] = (double)delta_ns / (double)delta_approx;
  }
  // 对比例因子进行排序
  std::sort(scale_factors.begin(), scale_factors.end());
  long double scale_factor = scale_factors[replicates / 2 + 1];

  // 选择一个时间修正量t0，用于数值稳定性，使中间值更接近零
  auto t0 = start_times_[0].t_;
  auto t0_approx = start_times_[0].approx_t_;
  std::array<double, replicates> t0_correction{};
  for (const auto i : c10::irange(replicates)) {
    auto dt = start_times_[i].t_ - t0;
    auto dt_approx =
        (double)(start_times_[i].approx_t_ - t0_approx) * scale_factor;
    // 计算时间修正量，根据NOLINT标记，忽略某些静态分析警告
    t0_correction[i] = dt - (time_t)dt_approx; // NOLINT
  }
  // 调整t0，以提高数值稳定性
  t0 += t0_correction[t0_correction.size() / 2 + 1]; // NOLINT

  // 返回一个lambda函数，将近似时间映射到Unix时间
  return [=](approx_time_t t_approx) {
    // 更稳定的计算方式，比简单的乘法更合适
    // A * t_approx + B，详见上文说明
    // 将 t_approx 和 t0_approx 的差值乘以比例因子，并转换为时间类型 time_t，然后加上 t0
    return (time_t)((double)(t_approx - t0_approx) * scale_factor) + t0;
};
}

// 结束命名空间 c10
} // namespace c10
```