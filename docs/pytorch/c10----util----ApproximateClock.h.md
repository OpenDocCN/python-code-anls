# `.\pytorch\c10\util\ApproximateClock.h`

```
// 版权声明，声明代码版权归Facebook所有，仅限2023年以及之后使用。
#pragma once

// 引入必要的头文件
#include <c10/macros/Export.h>  // 导出宏定义
#include <array>                // 标准数组库
#include <chrono>               // C++标准库中的时间操作
#include <cstddef>              // 标准库中定义的常量大小类型
#include <cstdint>              // 标准整数类型
#include <ctime>                // C风格时间操作
#include <functional>           // 标准函数库
#include <type_traits>          // C++类型特性库

// 根据条件包含特定平台相关头文件
#if defined(C10_IOS) && defined(C10_MOBILE)
#include <sys/time.h>  // iOS特定头文件，用于获取时间
#endif

// 根据CPU架构定义宏C10_RDTSC
#if defined(__i386__) || defined(__x86_64__) || defined(__amd64__)
#define C10_RDTSC
#if defined(_MSC_VER)
#include <intrin.h>  // Microsoft编译器下的内联汇编支持头文件
#elif defined(__CUDACC__) || defined(__HIPCC__)
#undef C10_RDTSC  // CUDA或HIP编译器下不支持RDTSC，取消定义
#elif defined(__clang__)
// Clang编译器默认支持__rdtsc指令
#elif defined(__GNUC__)
#include <x86intrin.h>  // GCC编译器下的x86内联汇编支持头文件
#else
#undef C10_RDTSC  // 其他情况下取消定义C10_RDTSC
#endif
#endif

// 命名空间定义
namespace c10 {

// 定义时间类型为int64_t
using time_t = int64_t;
// 定义稳定时钟类型，根据高分辨率时钟是否稳定选择高分辨率时钟或稳定时钟
using steady_clock_t = std::conditional_t<
    std::chrono::high_resolution_clock::is_steady,
    std::chrono::high_resolution_clock,
    std::chrono::steady_clock>;

// 获取当前时间距离Unix纪元的纳秒数
inline time_t getTimeSinceEpoch() {
  auto now = std::chrono::system_clock::now().time_since_epoch();
  return std::chrono::duration_cast<std::chrono::nanoseconds>(now).count();
}

// 获取当前时间的纳秒数，支持选择是否允许使用单调时钟
inline time_t getTime(bool allow_monotonic = false) {
#if defined(C10_IOS) && defined(C10_MOBILE)
  // 在iOS平台上获取当前时间的秒和微秒，并转换为纳秒
  struct timeval now;
  gettimeofday(&now, NULL);
  return static_cast<time_t>(now.tv_sec) * 1000000000 +
      static_cast<time_t>(now.tv_usec) * 1000;
#elif defined(_WIN32) || defined(__MACH__)
  // 在Windows和macOS平台上使用稳定时钟获取当前时间的纳秒数
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             steady_clock_t::now().time_since_epoch())
      .count();
#else
  // 在Linux平台上使用clock_gettime获取当前时间的纳秒数，支持选择是否使用单调时钟
  struct timespec t {};
  auto mode = CLOCK_REALTIME;
  if (allow_monotonic) {
    mode = CLOCK_MONOTONIC;
  }
  clock_gettime(mode, &t);
  return static_cast<time_t>(t.tv_sec) * 1000000000 +
      static_cast<time_t>(t.tv_nsec);
#endif
}

// 获取近似时间，如果可用快速机制如TSC，则使用TSC；否则返回当前时间的纳秒数
inline auto getApproximateTime() {
#if defined(C10_RDTSC)
  return static_cast<uint64_t>(__rdtsc());
#else
  return getTime();
#endif
}

// 定义近似时间类型为getApproximateTime函数的返回类型
using approx_time_t = decltype(getApproximateTime());
static_assert(
    std::is_same_v<approx_time_t, int64_t> ||
        std::is_same_v<approx_time_t, uint64_t>,
    "Expected either int64_t (`getTime`) or uint64_t (some TSC reads).");

// 将getCount函数的结果转换为自Unix纪元以来的纳秒数。
# 定义一个名为 ApproximateClockToUnixTimeConverter 的类，用于将近似时间转换为 Unix 时间
class C10_API ApproximateClockToUnixTimeConverter final {
 public:
  # 默认构造函数声明
  ApproximateClockToUnixTimeConverter();
  
  # 创建并返回一个函数对象，用于将近似时间转换为 time_t 类型的 Unix 时间
  std::function<time_t(approx_time_t)> makeConverter();

  # 结构体，包含 Unix 时间 t_ 和近似时间 approx_t_ 的对
  struct UnixAndApproximateTimePair {
    time_t t_;
    approx_time_t approx_t_;
  };

  # 静态方法，用于测量并返回 Unix 时间和近似时间的对
  static UnixAndApproximateTimePair measurePair();

 private:
  # 静态常量，表示测量次数的重复次数
  static constexpr size_t replicates = 1001;
  
  # 使用 std::array 存储 Unix 时间和近似时间的对，长度为 replicates
  using time_pairs = std::array<UnixAndApproximateTimePair, replicates>;

  # 测量并返回多对 Unix 时间和近似时间的数组
  time_pairs measurePairs();

  # 存储起始时间对的数组
  time_pairs start_times_;
};

# 命名空间 c10 的结束标记
} // namespace c10
```