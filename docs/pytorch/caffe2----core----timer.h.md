# `.\pytorch\caffe2\core\timer.h`

```
#ifndef CAFFE2_CORE_TIMER_H_
#define CAFFE2_CORE_TIMER_H_

#include <chrono>  // 包含标准库chrono，用于时间测量

#include "caffe2/core/common.h"  // 包含caffe2核心common.h文件

namespace caffe2 {

/**
 * @brief A simple timer object for measuring time.
 *
 * This is a minimal class around a std::chrono::high_resolution_clock that
 * serves as a utility class for testing code.
 */
class Timer {
 public:
  typedef std::chrono::high_resolution_clock clock;  // 使用高精度时钟high_resolution_clock
  typedef std::chrono::nanoseconds ns;  // 定义纳秒ns为时间单位
  Timer() { Start(); }  // 构造函数，初始化时即开始计时
  /**
   * @brief Starts a timer.
   */
  inline void Start() { start_time_ = clock::now(); }  // 开始计时，记录当前时间点
  inline float NanoSeconds() {
    return static_cast<float>(
        std::chrono::duration_cast<ns>(clock::now() - start_time_).count());  // 计算从开始到现在经过的纳秒数
  }
  /**
   * @brief Returns the elapsed time in milliseconds.
   */
  inline float MilliSeconds() { return NanoSeconds() / 1000000.f; }  // 将纳秒转换为毫秒
  /**
   * @brief Returns the elapsed time in microseconds.
   */
  inline float MicroSeconds() { return NanoSeconds() / 1000.f; }  // 将纳秒转换为微秒
  /**
   * @brief Returns the elapsed time in seconds.
   */
  inline float Seconds() { return NanoSeconds() / 1000000000.f; }  // 将纳秒转换为秒

 protected:
  std::chrono::time_point<clock> start_time_;  // 记录开始计时的时间点
  C10_DISABLE_COPY_AND_ASSIGN(Timer);  // 禁止复制和赋值操作
};
}

#endif  // CAFFE2_CORE_TIMER_H_
```