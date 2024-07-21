# `.\pytorch\torch\csrc\profiler\perf-inl.h`

```py
/*
 * PerfEvent
 * ---------
 */

// 禁用性能事件计数器
inline void PerfEvent::Disable() const {
#if defined(__ANDROID__) || defined(__linux__)
  ioctl(fd_, PERF_EVENT_IOC_DISABLE, 0);
#endif /* __ANDROID__ || __linux__ */
}

// 启用性能事件计数器
inline void PerfEvent::Enable() const {
#if defined(__ANDROID__) || defined(__linux__)
  ioctl(fd_, PERF_EVENT_IOC_ENABLE, 0);
#endif /* __ANDROID__ || __linux__ */
}

// 重置性能事件计数器
inline void PerfEvent::Reset() const {
#if defined(__ANDROID__) || defined(__linux__)
  ioctl(fd_, PERF_EVENT_IOC_RESET, 0);
#endif /* __ANDROID__ || __linux__ */
}

/*
 * PerfProfiler
 * ------------
 */

// 计算两个时间戳之间的差值
inline uint64_t PerfProfiler::CalcDelta(uint64_t start, uint64_t end) const {
  if (end < start) { // 溢出处理
    return end + (std::numeric_limits<uint64_t>::max() - start);
  }
  // 正常情况下的差值计算
  return end - start;
}

// 开始性能计数
inline void PerfProfiler::StartCounting() const {
  for (auto& e : events_) {
    e.Enable(); // 启用每个性能事件计数器
  }
}

// 停止性能计数
inline void PerfProfiler::StopCounting() const {
  for (auto& e : events_) {
    e.Disable(); // 禁用每个性能事件计数器
  }
}
```