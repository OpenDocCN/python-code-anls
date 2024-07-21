# `.\pytorch\torch\csrc\profiler\events.h`

```
#pragma once

#include <array>  // 引入数组容器的头文件
#include <cstdint>  // 引入固定宽度整数类型的头文件
#include <cstring>  // 引入C风格字符串操作的头文件
#include <vector>  // 引入向量容器的头文件

namespace torch::profiler {

/* 用于存储性能计数器列表的向量类型 */
using perf_counters_t = std::vector<uint64_t>;

/* 标准的性能事件列表，与硬件或后端无关 */
constexpr std::array<const char*, 2> ProfilerPerfEvents = {
    /*
     * 两个时间点之间的处理元素（PE）周期数。这应该与墙上时间正相关。
     * 以 uint64_t 表示。PE可以是非CPU的。待定的多个PE参与时的报告行为（例如线程池）。
     */
    "cycles",

    /* 两个时间点之间的PE指令数。这应该与墙上时间和计算量（即工作量）正相关。
     * 在重复执行中，指令数应该是几乎不变的。以 uint64_t 表示。PE可以是非CPU的。
     */
    "instructions"};
} // namespace torch::profiler
```