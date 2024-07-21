# `.\pytorch\torch\csrc\distributed\c10d\Work.hpp`

```py
#pragma once
// 只允许本头文件被包含一次，避免重复定义

#include <ATen/ATen.h>
// 引入 ATen 库，用于处理张量运算

#include <chrono>
// 引入 chrono 库，用于处理时间相关操作

#include <mutex>
// 引入 mutex 库，用于实现互斥锁

#include <vector>
// 引入 vector 库，用于操作动态数组

constexpr auto kNoTimeout = std::chrono::milliseconds(0);
// 定义一个常量 kNoTimeout，表示没有超时，设定为 0 毫秒

namespace c10d {

constexpr const char* const kSeqNumStoreKey = "SEQ_NUM_STORE_KEY";
// 定义常量指针 kSeqNumStoreKey，存储字符串 "SEQ_NUM_STORE_KEY"

enum class OpType : std::uint8_t {
  BROADCAST = 0,                  // 广播操作
  ALLREDUCE = 1,                  // 全局归约操作
  ALLREDUCE_COALESCED = 2,        // 合并全局归约操作
  REDUCE = 3,                     // 归约操作
  ALLGATHER = 4,                  // 全收集操作
  _ALLGATHER_BASE = 5,            // 全收集操作基础
  ALLGATHER_COALESCED = 6,        // 合并全收集操作
  GATHER = 7,                     // 收集操作
  SCATTER = 8,                    // 分散操作
  REDUCE_SCATTER = 9,             // 归约分散操作
  ALLTOALL_BASE = 10,             // 全交换操作基础
  ALLTOALL = 11,                  // 全交换操作
  SEND = 12,                      // 发送操作
  RECV = 13,                      // 接收操作
  RECVANYSOURCE = 14,             // 从任意源接收操作
  BARRIER = 15,                   // 屏障同步操作
  _REDUCE_SCATTER_BASE = 16,      // 归约分散操作基础
  COALESCED = 17,                 // 合并操作
  _ALLREDUCE_SPARSE = 18,         // 稀疏全局归约操作
  UNKNOWN = 100,                  // 未知操作
};

// 将 OpType 转换为人类可读的字符串
TORCH_API std::string opTypeToString(OpType opType);

// 检查操作是否是点对点操作 (SEND, RECV, RECVANYSOURCE)
TORCH_API bool isP2POp(OpType opType, bool batchP2P = false);

// 请不要使用 Work API，这部分功能将被移除，替代方案是 ivalue::Future
// Python 绑定可能会改变，请不要假设将用 pybind 绑定此类

};

struct TORCH_API WorkInfo {
  // 定义 WorkInfo 结构体，包含操作类型、序列号、开始时间、结束时间和活动时长
  WorkInfo(
      const OpType& opType,
      const uint64_t seq,
      const std::chrono::time_point<std::chrono::system_clock>& timeStarted,
      const std::chrono::time_point<std::chrono::system_clock>& timeFinished,
      const std::chrono::duration<float>& activeDuration)
      : opType(opType),
        seq(seq),
        timeStarted(timeStarted),
        timeFinished(timeFinished),
        activeDuration(activeDuration) {}

  OpType opType; // 操作类型
  uint64_t seq; // 序列号
  std::chrono::time_point<std::chrono::system_clock> timeStarted; // 开始时间点
  std::chrono::time_point<std::chrono::system_clock> timeFinished; // 结束时间点
  std::chrono::duration<float> activeDuration; // 活动持续时间
};

} // namespace c10d
// 结束 c10d 命名空间声明
```