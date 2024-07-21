# `.\pytorch\torch\csrc\distributed\c10d\logging.cpp`

```py
// 版权声明及许可证信息
// Meta Platforms, Inc. 及其关联公司保留所有权利。
//
// 此源代码根据位于根目录中的 LICENSE 文件中的 BSD 风格许可证进行许可。

// 包含 Torch 的分布式 C10D 模块中的日志记录头文件
#include <torch/csrc/distributed/c10d/logging.h>

// 包含 Torch 的分布式 C10D 模块中的调试功能头文件
#include <torch/csrc/distributed/c10d/debug.h>

// 定义 c10d 命名空间下的 detail 命名空间
namespace c10d::detail {

// 检查给定的日志级别是否已启用
bool isLogLevelEnabled(LogLevel level) noexcept {
  // 调整日志级别的整数值，以匹配 c10 日志级别的映射
  int level_int = static_cast<int>(level) - 2;

  // 如果调整后的级别大于等于 0，则判断是否启用该级别的日志
  if (level_int >= 0) {
    return FLAGS_caffe2_log_level <= level_int;
  }

  // 对于调试和跟踪级别，仅当 c10 日志级别设置为 INFO 时才启用
  if (FLAGS_caffe2_log_level != 0) {
    return false;
  }

  // 调试级别为 -1，仅当调试级别不为 Off 时启用
  if (level_int == -1) {
    return debug_level() != DebugLevel::Off;
  }
  // 调试级别为 -2，仅当调试级别为 Detail 时启用
  if (level_int == -2) {
    return debug_level() == DebugLevel::Detail;
  }

  // 默认情况下不启用日志
  return false;
}

} // namespace c10d::detail
```