# `.\pytorch\torch\csrc\distributed\c10d\debug.cpp`

```py
// 版权声明及许可信息
// Meta Platforms, Inc. 及其关联公司保留所有权利。
//
// 本源代码在根目录下的 LICENSE 文件中以 BSD 风格许可证授权。

#include <torch/csrc/distributed/c10d/debug.h>

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <string>

#include <torch/csrc/distributed/c10d/exception.h>
#include <torch/csrc/distributed/c10d/logging.h>

namespace c10d {
namespace detail {
namespace {

// 从环境变量 TORCH_DISTRIBUTED_DEBUG 中加载调试级别
DebugLevel loadDebugLevelFromEnvironment() {
  // 获取环境变量的值
  char* env_value = std::getenv("TORCH_DISTRIBUTED_DEBUG");

  // 如果环境变量值为空，则默认为调试级别为 Off
  if (env_value == nullptr) {
    return DebugLevel::Off;
  }

  // 定义调试级别变量
  DebugLevel level{};

  // 将环境变量值转换为大写
  std::string level_str{env_value};
  std::transform(
      level_str.begin(),
      level_str.end(),
      level_str.begin(),
      [](unsigned char c) { return toupper(c); });

  // 根据环境变量值设置调试级别
  if (level_str == "OFF") {
    level = DebugLevel::Off;
  } else if (level_str == "INFO") {
    level = DebugLevel::Info;
  } else if (level_str == "DETAIL") {
    level = DebugLevel::Detail;
  } else {
    // 如果环境变量值不在预期范围内，则抛出异常
    throw std::invalid_argument(
        "The value of TORCH_DISTRIBUTED_DEBUG must be OFF, INFO, or DETAIL.");
  }

  // 记录调试级别设置信息
  C10D_INFO("The debug level is set to {}.", level_str);

  return level;
}

} // namespace
} // namespace detail

namespace {

// 全局调试级别变量，默认为 Off
DebugLevel g_debug_level = DebugLevel::Off;

} // namespace

// 设置全局调试级别
void setDebugLevel(DebugLevel level) {
  g_debug_level = level;
}

// 从环境变量设置调试级别
void setDebugLevelFromEnvironment() {
  g_debug_level = detail::loadDebugLevelFromEnvironment();
}

// 获取当前调试级别
DebugLevel debug_level() noexcept {
  return g_debug_level;
}

} // namespace c10d
```