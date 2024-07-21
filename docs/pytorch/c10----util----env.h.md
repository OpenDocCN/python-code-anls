# `.\pytorch\c10\util\env.h`

```
#pragma once
// 防止头文件被多次包含的预处理指令

#include <c10/util/Exception.h>
// 包含c10库中的Exception.h头文件

#include <cstdlib>
// 包含标准库cstdlib，提供对环境变量操作的支持

#include <cstring>
// 包含标准库cstring，提供字符串操作函数

#include <optional>
// 包含标准库optional，提供std::optional类的支持

namespace c10::utils {
// 进入c10::utils命名空间

// 读取环境变量的值，并返回相应的optional<bool>：
// - optional<true>，如果设为"1"
// - optional<false>，如果设为"0"
// - nullopt，如果设为其他值
//
// 注意：
// 如果环境变量的值不是0或1，会发出警告。
inline std::optional<bool> check_env(const char* name) {
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4996)
#endif
  auto envar = std::getenv(name);
// 使用标准库函数获取环境变量name的值，并存储在envar中

#ifdef _MSC_VER
#pragma warning(pop)
#endif

  if (envar) {
    // 如果环境变量的值是"0"，返回false
    if (strcmp(envar, "0") == 0) {
      return false;
    }
    // 如果环境变量的值是"1"，返回true
    if (strcmp(envar, "1") == 0) {
      return true;
    }
    // 如果环境变量的值既不是"0"也不是"1"，发出警告并返回nullopt
    TORCH_WARN(
        "Ignoring invalid value for boolean flag ",
        name,
        ": ",
        envar,
        "valid values are 0 or 1.");
  }
  // 如果环境变量未设置或者其它情况，返回nullopt
  return std::nullopt;
}
} // namespace c10::utils
// 结束c10::utils命名空间
```