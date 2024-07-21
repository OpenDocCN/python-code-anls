# `.\pytorch\c10\util\flags_use_gflags.cpp`

```
#include <c10/macros/Macros.h>

#ifdef C10_USE_GFLAGS

#include <c10/util/Flags.h>
#include <string>

namespace c10 {

using std::string;

// 设置用法消息，如果已经设置则直接返回
C10_EXPORT void SetUsageMessage(const string& str) {
  if (UsageMessage() != nullptr) {
    // Usage message has already been set, so we will simply return.
    return;
  }
  // 调用 gflags 库的设置用法消息函数
  gflags::SetUsageMessage(str);
}

// 返回程序的用法消息
C10_EXPORT const char* UsageMessage() {
  return gflags::ProgramUsage();
}

// 解析命令行标志，如果没有标志需要解析则直接返回 true
C10_EXPORT bool ParseCommandLineFlags(int* pargc, char*** pargv) {
  // In case there is no commandline flags to parse, simply return.
  if (*pargc == 0)
    return true;
  // 调用 gflags 库的解析命令行标志函数
  return gflags::ParseCommandLineFlags(pargc, pargv, true);
}

// 查询命令行标志是否已经被解析，由于当前无法查询 gflags，直接返回 true
C10_EXPORT bool CommandLineFlagsHasBeenParsed() {
  // There is no way we query gflags right now, so we will simply return true.
  return true;
}

} // namespace c10
#endif // C10_USE_GFLAGS
```