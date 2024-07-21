# `.\pytorch\torch\csrc\utils\cpp_stacktraces.cpp`

```py
#include <torch/csrc/utils/cpp_stacktraces.h>
// 引入 Torch 的 C++ 栈跟踪工具头文件

#include <cstdlib>
#include <cstring>

#include <c10/util/Exception.h>

namespace torch {
namespace {
// 命名空间 torch 下的匿名命名空间，用于定义内部函数和变量

bool compute_cpp_stack_traces_enabled() {
  // 计算是否启用 C++ 栈跟踪
  auto envar = std::getenv("TORCH_SHOW_CPP_STACKTRACES");
  if (envar) {
    // 如果环境变量 TORCH_SHOW_CPP_STACKTRACES 存在
    if (strcmp(envar, "0") == 0) {
      return false; // 如果其值为 "0"，返回 false
    }
    if (strcmp(envar, "1") == 0) {
      return true; // 如果其值为 "1"，返回 true
    }
    TORCH_WARN(
        "ignoring invalid value for TORCH_SHOW_CPP_STACKTRACES: ",
        envar,
        " valid values are 0 or 1.");
    // 如果值不是 "0" 或 "1"，则发出警告并返回 false
  }
  return false; // 默认情况下不启用 C++ 栈跟踪
}

bool compute_disable_addr2line() {
  // 计算是否禁用 addr2line 符号化工具
  auto envar = std::getenv("TORCH_DISABLE_ADDR2LINE");
  if (envar) {
    // 如果环境变量 TORCH_DISABLE_ADDR2LINE 存在
    if (strcmp(envar, "0") == 0) {
      return false; // 如果其值为 "0"，返回 false
    }
    if (strcmp(envar, "1") == 0) {
      return true; // 如果其值为 "1"，返回 true
    }
    TORCH_WARN(
        "ignoring invalid value for TORCH_DISABLE_ADDR2LINE: ",
        envar,
        " valid values are 0 or 1.");
    // 如果值不是 "0" 或 "1"，则发出警告并返回 false
  }
  return false; // 默认情况下不禁用 addr2line 符号化工具
}
} // namespace

bool get_cpp_stacktraces_enabled() {
  // 获取 C++ 栈跟踪是否启用的全局函数
  static bool enabled = compute_cpp_stack_traces_enabled();
  return enabled; // 返回是否启用 C++ 栈跟踪
}

static torch::unwind::Mode compute_symbolize_mode() {
  // 计算符号化模式
  auto envar_c = std::getenv("TORCH_SYMBOLIZE_MODE");
  if (envar_c) {
    std::string envar = envar_c;
    if (envar == "dladdr") {
      return unwind::Mode::dladdr; // 如果环境变量为 "dladdr"，返回 dladdr 模式
    } else if (envar == "addr2line") {
      return unwind::Mode::addr2line; // 如果环境变量为 "addr2line"，返回 addr2line 模式
    } else if (envar == "fast") {
      return unwind::Mode::fast; // 如果环境变量为 "fast"，返回 fast 模式
    } else {
      TORCH_CHECK(
          false,
          "expected {dladdr, addr2line, fast} for TORCH_SYMBOLIZE_MODE, got ",
          envar);
      // 如果环境变量值无效，则发出错误信息并终止程序
    }
  } else {
    // 如果环境变量 TORCH_SYMBOLIZE_MODE 不存在
    return compute_disable_addr2line() ? unwind::Mode::dladdr
                                       : unwind::Mode::addr2line;
    // 根据 TORCH_DISABLE_ADDR2LINE 的值决定返回 dladdr 或 addr2line 模式
  }
}

unwind::Mode get_symbolize_mode() {
  // 获取符号化模式的全局函数
  static unwind::Mode mode = compute_symbolize_mode();
  return mode; // 返回当前符号化模式
}

} // namespace torch
// 命名空间 torch 的结束标记
```