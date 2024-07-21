# `.\pytorch\c10\util\Backtrace.h`

```py
#ifndef C10_UTIL_BACKTRACE_H_
#define C10_UTIL_BACKTRACE_H_

// 包含必要的头文件，用于后续定义和声明
#include <cstddef>
#include <memory>
#include <string>
#include <typeinfo>

// 包含 C10 库的宏定义
#include <c10/macros/Macros.h>
// 包含 Lazy 类的声明，用于延迟加载字符串
#include <c10/util/Lazy.h>

// 进入 c10 命名空间
namespace c10 {

// Backtrace 类型定义为指向 LazyValue<std::string> 共享指针
// LazyValue 用于延迟加载字符串，以便只在需要时进行符号化
using Backtrace = std::shared_ptr<const LazyValue<std::string>>;

// DEPRECATED: Prefer get_lazy_backtrace().
// 获取当前的函数调用堆栈信息，并返回字符串形式的堆栈信息
C10_API std::string get_backtrace(
    size_t frames_to_skip = 0,              // 跳过堆栈中的前几帧
    size_t maximum_number_of_frames = 64,   // 最大堆栈帧数
    bool skip_python_frames = true);        // 是否跳过 Python 的堆栈帧

// 获取延迟加载的函数调用堆栈信息
C10_API Backtrace get_lazy_backtrace(
    size_t frames_to_skip = 0,              // 跳过堆栈中的前几帧
    size_t maximum_number_of_frames = 64,   // 最大堆栈帧数
    bool skip_python_frames = true);        // 是否跳过 Python 的堆栈帧

} // namespace c10

#endif // C10_UTIL_BACKTRACE_H_
```