# `.\pytorch\torch\csrc\profiler\unwind\unwind.h`

```py
#pragma once
// 防止头文件被多次包含的预处理指令

#include <c10/macros/Export.h>
// 包含 c10 库中的 Export.h 文件

#include <c10/util/Optional.h>
// 包含 c10 库中的 Optional.h 文件

#include <cstdint>
// 包含 C++ 标准库中的 cstdint 头文件，定义了标准整数类型

#include <string>
// 包含 C++ 标准库中的 string 头文件，定义了字符串类型和相关操作

#include <vector>
// 包含 C++ 标准库中的 vector 头文件，定义了动态数组类型和相关操作

namespace torch::unwind {
// 命名空间 torch::unwind，包含以下符号

// gather current stack, relatively fast.
// gets faster once the cache of program counter locations is warm.
// TORCH_API is defined for symbol visibility.
// 获取当前堆栈信息，相对较快。
// 一旦程序计数器位置缓存热起来，速度会更快。
// TORCH_API 用于定义符号的可见性。
TORCH_API std::vector<void*> unwind();
// 声明函数 unwind()，返回一个 void* 类型的指针数组 vector，用于获取当前堆栈信息

struct Frame {
  std::string filename;
  std::string funcname;
  uint64_t lineno;
};
// 定义结构体 Frame，包含文件名 filename、函数名 funcname 和行号 lineno 成员

enum class Mode { addr2line, fast, dladdr };
// 声明枚举类型 Mode，包含 addr2line、fast 和 dladdr 三种模式

// note: symbolize is really slow
// it will launch an addr2line process that has to parse dwarf
// information from the libraries that frames point into.
// Callers should first batch up all the unique void* pointers
// across a number of unwind states and make a single call to
// symbolize.
// 注意：symbolize 函数非常慢
// 它会启动一个 addr2line 进程，需要解析帧指向的库中的 dwarf 信息。
// 调用者应该首先将所有唯一的 void* 指针批量收集起来，
// 在多次 unwind 状态后作单次调用 symbolize。
TORCH_API std::vector<Frame> symbolize(
    const std::vector<void*>& frames,
    Mode mode);
// 声明函数 symbolize()，接受 void* 指针数组 frames 和 Mode 枚举类型 mode 作为参数，
// 返回一个 Frame 结构体数组 vector，用于将 void* 指针转换为源码位置信息

// returns path to the library, and the offset of the addr inside the library
// 返回库的路径和地址在库中的偏移量
TORCH_API std::optional<std::pair<std::string, uint64_t>> libraryFor(
    void* addr);
// 声明函数 libraryFor()，接受 void* 类型的指针 addr 作为参数，
// 返回一个 std::optional 包含 std::pair<std::string, uint64_t> 类型的对象，
// 用于查找包含给定地址的库文件路径和地址偏移量

struct Stats {
  size_t hits = 0;
  size_t misses = 0;
  size_t unsupported = 0;
  size_t resets = 0;
};
// 定义结构体 Stats，包含 hits、misses、unsupported 和 resets 四个成员，均为 size_t 类型

Stats stats();
// 声明函数 stats()，返回一个 Stats 结构体对象
// 用于获取统计信息，包括命中次数、未命中次数、不支持次数和重置次数

} // namespace torch::unwind
// 命名空间 torch::unwind 结束
```