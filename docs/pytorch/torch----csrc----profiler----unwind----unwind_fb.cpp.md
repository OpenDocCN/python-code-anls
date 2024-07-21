# `.\pytorch\torch\csrc\profiler\unwind\unwind_fb.cpp`

```py
#if defined(__linux__) && (defined(__x86_64__) || defined(__aarch64__)) && \
    defined(__has_include) &&                                              \
    __has_include("ext/stdio_filebuf.h") && defined(FBCODE_CAFFE2)
// 如果目标平台是 Linux，并且是 x86_64 或者 aarch64 架构，同时满足以下条件：
// 1. 支持 __has_include 特性，
// 2. 包含 "ext/stdio_filebuf.h" 头文件，
// 3. 定义了 FBCODE_CAFFE2 宏，
// 则执行以下代码段

#include <c10/util/flat_hash_map.h>
#include <llvm/DebugInfo/Symbolize/Symbolize.h>
#include <torch/csrc/profiler/unwind/unwind.h>

namespace torch::unwind {

// 符号化给定的调用栈帧，返回符号化后的信息列表
std::vector<Frame> symbolize(const std::vector<void*>& frames, Mode mode) {
  static std::mutex symbolize_mutex;  // 静态互斥锁，用于保护共享资源
  static llvm::symbolize::LLVMSymbolizer symbolizer;  // 静态 LLVM 符号化器对象
  static ska::flat_hash_map<void*, Frame> frame_map_;  // 静态哈希映射，保存地址到帧信息的映射

  std::lock_guard<std::mutex> guard(symbolize_mutex);  // 加锁保护临界区，防止多线程冲突
  std::vector<Frame> results;  // 存储符号化后的结果
  results.reserve(frames.size());  // 预留足够的空间以避免不必要的重新分配
  for (auto addr : frames) {
    if (!frame_map_.count(addr)) {  // 如果地址尚未在映射表中
      auto frame = Frame{"??", "<unwind unsupported>", 0};  // 初始化一个未知帧信息
      auto maybe_library = libraryFor(addr);  // 获取包含指定地址的库信息
      if (maybe_library) {
        auto libaddress = maybe_library->second - 1;  // 库地址减一，可能是为了修正
        auto r = symbolizer.symbolizeCode(
            maybe_library->first,
            {libaddress, llvm::object::SectionedAddress::UndefSection});  // 符号化给定地址
        if (r) {
          frame.filename = r->FileName;  // 设置文件名
          frame.funcname = r->FunctionName;  // 设置函数名
          frame.lineno = r->Line;  // 设置行号
        }
      }
      frame_map_[addr] = std::move(frame);  // 将符号化后的帧信息存入映射表
    }
    results.emplace_back(frame_map_[addr]);  // 将结果加入到输出列表中
  }
  return results;  // 返回符号化后的结果列表
}

} // namespace torch::unwind

#endif
```