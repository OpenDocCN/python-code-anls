# `.\pytorch\torch\csrc\jit\runtime\print_handler.cpp`

```
#include <torch/csrc/jit/runtime/print_handler.h>
// 包含打印处理器的头文件

#include <atomic>
// 包含原子操作的头文件
#include <iostream>
// 包含输入输出流的头文件
#include <string>
// 包含字符串处理的头文件

namespace torch::jit {

namespace {
// torch::jit 命名空间下的匿名命名空间

std::atomic<PrintHandler> print_handler(getDefaultPrintHandler());
// 原子操作的打印处理器对象，初始化为默认打印处理器

} // namespace

PrintHandler getDefaultPrintHandler() {
  return [](const std::string& s) { std::cout << s; };
}
// 返回默认的打印处理器，输出字符串到标准输出流

PrintHandler getPrintHandler() {
  return print_handler.load();
}
// 获取当前的打印处理器

void setPrintHandler(PrintHandler ph) {
  print_handler.store(ph);
}
// 设置新的打印处理器

} // namespace torch::jit
// torch::jit 命名空间结束
```