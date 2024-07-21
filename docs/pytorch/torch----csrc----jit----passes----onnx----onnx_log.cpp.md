# `.\pytorch\torch\csrc\jit\passes\onnx\onnx_log.cpp`

```
// 包含 Torch 的 ONNX 模块日志相关头文件
#include <torch/csrc/jit/passes/onnx/onnx_log.h>
// 包含输入输出流的标准库
#include <iostream>

// 定义 Torch JIT 的命名空间
namespace torch {
namespace jit {
namespace onnx {

// 匿名命名空间，用于限定变量作用域
namespace {
// 日志开关，默认关闭
bool log_enabled = false;
// 输出流的共享指针
std::shared_ptr<std::ostream> out;
} // namespace

// 判断日志是否开启的函数
bool is_log_enabled() {
  return log_enabled;
}

// 设置日志开关状态的函数
void set_log_enabled(bool enabled) {
  log_enabled = enabled;
}

// 设置日志输出流的函数
void set_log_output_stream(std::shared_ptr<std::ostream> out_stream) {
  out = std::move(out_stream);
}

// 获取日志输出流的函数，如果未设置则使用标准输出流
std::ostream& _get_log_output_stream() {
  return out ? *out : std::cout;
}

} // namespace onnx
} // namespace jit
} // namespace torch
```