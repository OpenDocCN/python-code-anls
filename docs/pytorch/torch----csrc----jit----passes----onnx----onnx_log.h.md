# `.\pytorch\torch\csrc\jit\passes\onnx\onnx_log.h`

```
#pragma once
// 预处理指令，确保头文件仅被编译一次

#include <torch/csrc/Export.h>
// 引入 Torch 的导出宏定义头文件

#include <memory>
// 引入标准库中的内存管理头文件

#include <ostream>
// 引入标准输出流头文件

#include <string>
// 引入标准字符串处理头文件

namespace torch {
namespace jit {
namespace onnx {

TORCH_API bool is_log_enabled();
// 声明一个函数用于检查日志记录是否启用的 Torch API

TORCH_API void set_log_enabled(bool enabled);
// 声明一个函数用于设置是否启用日志记录的 Torch API

TORCH_API void set_log_output_stream(std::shared_ptr<std::ostream> out_stream);
// 声明一个函数用于设置日志输出流的 Torch API，传入一个共享指针指向输出流对象

TORCH_API std::ostream& _get_log_output_stream();
// 声明一个函数用于获取日志输出流的 Torch API，返回一个输出流的引用

#define ONNX_LOG(...)                            \
  if (::torch::jit::onnx::is_log_enabled()) {    \
    ::torch::jit::onnx::_get_log_output_stream() \
        << ::c10::str(__VA_ARGS__) << std::endl; \
  }
// 宏定义 ONNX_LOG，如果日志记录已启用，则将传入的参数作为字符串输出到日志输出流

} // namespace onnx
} // namespace jit
} // namespace torch
```