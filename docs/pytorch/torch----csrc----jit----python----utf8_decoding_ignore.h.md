# `.\pytorch\torch\csrc\jit\python\utf8_decoding_ignore.h`

```py
#pragma once
// 使用 #pragma once 指令，确保头文件只被编译一次，防止多重包含

#include <torch/csrc/Export.h>
// 包含 Torch 库的导出头文件

namespace torch::jit {
// 定义命名空间 torch::jit

TORCH_API void setUTF8DecodingIgnore(bool o);
// 声明一个函数 setUTF8DecodingIgnore，用于设置 UTF-8 解码时是否忽略错误，接受一个布尔参数

TORCH_API bool getUTF8DecodingIgnore();
// 声明一个函数 getUTF8DecodingIgnore，用于获取当前是否忽略 UTF-8 解码错误的设置，并返回布尔值

} // namespace torch::jit
// 结束命名空间 torch::jit
```