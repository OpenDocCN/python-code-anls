# `.\pytorch\torch\csrc\cuda\THCP.h`

```
#ifndef THCP_H
#define THCP_H
// 如果 THCP_H 宏未定义，则定义它，用于防止头文件被多次包含

#include <torch/csrc/THP.h>
// 包含 Torch 的 THP.h 头文件，该文件提供了与 Python 的交互功能

#include <torch/csrc/cuda/Event.h>
// 包含 Torch CUDA 相关的事件处理头文件

#include <torch/csrc/cuda/Module.h>
// 包含 Torch CUDA 模块处理头文件

#include <torch/csrc/cuda/Stream.h>
// 包含 Torch CUDA 流处理头文件

#include <torch/csrc/python_headers.h>
// 包含 Torch 使用的 Python 头文件

#endif
// 结束宏定义区域，确保头文件内容只被编译一次
```