# `.\pytorch\torch\csrc\api\include\torch\all.h`

```
#pragma once

#if !defined(_MSC_VER) && __cplusplus < 201703L
#error C++17 or later compatible compiler is required to use PyTorch.
#endif

#include <torch/autograd.h>   // 包含 PyTorch 的自动求导功能头文件
#include <torch/cuda.h>       // 包含 PyTorch 的 CUDA 功能头文件
#include <torch/data.h>       // 包含 PyTorch 的数据处理头文件
#include <torch/enum.h>       // 包含 PyTorch 的枚举类型头文件
#include <torch/fft.h>        // 包含 PyTorch 的 FFT 处理头文件
#include <torch/jit.h>        // 包含 PyTorch 的 JIT 编译头文件
#include <torch/linalg.h>     // 包含 PyTorch 的线性代数头文件
#include <torch/mps.h>        // 包含 PyTorch 的内存池系统头文件
#include <torch/nested.h>     // 包含 PyTorch 的嵌套类型头文件
#include <torch/nn.h>         // 包含 PyTorch 的神经网络头文件
#include <torch/optim.h>      // 包含 PyTorch 的优化算法头文件
#include <torch/serialize.h>  // 包含 PyTorch 的序列化功能头文件
#include <torch/sparse.h>     // 包含 PyTorch 的稀疏矩阵处理头文件
#include <torch/special.h>    // 包含 PyTorch 的特殊数学函数头文件
#include <torch/types.h>      // 包含 PyTorch 的数据类型定义头文件
#include <torch/utils.h>      // 包含 PyTorch 的实用工具头文件
#include <torch/version.h>    // 包含 PyTorch 的版本信息头文件
#include <torch/xpu.h>        // 包含 PyTorch 的异构计算设备头文件
```