# `.\pytorch\torch\csrc\utils\tensor_memoryformats.h`

```
#pragma once

// 使用 `#pragma once` 预处理指令，确保头文件只被包含一次，防止重复包含导致的编译错误


#include <c10/core/MemoryFormat.h>
#include <torch/csrc/Export.h>
#include <torch/csrc/utils/python_stub.h>

// 包含必要的头文件，分别为 `c10::MemoryFormat` 的定义、Torch 的导出声明和 Python 语言的存根函数声明


namespace torch::utils {

// 进入命名空间 `torch::utils`，定义一组实用函数和工具类的范围


void initializeMemoryFormats();

// 声明函数 `initializeMemoryFormats()`，用于初始化内存格式相关的设置


// This methods returns a borrowed reference!
TORCH_PYTHON_API PyObject* getTHPMemoryFormat(c10::MemoryFormat);

// 声明 `getTHPMemoryFormat` 函数，接受 `c10::MemoryFormat` 类型参数，并使用 `TORCH_PYTHON_API` 修饰符表示其在 Python 中的 API 接口，返回一个借用的 Python 对象引用 (`PyObject*`)


} // namespace torch::utils

// 结束 `torch::utils` 命名空间声明
```