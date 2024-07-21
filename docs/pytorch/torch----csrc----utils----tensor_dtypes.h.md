# `.\pytorch\torch\csrc\utils\tensor_dtypes.h`

```
#pragma once
// 预处理指令：确保本文件仅被编译一次

#include <c10/core/ScalarType.h>
// 包含 C10 库中的 ScalarType 类定义

#include <string>
// 包含 C++ 标准库中的 string 类定义

#include <tuple>
// 包含 C++ 标准库中的 tuple 类定义

namespace torch::utils {
// 命名空间 torch::utils，用于组织代码，避免命名冲突

std::pair<std::string, std::string> getDtypeNames(at::ScalarType scalarType);
// 函数声明：根据给定的 scalarType 返回对应的数据类型名称，返回值为 string 对组成的 pair

void initializeDtypes();
// 函数声明：初始化数据类型相关设置，无返回值

} // namespace torch::utils
// 命名空间结束
```