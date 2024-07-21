# `.\pytorch\aten\src\ATen\miopen\Types.h`

```
#pragma once

# 使用 `#pragma once` 预处理指令，确保头文件只被编译一次，以避免重复包含。


#include <ATen/miopen/miopen-wrapper.h>
#include <ATen/Tensor.h>

# 包含了两个头文件：
# - `<ATen/miopen/miopen-wrapper.h>`：包含了与 MIOpen 相关的包装函数和类型声明。
# - `<ATen/Tensor.h>`：包含了 PyTorch 的张量（Tensor）相关的定义和声明。


namespace at { namespace native {

# 进入命名空间 `at::native`，用于封装与 PyTorch 张量相关的本地函数和类型。


miopenDataType_t getMiopenDataType(const at::Tensor& tensor);

# 声明一个函数 `getMiopenDataType`，用于获取给定 PyTorch 张量的 MIOpen 数据类型。


int64_t miopen_version();

# 声明一个函数 `miopen_version`，用于获取当前 MIOpen 的版本号。


}}  // namespace at::miopen

# 结束命名空间 `at::native` 和 `at::miopen`。
```