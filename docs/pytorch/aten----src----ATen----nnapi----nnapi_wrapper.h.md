# `.\pytorch\aten\src\ATen\nnapi\nnapi_wrapper.h`

```
// 版权声明和许可证信息，表明此文件受 Apache 许可证 2.0 版本保护
#ifndef NNAPI_WRAPPER_H_
#define NNAPI_WRAPPER_H_
// 包含所需的标准头文件：大小和整数类型
#include <stddef.h>
#include <stdint.h>
// 包含 ATen 库的 NNAPI NeuralNetworks 头文件
#include <ATen/nnapi/NeuralNetworks.h>
// C++ 结束声明
};
// 如果是 C++ 环境，声明一个函数 nnapi_wrapper_load，用于加载 NNAPI 的包装器和检查 NNAPI 的包装器
#ifdef __cplusplus
void nnapi_wrapper_load(struct nnapi_wrapper** nnapi, struct nnapi_wrapper** check_nnapi);
#endif
// 结束头文件保护宏定义
#endif
```