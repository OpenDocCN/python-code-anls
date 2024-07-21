# `.\pytorch\torch\csrc\distributed\c10d\exception.h`

```
// 版权声明和许可证声明，声明此源代码受BSD风格许可证保护，许可证文件位于根目录的LICENSE文件中
// 版权所有，Facebook公司及其关联公司保留所有权利。
//
// 此源代码受BSD风格许可证保护，许可证文件位于根目录的LICENSE文件中。

// 预编译指令，确保头文件只被包含一次
#pragma once

// 包含标准异常类
#include <stdexcept>

// 包含C10库的宏定义
#include <c10/macros/Macros.h>
// 包含C10库的异常处理类
#include <c10/util/Exception.h>

// 定义宏，类似于C10_THROW_ERROR，但是用于c10d命名空间下的异常类型
#define C10D_THROW_ERROR(err_type, msg) \
  throw ::c10d::err_type(               \
      {__func__, __FILE__, static_cast<uint32_t>(__LINE__)}, msg)

// c10d命名空间
namespace c10d {

// 使用c10::DistNetworkError作为基类的SocketError异常类
class TORCH_API SocketError : public DistNetworkError {
  using DistNetworkError::DistNetworkError;
};

// 使用c10::DistNetworkError作为基类的TimeoutError异常类
class TORCH_API TimeoutError : public DistNetworkError {
  using DistNetworkError::DistNetworkError;
};

} // namespace c10d
```