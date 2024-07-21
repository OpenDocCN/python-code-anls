# `.\pytorch\aten\src\ATen\miopen\Exceptions.h`

```
#pragma once
// 预处理指令，确保本文件只被编译一次

#include <ATen/miopen/miopen-wrapper.h>
// 包含 miopen 库的头文件 miopen-wrapper.h

#include <string>
// 包含标准字符串处理库

#include <stdexcept>
// 包含标准异常处理库

#include <sstream>
// 包含标准字符串流处理库

namespace at { namespace native {

class miopen_exception : public std::runtime_error {
// miopen_exception 类继承自 std::runtime_error 类
public:
  miopenStatus_t status;
  // miopen 状态变量

  miopen_exception(miopenStatus_t status, const char* msg)
      : std::runtime_error(msg)
      , status(status) {}
  // miopen_exception 构造函数，使用给定的状态和消息构造 std::runtime_error

  miopen_exception(miopenStatus_t status, const std::string& msg)
      : std::runtime_error(msg)
      , status(status) {}
  // miopen_exception 构造函数，使用给定的状态和消息字符串构造 std::runtime_error
};

inline void MIOPEN_CHECK(miopenStatus_t status)
{
  // MIOPEN_CHECK 函数，检查 miopen 操作的状态
  if (status != miopenStatusSuccess) {
    // 如果状态不是成功状态
    if (status == miopenStatusNotImplemented) {
        // 如果状态是未实现
        throw miopen_exception(status, std::string(miopenGetErrorString(status)) +
                ". This error may appear if you passed in a non-contiguous input.");
        // 抛出 miopen_exception 异常，包含状态和错误消息
    }
    throw miopen_exception(status, miopenGetErrorString(status));
    // 抛出 miopen_exception 异常，包含状态和错误消息
  }
}

inline void HIP_CHECK(hipError_t error)
{
  // HIP_CHECK 函数，检查 HIP 操作的状态
  if (error != hipSuccess) {
    // 如果错误状态不是成功状态
    std::string msg("HIP error: ");
    // 创建错误消息字符串
    msg += hipGetErrorString(error);
    // 将 HIP 错误消息追加到字符串末尾
    throw std::runtime_error(msg);
    // 抛出 std::runtime_error 异常，包含错误消息
  }
}

}} // namespace at::native
// 命名空间声明结束
```