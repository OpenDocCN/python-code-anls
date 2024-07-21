# `.\pytorch\aten\src\ATen\BlasBackend.h`

```py
#pragma once
// 预处理指令，确保头文件只包含一次

#include <c10/util/Exception.h>
// 包含异常处理工具的头文件

#include <ostream>
// 包含输出流的头文件

#include <string>
// 包含字符串处理的头文件

namespace at {
// 命名空间开始

enum class BlasBackend : int8_t { Cublas, Cublaslt };
// 枚举类型 BlasBackend，表示 BLAS 后端，包括 Cublas 和 Cublaslt

inline std::string BlasBackendToString(at::BlasBackend backend) {
  // 内联函数，将 BlasBackend 枚举值转换为对应的字符串表示
  switch (backend) {
    case BlasBackend::Cublas:
      return "at::BlasBackend::Cublas";
    case BlasBackend::Cublaslt:
      return "at::BlasBackend::Cublaslt";
    default:
      // 如果传入的枚举值不在已知的 Cublas 或 Cublaslt 中，抛出异常
      TORCH_CHECK(false, "Unknown blas backend");
  }
}

inline std::ostream& operator<<(std::ostream& stream, at::BlasBackend backend) {
  // 重载流插入运算符，使得可以将 BlasBackend 枚举值输出到流中
  return stream << BlasBackendToString(backend);
}

} // namespace at
// 命名空间结束
```