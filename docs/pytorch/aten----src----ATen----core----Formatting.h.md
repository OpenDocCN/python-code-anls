# `.\pytorch\aten\src\ATen\core\Formatting.h`

```
#pragma once
// 预处理指令，确保此头文件只被编译一次

#include <ostream>
// 包含输出流的标准头文件

#include <string>
// 包含处理字符串的标准头文件

#include <c10/core/Scalar.h>
// 包含 C10 库中的 Scalar 类头文件

#include <ATen/core/Tensor.h>
// 包含 ATen 库中的 Tensor 类头文件

namespace c10 {
// 定义 c10 命名空间

TORCH_API std::ostream& operator<<(std::ostream& out, Backend b);
// 声明输出流中打印 Backend 枚举类型的重载操作符函数

TORCH_API std::ostream& operator<<(std::ostream & out, const Scalar& s);
// 声明输出流中打印 Scalar 类型的重载操作符函数

TORCH_API std::string toString(const Scalar& s);
// 声明将 Scalar 类型转换为字符串的函数
}

namespace at {
// 定义 at 命名空间

TORCH_API std::ostream& operator<<(std::ostream& out, const DeprecatedTypeProperties& t);
// 声明输出流中打印 DeprecatedTypeProperties 结构的重载操作符函数

TORCH_API std::ostream& print(
    std::ostream& stream,
    const Tensor& tensor,
    int64_t linesize);
// 声明输出流中打印 Tensor 类型的打印函数，可以指定每行的字符数

static inline std::ostream& operator<<(std::ostream & out, const Tensor & t) {
  return print(out,t,80);
}
// 定义 inline 的重载操作符函数，输出流中打印 Tensor 对象，每行默认为 80 个字符

TORCH_API void print(const Tensor & t, int64_t linesize=80);
// 声明输出 Tensor 对象内容到标准输出的函数，可以指定每行的字符数
}
```