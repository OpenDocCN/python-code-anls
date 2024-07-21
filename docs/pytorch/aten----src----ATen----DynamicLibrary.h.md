# `.\pytorch\aten\src\ATen\DynamicLibrary.h`

```
#pragma once
// 预处理指令，确保头文件只包含一次

#include <ATen/Utils.h>
// 包含 ATen 库中的 Utils 头文件

#include <c10/macros/Export.h>
// 包含 c10 库中的 Export 宏定义头文件

#include <c10/util/Exception.h>
// 包含 c10 库中的 Exception 头文件

namespace c10 {
// 进入 c10 命名空间

class DynamicLibraryError : public Error {
  using Error::Error;
  // 定义 DynamicLibraryError 类，继承自 Error 类
  // 使用 Error 类的构造函数
};

} // namespace c10
// 退出 c10 命名空间

namespace at {
// 进入 at 命名空间

struct DynamicLibrary {
  AT_DISALLOW_COPY_AND_ASSIGN(DynamicLibrary);
  // 定义 DynamicLibrary 结构体，并禁止复制和赋值操作

  TORCH_API DynamicLibrary(
      const char* name,
      const char* alt_name = nullptr,
      bool leak_handle = false);
  // 声明 DynamicLibrary 结构体的构造函数，接受库名、备选名和是否泄漏句柄的参数

  TORCH_API void* sym(const char* name);
  // 声明 sym 方法，用于获取库中符号的地址

  TORCH_API ~DynamicLibrary();
  // 声明 DynamicLibrary 结构体的析构函数

 private:
  bool leak_handle;
  // 成员变量，标记是否泄漏句柄
  void* handle = nullptr;
  // 成员变量，动态库句柄，默认为 nullptr，用于存储库的句柄
};

} // namespace at
// 退出 at 命名空间
```