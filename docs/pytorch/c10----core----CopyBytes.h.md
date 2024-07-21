# `.\pytorch\c10\core\CopyBytes.h`

```py
#pragma once
// 防止头文件重复包含的预处理指令

#include <c10/core/Device.h>
// 引入设备相关的头文件，定义了设备类和相关操作

#include <c10/core/DeviceType.h>
// 引入设备类型相关的头文件，定义了设备类型枚举

#include <c10/macros/Export.h>
// 引入导出宏定义的头文件，用于声明导出符号

#include <c10/macros/Macros.h>
// 引入通用宏定义的头文件，包含了一些通用的宏定义

#include <cstddef>
// 引入标准库的 cstddef 头文件，定义了 size_t 类型

namespace c10 {
// 进入 c10 命名空间

using CopyBytesFunction = void (*)(
    size_t nbytes,
    const void* src,
    Device src_device,
    void* dst,
    Device dst_device);
// 定义了 CopyBytesFunction 类型别名，是一个函数指针类型，用于定义拷贝字节的函数签名

struct C10_API _CopyBytesFunctionRegisterer {
  _CopyBytesFunctionRegisterer(
      DeviceType from,
      DeviceType to,
      CopyBytesFunction func_sync,
      CopyBytesFunction func_async = nullptr);
};
// 定义了 _CopyBytesFunctionRegisterer 结构体，用于注册拷贝字节的函数实现

#define REGISTER_COPY_BYTES_FUNCTION(from, to, ...)           \
  namespace {                                                 \
  static _CopyBytesFunctionRegisterer C10_ANONYMOUS_VARIABLE( \
      g_copy_function)(from, to, __VA_ARGS__);                \
  }
// 定义了宏 REGISTER_COPY_BYTES_FUNCTION，用于注册拷贝字节的函数实现映射

/*
 * WARNING: Implementations for this function are currently registered from
 * ATen and caffe2, not yet from c10. Don't use this if not either ATen
 * or caffe2 is present as well.
 * We can't move them yet, because the CUDA implementations aren't unified yet
 * between ATen and caffe2.
 * We're planning to move the implementations into c10/backend/xxx
 * to make c10 self contained again.
 */
C10_API void CopyBytes(
    size_t nbytes,
    const void* src,
    Device src_device,
    void* dst,
    Device dst_device,
    bool async);
// 声明了 CopyBytes 函数原型，用于拷贝指定数量的字节数据，支持同步和异步模式

} // namespace c10
// 结束 c10 命名空间
```