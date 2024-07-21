# `.\pytorch\aten\src\ATen\ParallelFuture.h`

```
#pragma once
// 预处理指令：防止头文件被多次包含

#include <ATen/core/ivalue.h>
// 包含 ATen 库的 IValue 头文件

#include <c10/macros/Macros.h>
// 包含 c10 库的宏定义头文件

#include <functional>
// 包含 C++ 标准库中的函数对象头文件

namespace at {
// 命名空间 at，包含了 ATen 库的内容

// 启动一个内部操作的并行任务，返回一个 Future 对象
TORCH_API c10::intrusive_ptr<c10::ivalue::Future> intraop_launch_future(
    std::function<void()> func);
    // 函数签名：接受一个无返回值的函数对象 func 作为参数，返回一个指向 Future 对象的智能指针

} // namespace at
// 命名空间结束
```