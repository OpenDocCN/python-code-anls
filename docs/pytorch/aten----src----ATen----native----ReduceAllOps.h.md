# `.\pytorch\aten\src\ATen\native\ReduceAllOps.h`

```py
#pragma once


// 指令：#pragma once
// 作用：确保头文件只被编译一次，避免重复包含

#include <ATen/native/DispatchStub.h>


// 包含头文件：ATen/native/DispatchStub.h
// 作用：引入 DispatchStub.h 头文件，其中可能包含了分发函数的声明和定义

namespace at {
class Tensor;
}


// 命名空间：at
// 作用：定义了 at 命名空间，用于存放与 Tensor 相关的类和函数

namespace at::native {


// 命名空间：at::native
// 作用：定义了 at 命名空间下的 native 命名空间，用于存放与本地（native）相关的函数和类型

using reduce_all_fn = void (*)(Tensor & result, const Tensor & self);


// 定义：using reduce_all_fn = void (*)(Tensor & result, const Tensor & self);
// 类型：reduce_all_fn 是一个函数指针类型，指向一个函数，该函数接受两个 Tensor 引用参数，并返回 void

using reduce_min_max_fn = void (*)(Tensor & max_result, Tensor & min_result, const Tensor & self);


// 定义：using reduce_min_max_fn = void (*)(Tensor & max_result, Tensor & min_result, const Tensor & self);
// 类型：reduce_min_max_fn 是一个函数指针类型，指向一个函数，该函数接受三个 Tensor 引用参数，并返回 void

DECLARE_DISPATCH(reduce_all_fn, min_all_stub);


// 宏定义：DECLARE_DISPATCH(reduce_all_fn, min_all_stub);
// 作用：声明一个名为 min_all_stub 的分发函数，该函数由宏 DECLARE_DISPATCH 定义，接受 reduce_all_fn 类型的函数指针作为参数

DECLARE_DISPATCH(reduce_all_fn, max_all_stub);


// 宏定义：DECLARE_DISPATCH(reduce_all_fn, max_all_stub);
// 作用：声明一个名为 max_all_stub 的分发函数，该函数由宏 DECLARE_DISPATCH 定义，接受 reduce_all_fn 类型的函数指针作为参数

} // namespace at::native


// 命名空间结束：at::native
// 作用：结束 at::native 命名空间的定义，确保其中的声明和定义不会影响其它命名空间的内容
```