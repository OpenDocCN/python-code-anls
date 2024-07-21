# `.\pytorch\aten\src\ATen\native\cpu\SpmmReduceKernel.h`

```
#pragma once



// 使用 #pragma once 来确保头文件只被编译一次，避免多重包含问题

#include <ATen/core/Tensor.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/ReductionType.h>



// 引入 ATen 库的头文件，包括 Tensor 类和相关的调度声明和归约类型声明

namespace at::native {



// 进入 at::native 命名空间

using spmm_reduce_fn = void(*)(const Tensor&, const Tensor&, const Tensor&, const Tensor&, const Tensor&, ReductionType op);
using spmm_reduce_arg_fn = void(*)(const Tensor&, const Tensor&, const Tensor&, const Tensor&, const Tensor&, const Tensor&, ReductionType op);
using spmm_reduce_backward_input_fn = void(*)(const Tensor&, const Tensor&, const Tensor&, const Tensor&, const Tensor&, const Tensor&, ReductionType op);
using spmm_reduce_backward_input_arg_fn = void(*)(const Tensor&, const Tensor&, const Tensor&, const Tensor&, const Tensor&, ReductionType op);
using spmm_reduce_backward_other_fn = void(*)(const Tensor&, const Tensor&, const Tensor&, const Tensor&, const Tensor&, const Tensor&, const Tensor&, ReductionType op);
using spmm_reduce_backward_input_arg_fn = void(*)(const Tensor&, const Tensor&, const Tensor&, const Tensor&, const Tensor&, ReductionType op);



// 定义了一系列函数指针类型，用于声明 spmm_reduce 系列函数的接口，参数类型包括 Tensor 和 ReductionType

DECLARE_DISPATCH(spmm_reduce_fn, spmm_reduce_stub);
DECLARE_DISPATCH(spmm_reduce_arg_fn, spmm_reduce_arg_stub);
DECLARE_DISPATCH(spmm_reduce_backward_input_fn, spmm_reduce_backward_input_stub);
DECLARE_DISPATCH(spmm_reduce_backward_input_arg_fn, spmm_reduce_backward_input_arg_stub);
DECLARE_DISPATCH(spmm_reduce_backward_other_fn, spmm_reduce_backward_other_stub);
DECLARE_DISPATCH(spmm_reduce_backward_input_arg_fn, spmm_reduce_backward_other_arg_stub);



// 使用 DECLARE_DISPATCH 宏来声明不同的 spmm_reduce 相关函数的调度接口

} // at::native



// 结束 at::native 命名空间声明


这些注释按照要求为给定的 C++ 头文件代码添加了解释和说明。
```