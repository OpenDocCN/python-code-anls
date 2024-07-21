# `.\pytorch\aten\src\ATen\native\cpu\StackKernel.h`

```py
// 声明一个命名空间，命名空间为 at::native，用于组织相关的函数和类型
namespace at { namespace native {

// 定义一个函数指针类型 stack_serial_fn，该函数接受一个 Tensor 引用、一个 TensorList（Tensor 的列表）、一个 int64_t 参数，并返回 void
using stack_serial_fn = void(*)(Tensor &, TensorList, int64_t);

// 声明一个函数声明宏，用于声明名为 stack_serial_stub 的函数指针，并将其与给定的分发名绑定
DECLARE_DISPATCH(stack_serial_fn, stack_serial_stub);

}}  // namespace at::native
```