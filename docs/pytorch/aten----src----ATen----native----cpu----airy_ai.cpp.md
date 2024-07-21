# `.\pytorch\aten\src\ATen\native\cpu\airy_ai.cpp`

```py
#define TORCH_ASSERT_NO_OPERATORS

这行代码定义了宏 `TORCH_ASSERT_NO_OPERATORS`，用于禁用 Torch 的运算符断言。


#include <ATen/native/UnaryOps.h>

包含了 `ATen` 库中的 `UnaryOps.h` 头文件，该文件包含了一元操作的定义和声明。


#include <ATen/Dispatch.h>
#include <ATen/native/Math.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cpu/Loops.h>

这几行代码分别包含了 `ATen` 库中的 `Dispatch.h`、`Math.h`、`TensorIterator.h` 和 `Loops.h` 头文件，这些文件提供了分发机制、数学函数、张量迭代器和 CPU 循环等功能。


namespace at::native {
inline namespace CPU_CAPABILITY {

定义了命名空间 `at::native`，并嵌套了内联命名空间 `CPU_CAPABILITY`。


static void airy_ai_kernel(TensorIteratorBase& iterator) {

定义了静态函数 `airy_ai_kernel`，该函数接受一个 `TensorIteratorBase` 的引用作为参数。


TORCH_INTERNAL_ASSERT(iterator.ntensors() == 2);

在 `airy_ai_kernel` 函数内部，使用 `TORCH_INTERNAL_ASSERT` 宏断言 `iterator.ntensors()` 的返回值为 2。


AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "airy_ai_cpu", [&]() {
    cpu_kernel(iterator, [](scalar_t x) {
        return airy_ai_forward(x);
    });
});

根据迭代器 `iterator` 的通用数据类型，使用 `AT_DISPATCH_FLOATING_TYPES` 宏分发到相应的浮点数类型，然后调用 `cpu_kernel` 函数，并传入一个 Lambda 表达式作为参数，Lambda 表达式中调用了 `airy_ai_forward(x)` 函数。


} // airy_ai_kernel(TensorIteratorBase& iterator)

结束 `airy_ai_kernel` 函数的定义。


} // namespace CPU_CAPABILITY

结束内联命名空间 `CPU_CAPABILITY` 的定义。


REGISTER_DISPATCH(special_airy_ai_stub, &CPU_CAPABILITY::airy_ai_kernel);

注册了名为 `special_airy_ai_stub` 的分发函数，指向 `CPU_CAPABILITY::airy_ai_kernel` 函数。


} // namespace at::native

结束 `at::native` 命名空间的定义。
```