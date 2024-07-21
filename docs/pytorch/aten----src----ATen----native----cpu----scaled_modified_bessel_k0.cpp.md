# `.\pytorch\aten\src\ATen\native\cpu\scaled_modified_bessel_k0.cpp`

```py
// 定义宏，禁用 Torch 运算符断言功能
#define TORCH_ASSERT_NO_OPERATORS

// 引入 ATen 库中的头文件，包括一元操作、分发、数学计算、张量迭代等功能
#include <ATen/native/UnaryOps.h>
#include <ATen/Dispatch.h>
#include <ATen/native/Math.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cpu/Loops.h>

// ATen 库的命名空间 at::native 中的内联命名空间 CPU_CAPABILITY
namespace at::native {
inline namespace CPU_CAPABILITY {

    // 定义静态函数 scaled_modified_bessel_k0_kernel，操作基于 TensorIteratorBase 对象
    static void scaled_modified_bessel_k0_kernel(TensorIteratorBase& iterator) {
        // 内部断言，确保 TensorIteratorBase 对象的张量数目为 2
        TORCH_INTERNAL_ASSERT(iterator.ntensors() == 2);

        // 使用 AT_DISPATCH_FLOATING_TYPES 宏，根据迭代器的公共数据类型分发操作
        AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "scaled_modified_bessel_k0_cpu", [&]() {
            // 调用 cpu_kernel 函数，对迭代器中每个标量进行操作，使用 scaled_modified_bessel_k0_forward 函数
            cpu_kernel(iterator, [](scalar_t x) {
                return scaled_modified_bessel_k0_forward(x);
            });
        });
    } // scaled_modified_bessel_k0_kernel(TensorIteratorBase& iterator)

} // namespace CPU_CAPABILITY

// 注册分发函数，将 special_scaled_modified_bessel_k0_stub 映射到 CPU_CAPABILITY 命名空间中的 scaled_modified_bessel_k0_kernel 函数
REGISTER_DISPATCH(special_scaled_modified_bessel_k0_stub, &CPU_CAPABILITY::scaled_modified_bessel_k0_kernel);

} // namespace at::native
```