# `.\pytorch\aten\src\ATen\Dispatch.cpp`

```py
#include <ATen/Dispatch.h>
#if defined ENABLE_RECORD_KERNEL_FUNCTION_DTYPE
#include <ATen/record_function.h>

// 进入ATen库的详细命名空间
namespace at::detail {

// 定义记录内核函数数据类型的函数，接受一个字符串参数作为函数名
void record_kernel_function_dtype(std::string name) {
  // 使用ATen库提供的记录函数，记录内核函数的数据类型
  RECORD_FUNCTION_WITH_SCOPE(
        at::RecordScope::KERNEL_FUNCTION_DTYPE,
        std::move(name),
        c10::ArrayRef<const c10::IValue>{});
}

}  // 结束at::detail命名空间
#endif
```