# `.\pytorch\aten\src\ATen\native\cpu\CopyKernel.h`

```py
#pragma once


// 使用#pragma once确保头文件只被编译一次，防止多重包含的问题
namespace at {
struct TensorIteratorBase;

namespace native {
inline namespace CPU_CAPABILITY {

// 声明函数direct_copy_kernel，接受TensorIteratorBase的引用参数
void direct_copy_kernel(TensorIteratorBase &iter);

// 声明函数copy_kernel，接受TensorIterator的引用参数和一个bool类型的参数（但在函数体内未使用）
void copy_kernel(TensorIterator& iter, bool /*non_blocking*/);

}}}  // namespace at::native::CPU_CAPABILITY
```