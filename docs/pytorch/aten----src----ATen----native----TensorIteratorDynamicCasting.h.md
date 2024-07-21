# `.\pytorch\aten\src\ATen\native\TensorIteratorDynamicCasting.h`

```
#pragma once
// 预处理指令，确保本头文件只被编译一次

#include <complex>
// 包含复数的标准库头文件

#include <type_traits>
// 包含类型特性的标准库头文件

#include <c10/core/ScalarType.h>
// 包含标量类型定义的头文件

#include <ATen/detail/FunctionTraits.h>
// 包含函数特性的头文件

#include <ATen/native/TensorIterator.h>
// 包含张量迭代器的头文件

// This file includes utilities for dynamic_casting done by TensorIterator, see CUDALoops.cuh and Loops.h.

// dynamic_casting handles when the types expected by the iterator do not match the types of the arguments
// to the function that is being called.
// On CUDA, the cast is currently pushed down into the kernel (for performance reasons).
// On CPU, there is currently an internal assert that a dynamic_cast is not needed.

namespace at::native {

// `needs_dynamic_casting` compares the types expected by iterator
// (i.e. dtypes of the operands) with the actual type of the arguments
// (and returns) of func_t
template<typename func_t, int nargs=function_traits<func_t>::arity>
struct needs_dynamic_casting {
  static bool check(TensorIteratorBase& iter) {
    using traits = function_traits<func_t>;
    using cpp_type = typename traits::template arg<nargs - 1>::type;
    using cpp_map = c10::CppTypeToScalarType<cpp_type>;

    // 检查迭代器所期望的操作数的数据类型与实际函数参数的数据类型是否匹配
    if (iter.input_dtype(nargs-1) != cpp_map::value) {
      return true;
    }
    // 递归检查前一个参数
    return needs_dynamic_casting<func_t, nargs - 1>::check(iter);
  }
};

// 特化模板类 `needs_dynamic_casting`，处理参数个数为 0 的情况
template<typename func_t>
struct needs_dynamic_casting<func_t, 0> {
  static bool check(TensorIteratorBase& iter) {
    using traits = function_traits<func_t>;
    using cpp_type = typename traits::result_type;

    // 对于返回类型为 void 的函数，不需要动态类型转换
    if constexpr (std::is_void_v<cpp_type>) {
      return false;
    } else {
      // 检查输出的数据类型是否匹配函数的返回类型
      return iter.dtype(0) != c10::CppTypeToScalarType<cpp_type>::value;
    }
  }
};

} //namespace at::native
```