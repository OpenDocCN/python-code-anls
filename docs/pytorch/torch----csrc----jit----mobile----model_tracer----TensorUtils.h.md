# `.\pytorch\torch\csrc\jit\mobile\model_tracer\TensorUtils.h`

```
#pragma once

// 使用 `#pragma once` 预处理指令，确保头文件只被编译一次，避免重复包含


#include <ATen/core/ivalue.h>

// 包含 ATen 库的 IValue 头文件，用于处理 Torch 中的值对象


namespace torch {
namespace jit {
namespace mobile {

// 声明命名空间 `torch::jit::mobile`，用于组织相关的代码，提供模块化和作用域限定


/**
 * Recursively scan the IValue object, traversing lists, tuples, dicts, and stop
 * and call the user provided callback function 'func' when a Tensor is found.
 */

// 函数说明文档注释：
// 递归扫描 IValue 对象，遍历列表、元组、字典，当找到张量时调用用户提供的回调函数 'func'。


void for_each_tensor_in_ivalue(
    const ::c10::IValue& iv,
    std::function<void(const ::at::Tensor&)> const& func);

// 声明函数 `for_each_tensor_in_ivalue`，接受一个常引用的 `c10::IValue` 对象 `iv`，
// 以及一个常引用的函数对象 `func`，该函数对象接受 `at::Tensor` 类型的参数，并无返回值。


} // namespace mobile
} // namespace jit
} // namespace torch

// 命名空间结束标记，结束了 `torch::jit::mobile` 命名空间的声明。
```