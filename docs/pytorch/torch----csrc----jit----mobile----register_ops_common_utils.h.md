# `.\pytorch\torch\csrc\jit\mobile\register_ops_common_utils.h`

```py
#pragma once
// 预处理指令，确保本文件只被编译一次

#include <ATen/Context.h>
// 包含 ATen 库的 Context 模块，提供与计算设备相关的功能
#include <ATen/NativeFunctions.h>
// 包含 ATen 库的 NativeFunctions 模块，提供张量操作的原生函数
#include <ATen/core/ivalue.h>
// 包含 ATen 库的 ivalue 模块，定义了表示任意值的 IValue 类型
#include <ATen/core/stack.h>
// 包含 ATen 库的 stack 模块，定义了用于操作栈的 Stack 类
#include <torch/csrc/jit/runtime/jit_exception.h>
// 包含 torch 库的 jit/runtime/jit_exception.h 文件，定义了 JIT 运行时的异常
#include <torch/csrc/jit/runtime/vararg_functions.h>
// 包含 torch 库的 jit/runtime/vararg_functions.h 文件，定义了支持变长参数函数的功能

namespace torch {
namespace jit {

inline void noop(Stack& n) {}
// 内联函数 noop，接受一个 Stack 引用，执行空操作

int64_t normalizeIndex(int64_t idx, int64_t list_size);
// 声明函数 normalizeIndex，用于规范化索引，参数为索引值和列表大小

// reference function THPVariable_to in python_variable_methods.cpp
// 引用 python_variable_methods.cpp 文件中的 THPVariable_to 函数

static C10_UNUSED at::Tensor to_dispatch(
    at::Tensor self,
    std::optional<at::Device> device,
    std::optional<at::ScalarType> scalarType,
    bool non_blocking,
    bool copy) {
  // 静态函数 to_dispatch，根据输入参数对张量 self 进行转换，并返回转换后的张量

  if (device && device->is_cuda()) {
    // 如果指定了设备且设备是 CUDA 设备，则初始化 CUDA 上下文
    at::globalContext().lazyInitCUDA();
  }

  if (!device && !scalarType && !copy) {
    // 如果未指定设备、标量类型和拷贝标志，则返回原始张量 self
    return self;
  } else if (!device) {
    // 如果未指定设备，则按照标量类型 scalarType 进行转换
    return self.to(*scalarType, non_blocking, copy);
  } else if (!scalarType) {
    // 如果未指定标量类型，则按照设备 device 进行转换
    return self.to(*device, non_blocking, copy);
  } else {
    // 同时指定了设备和标量类型，则按照指定的设备和标量类型进行转换
    return self.to(*device, *scalarType, non_blocking, copy);
  }
}

// Convert the tensor pointed to by \p data to a nested list. \p dim is the
// number of dimensions in the tensor and \p cur_dim is the dimension being
// processed by the current invocation. \p ty is the expected output IR type of
// the operation. \p is the scalar type of \p data. \p sizes and \p strides are
// the sizes and strides of the tensor operand and \p element_size is the size
// in bytes of one tensor element.
// 将由 data 指向的张量转换为嵌套列表。dim 是张量的维数，cur_dim 是当前调用处理的维度。
// ty 是操作的期望输出 IR 类型。data 的标量类型是 scalar_ty。
// sizes 和 strides 是张量操作数的大小和步长，element_size 是张量元素的字节大小。
IValue tensorToListRecursive(
    char* data,
    int64_t cur_dim,
    int64_t num_tensor_dims,
    at::TypePtr ty,
    at::ScalarType scalar_ty,
    at::IntArrayRef sizes,
    at::IntArrayRef strides,
    size_t element_size);

} // namespace jit
} // namespace torch
```