# `.\pytorch\torch\csrc\tensor\python_tensor.h`

```py
// 预处理指令，确保本头文件只被编译一次
#pragma once

// 引入C++标准库和Torch相关的头文件
#include <c10/core/Device.h>
#include <c10/core/DispatchKey.h>
#include <c10/core/ScalarType.h>
#include <torch/csrc/python_headers.h>

// Torch命名空间中的at命名空间，定义了Tensor类
namespace at {
    class Tensor;
} // namespace at

// Torch命名空间中的tensors命名空间，包含了Tensor相关的函数和类型
namespace torch {
namespace tensors {

// 初始化Python中的张量类型对象：torch.FloatTensor、torch.DoubleTensor等，并将它们绑定到相应的模块中
void initialize_python_bindings();

// 设置默认的张量类型，接受一个PyObject*参数
void py_set_default_tensor_type(PyObject* type_obj);

// 设置默认的数据类型（ScalarType），接受一个PyObject*参数
void py_set_default_dtype(PyObject* dtype_obj);

// 获取默认张量类型的调度键（DispatchKey）
//
// TODO: 这太复杂了！没有理由让默认张量类型id改变。可能只需要存储ScalarType，因为这是我们支持的唯一弹性点。
TORCH_API c10::DispatchKey get_default_dispatch_key();

// 获取默认设备
at::Device get_default_device();

// 获取默认张量类型的标量类型（ScalarType）
at::ScalarType get_default_scalar_type();

} // namespace tensors
} // namespace torch
```