# `.\pytorch\torch\csrc\utils\tensor_numpy.h`

```
#pragma once

// 包含 ATen 库中的 Tensor 类
#include <ATen/core/Tensor.h>
// 包含 Python 头文件，用于与 Python 交互
#include <torch/csrc/python_headers.h>

// 定义了 torch::utils 命名空间，包含了一系列与 PyTorch 和 NumPy 交互相关的函数

namespace torch::utils {

// 将 ATen 的 Tensor 对象转换为 NumPy 的 PyObject 对象
PyObject* tensor_to_numpy(const at::Tensor& tensor, bool force = false);

// 根据 NumPy 的 PyObject 对象创建 ATen 的 Tensor 对象
at::Tensor tensor_from_numpy(PyObject* obj, bool warn_if_not_writeable = true);

// 将 ATen 的 ScalarType 转换为 NumPy 的 dtype
int aten_to_numpy_dtype(const at::ScalarType scalar_type);

// 将 NumPy 的 dtype 转换为 ATen 的 ScalarType
at::ScalarType numpy_dtype_to_aten(int dtype);

// 检查系统中是否可用 NumPy 库
bool is_numpy_available();

// 检查 PyObject 是否为 NumPy 的整数类型
bool is_numpy_int(PyObject* obj);

// 检查 PyObject 是否为 NumPy 的布尔类型
bool is_numpy_bool(PyObject* obj);

// 检查 PyObject 是否为 NumPy 的标量（scalar）
bool is_numpy_scalar(PyObject* obj);

// 发出警告，指出 NumPy 数组不可写
void warn_numpy_not_writeable();

// 根据 CUDA 数组接口创建 ATen 的 Tensor 对象
at::Tensor tensor_from_cuda_array_interface(PyObject* obj);

// 验证 NumPy 是否存在与 DLPack 删除器相关的 bug
void validate_numpy_for_dlpack_deleter_bug();

// 检查 NumPy 是否存在与 DLPack 删除器相关的 bug
bool is_numpy_dlpack_deleter_bugged();

} // namespace torch::utils
```