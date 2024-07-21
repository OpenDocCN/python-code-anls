# `.\pytorch\torch\csrc\utils\tensor_new.h`

```
// 一次性预处理指令，确保头文件只被包含一次
#pragma once

// 包含 Python 的头文件
#include <torch/csrc/python_headers.h>
// 包含 Python 参数解析的实用工具
#include <torch/csrc/utils/python_arg_parser.h>

// 包含 ATen 核心的张量定义
#include <ATen/core/Tensor.h>

// 定义了 torch::utils 命名空间，用于存放工具函数和类
namespace torch::utils {

// 注意事项区域开始
// [torch.tensor, lift_fresh, and device movement]
//
// 控制是否只将 CPU 张量转移到 CUDA 设备
//
// 如果 only_lift_cpu_tensors=false（默认行为）：
// - 数据会被转移到 CPU 张量
// - 然后通过 .to 方法转移到 CUDA 设备
// - 最后调用 lift_fresh() 函数处理
// 步骤 1 和 2 在所有模式禁用时都会执行。
//
// 如果 only_lift_cpu_tensors=true：
// - 数据会被转移到 CPU 张量（保持正确的数据类型）
// - 直接调用 lift_fresh() 函数处理
// - 最后通过 .to 方法转移到 CUDA 设备
// 步骤 1 在所有模式禁用时执行。
//
// only_lift_cpu_tensors=true 在 FakeTensorMode 下非常有用，因为它避免了将具体数据转移到 CUDA 设备导致的 CUDA 初始化。
TORCH_API bool only_lift_cpu_tensors();
TORCH_API void set_only_lift_cpu_tensors(bool value);

// 使用给定的 Python 参数构造基础张量
at::Tensor base_tensor_ctor(PyObject* args, PyObject* kwargs);

// 使用给定的分发键、标量类型以及 Python 参数构造传统张量
at::Tensor legacy_tensor_ctor(
    c10::DispatchKey dispatch_key,
    at::ScalarType scalar_type,
    PyObject* args,
    PyObject* kwargs);

// 使用给定的分发键、标量类型以及 Python 参数构造新的传统张量
at::Tensor legacy_tensor_new(
    c10::DispatchKey dispatch_key,
    at::ScalarType scalar_type,
    PyObject* args,
    PyObject* kwargs);

// 使用给定的选项、标量类型、设备（可选）、数据对象构造索引张量
at::Tensor indexing_tensor_from_data(
    c10::TensorOptions options,
    at::ScalarType scalar_type,
    std::optional<at::Device> device,
    PyObject* data);

// 使用给定的分发键、标量类型以及 Python 参数构造稀疏 COO 张量
at::Tensor sparse_coo_tensor_ctor(
    c10::DispatchKey dispatch_key,
    at::ScalarType scalar_type,
    PythonArgs& r);

// 验证稀疏 COO 张量构造函数的参数
void _validate_sparse_coo_tensor_args(
    c10::DispatchKey dispatch_key,
    at::ScalarType scalar_type,
    PyObject* args,
    PyObject* kwargs);

// 使用给定的分发键、标量类型以及 Python 参数构造稀疏压缩张量
at::Tensor sparse_compressed_tensor_ctor(
    c10::DispatchKey dispatch_key,
    at::ScalarType scalar_type,
    PythonArgs& r);

// 使用给定的分发键、标量类型以及 Python 参数构造稀疏 CSR 张量
at::Tensor sparse_csr_tensor_ctor(
    c10::DispatchKey dispatch_key,
    at::ScalarType scalar_type,
    PythonArgs& r);

// 使用给定的分发键、标量类型以及 Python 参数构造稀疏 CSC 张量
at::Tensor sparse_csc_tensor_ctor(
    c10::DispatchKey dispatch_key,
    at::ScalarType scalar_type,
    PythonArgs& r);

// 使用给定的分发键、标量类型以及 Python 参数构造稀疏 BSR 张量
at::Tensor sparse_bsr_tensor_ctor(
    c10::DispatchKey dispatch_key,
    at::ScalarType scalar_type,
    PythonArgs& r);

// 使用给定的分发键、标量类型以及 Python 参数构造稀疏 BSC 张量
at::Tensor sparse_bsc_tensor_ctor(
    c10::DispatchKey dispatch_key,
    at::ScalarType scalar_type,
    PythonArgs& r);

// 验证稀疏压缩张量构造函数的参数
void _validate_sparse_compressed_tensor_args(
    c10::DispatchKey dispatch_key,
    at::ScalarType scalar_type,
    PyObject* args,
    PyObject* kwargs);

// 验证稀疏 CSR 张量构造函数的参数
void _validate_sparse_csr_tensor_args(
    c10::DispatchKey dispatch_key,
    at::ScalarType scalar_type,
    PyObject* args,
    PyObject* kwargs);

// 验证稀疏 CSC 张量构造函数的参数
void _validate_sparse_csc_tensor_args(
    c10::DispatchKey dispatch_key,
    at::ScalarType scalar_type,
    PyObject* args,
    PyObject* kwargs);

// 验证稀疏 BSR 张量构造函数的参数
void _validate_sparse_bsr_tensor_args(
    c10::DispatchKey dispatch_key,
    at::ScalarType scalar_type,
    PyObject* args,
    PyObject* kwargs);
// 验证稀疏 BSC 张量参数的有效性
void _validate_sparse_bsc_tensor_args(
    c10::DispatchKey dispatch_key,  // 分发键，用于确定张量操作的后端引擎
    at::ScalarType scalar_type,     // 张量的数据类型
    PyObject* args,                 // Python 参数元组
    PyObject* kwargs);              // Python 关键字参数字典

// 张量构造函数，根据分发键和数据类型创建张量对象
at::Tensor tensor_ctor(
    c10::DispatchKey dispatch_key,  // 分发键，决定张量操作的后端引擎
    at::ScalarType scalar_type,     // 张量的数据类型
    PythonArgs& r);                 // Python 参数对象

// 将 Python 对象转换为张量对象
at::Tensor as_tensor(
    c10::DispatchKey dispatch_key,  // 分发键，用于确定张量操作的后端引擎
    at::ScalarType scalar_type,     // 张量的数据类型
    PythonArgs& r);                 // Python 参数对象

// 创建新的张量对象，根据分发键、数据类型、Python 参数和关键字参数
at::Tensor new_tensor(
    c10::DispatchKey dispatch_key,  // 分发键，用于确定张量操作的后端引擎
    at::ScalarType scalar_type,     // 张量的数据类型
    PyObject* args,                 // Python 参数元组
    PyObject* kwargs);              // Python 关键字参数字典

// 创建全为 1 的新张量，根据分发键、数据类型、Python 参数和关键字参数
at::Tensor new_ones(
    c10::DispatchKey dispatch_key,  // 分发键，用于确定张量操作的后端引擎
    at::ScalarType scalar_type,     // 张量的数据类型
    PyObject* args,                 // Python 参数元组
    PyObject* kwargs);              // Python 关键字参数字典

// 从给定缓冲区创建张量，根据数据类型、元素数量、偏移量和是否需要梯度
at::Tensor tensor_frombuffer(
    PyObject* buffer,               // Python 缓冲区对象
    at::ScalarType dtype,           // 张量的数据类型
    int64_t count,                  // 元素数量
    int64_t offset,                 // 偏移量
    bool requires_grad);            // 是否需要计算梯度

// 从 DLPack 数据结构创建张量对象
at::Tensor tensor_fromDLPack(PyObject* data);  // DLPack 数据对象

// 将 Python 对象转换为张量对象，支持指定数据类型、设备和是否复制数据
at::Tensor asarray(
    PyObject* obj,                  // Python 对象
    std::optional<c10::ScalarType> dtype,   // 张量的数据类型（可选）
    std::optional<c10::Device> device,      // 张量的设备（可选）
    std::optional<bool> copy,       // 是否复制数据（可选）
    bool requires_grad);            // 是否需要计算梯度
} // namespace torch::utils
```