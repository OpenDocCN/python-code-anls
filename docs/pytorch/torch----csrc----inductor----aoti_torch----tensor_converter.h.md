# `.\pytorch\torch\csrc\inductor\aoti_torch\tensor_converter.h`

```py
#pragma once

#include <ATen/Tensor.h>
#include <torch/csrc/inductor/aoti_torch/c/shim.h>

namespace torch {
namespace aot_inductor {

// Functions declared here are not meant to be called from the AOTInductor
// generated model.so

// unsafe_alloc_new_handles_from_tensors is used for allocating new aten
// tensor objects and return them as a vector of AtenTensorHandle (raw
// pointers), and those pointers will be stolen by model.so.
// 从给定的 at::Tensor 向量中分配新的 aten 张量对象，并将它们作为 AtenTensorHandle（原始指针）向量返回，
// 这些指针将被 model.so 接管。
TORCH_API std::vector<AtenTensorHandle> unsafe_alloc_new_handles_from_tensors(
    std::vector<at::Tensor>& tensors);

// alloc_tensors_by_stealing_from_handles is used for creating a vector of aten
// tensors by stealing from an array of handles. Only the handles are stolen,
// and the array itself is borrowed.
//
// WARNING: Can NOT be called in model.so unless in the non-ABI-compatible mode
// 通过从 handles 数组中窃取数据来创建 aten 张量向量。仅窃取句柄，数组本身是借用的。
//
// 警告：除非处于非 ABI 兼容模式，否则不可以在 model.so 中调用此函数。
TORCH_API std::vector<at::Tensor> alloc_tensors_by_stealing_from_handles(
    AtenTensorHandle* handles,
    size_t length);

} // namespace aot_inductor
} // namespace torch
```