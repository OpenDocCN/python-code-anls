# `.\pytorch\torch\csrc\inductor\aoti_runtime\interface.h`

```
#pragma once

// WARNING: Be careful when adding new includes here. This header will be used
// in model.so, and should not refer to any aten/c10 headers except the stable
// C ABI defined in torch/csrc/inductor/aoti_torch/c/shim.h. The same rule
// applies to other files under torch/csrc/inductor/aoti_runtime/.
#include <torch/csrc/inductor/aoti_runtime/utils.h>

extern "C" {
struct AOTInductorModelOpaque;
using AOTInductorModelHandle = AOTInductorModelOpaque*;

struct AOTInductorModelContainerOpaque;
using AOTInductorModelContainerHandle = AOTInductorModelContainerOpaque*;

struct AOTInductorStreamOpaque;
using AOTInductorStreamHandle = AOTInductorStreamOpaque*;

struct AOTInductorConstantMap;
using AOTInductorConstantMapHandle = AOTInductorConstantMap*;

// TODO: Deprecate this API. This was kept for BC compatibility.
// Please use AOTInductorModelContainerCreateWithDevice instead.
// 创建一个 AOTInductor 模型容器。参数 num_models 指定可以同时运行相同输入模型的模型实例数。
// cubin_dir 是包含 CUDA 二进制对象的目录路径。
AOTIRuntimeError AOTInductorModelContainerCreate(
    AOTInductorModelContainerHandle* container_handle,
    size_t num_models,
    bool is_cpu,
    const char* cubin_dir);

// 使用指定设备创建 AOTInductor 模型容器。参数 num_models 指定可以同时运行相同输入模型的模型实例数。
// device_str 必须是有效的设备字符串，例如 "cpu"、"cuda"、"cuda:0" 等。
// cubin_dir 是包含 CUDA 二进制对象的目录路径。
AOTIRuntimeError AOTInductorModelContainerCreateWithDevice(
    AOTInductorModelContainerHandle* container_handle,
    size_t num_models,
    const char* device_str,
    const char* cubin_dir);

// 删除 AOTInductor 模型容器。
AOTIRuntimeError AOTInductorModelContainerDelete(
    AOTInductorModelContainerHandle container_handle);

// 运行推理过程。
// input_handles 是输入 AtenTensorHandle 数组；这些句柄会被接管，但数组本身是借用的。
// num_inputs 是输入数量。
// output_handles 是用于写入输出 AtenTensorHandle 的数组；这些句柄将由调用方接管，但数组本身是借用的。
// num_outputs 是输出数量。
// stream_handle 是 AOTInductor 流句柄。
// proxy_executor_handle 是 AOTI 代理执行句柄。
AOTIRuntimeError AOTInductorModelContainerRun(
    AOTInductorModelContainerHandle container_handle,
    AtenTensorHandle* input_handles,
    size_t num_inputs,
    AtenTensorHandle* output_handles,
    size_t num_outputs,
    AOTInductorStreamHandle stream_handle,
    AOTIProxyExecutorHandle proxy_executor_handle);

// 获取模型的常量数量。
AOTIRuntimeError AOTInductorModelContainerGetNumConstants(
    AOTInductorModelContainerHandle container_handle,
    size_t* num_constants);

// 获取常量的名称。
// idx 是常量的索引，应小于 AOTInductorModelContainerGetNumConstants 返回的常量数量。
AOTIRuntimeError AOTInductorModelContainerGetConstantName(
    AOTInductorModelContainerHandle container_handle,
    size_t idx,
    const char** name);
// Retrieves a constant's original fully qualified name (FQN) from the model container.
// idx is the index of the constant in the container.
// Ensure idx < num_constants obtained from AOTInductorModelContainerGetNumConstants.
AOTIRuntimeError AOTInductorModelContainerGetConstantOriginalFQN(
    AOTInductorModelContainerHandle container_handle,
    size_t idx,
    const char** original_fqn);

// Retrieves whether a constant is from folded (precomputed).
// idx is the index of the constant in the container.
// Ensure idx < num_constants obtained from AOTInductorModelContainerGetNumConstants.
AOTIRuntimeError AOTInductorModelContainerGetConstantFromFolded(
    AOTInductorModelContainerHandle container_handle,
    size_t idx,
    bool* from_folded);

// Retrieves a constant's data type (dtype).
// idx is the index of the constant in the container.
// Ensure idx < num_constants obtained from AOTInductorModelContainerGetNumConstants.
AOTIRuntimeError AOTInductorModelContainerGetConstantDtype(
    AOTInductorModelContainerHandle container_handle,
    size_t idx,
    int32_t* dtype);

// Updates the constant buffer in the model container with the provided ConstantMap.
// use_inactive should be set to true if updating the inactive buffer is desired.
// validate_full_update checks if all constants are included in the ConstantMap.
AOTIRuntimeError AOTInductorModelContainerUpdateConstantBuffer(
    AOTInductorModelContainerHandle container_handle,
    AOTInductorConstantMapHandle constant_map_handle,
    bool use_inactive,
    bool validate_full_update);

// Updates the inactive constant buffer in the model container with the provided ConstantMap.
AOTIRuntimeError AOTInductorModelContainerUpdateInactiveConstantBuffer(
    AOTInductorModelContainerHandle container_handle,
    AOTInductorConstantMapHandle constant_map_handle);

// Performs constant folding on the constant buffer of the model container.
AOTIRuntimeError AOTInductorModelContainerRunConstantFolding(
    AOTInductorModelContainerHandle container_handle,
    bool use_inactive,
    AOTInductorStreamHandle stream_handle,
    AOTIProxyExecutorHandle proxy_executor_handle);

// Swaps the active constant buffer being used to the inactive one in the model container.
AOTIRuntimeError AOTInductorModelContainerSwapConstantBuffer(
    AOTInductorModelContainerHandle container_handle);

// Retrieves the number of input tensors for the model.
AOTIRuntimeError AOTInductorModelContainerGetNumInputs(
    AOTInductorModelContainerHandle container_handle,
    size_t* ret_num_inputs);

// Retrieves the name of the input tensor at the given index.
AOTIRuntimeError AOTInductorModelContainerGetInputName(
    AOTInductorModelContainerHandle container_handle,
    size_t input_idx,
    const char** ret_input_names);

// Retrieves the number of output tensors for the model.
AOTIRuntimeError AOTInductorModelContainerGetNumOutputs(
    AOTInductorModelContainerHandle container_handle,
    size_t* ret_num_outputs);

// Retrieves the name of the output tensor at the given index.
AOTIRuntimeError AOTInductorModelContainerGetOutputName(
    AOTInductorModelContainerHandle container_handle,
    size_t output_idx,
    const char** ret_output_names);
// 创建 AOTInductorModel 实例。这是编译模型的轻量级封装，不处理并发、排队、设备管理等。如果需要裸金属性能并且愿意自行处理管理方面的问题，请使用此函数。
//
// constant_map_handle 是一个不透明类型，用于满足 C ABI。它应该是一个 std::unordered_map<std::string, at::Tensor*>*。
AOTIRuntimeError AOTInductorModelCreate(
    AOTInductorModelHandle* model_handle,
    AOTInductorConstantMapHandle constant_map_handle);

// 运行 AOTInductorModel（参见 AOTInductorModelCreate 了解何时应该使用此函数而不是 AOTInductorModelContainerRun）。
AOTIRuntimeError AOTInductorModelRun(
    AOTInductorModelHandle model_handle,
    AtenTensorHandle* input_handles,
    AtenTensorHandle* output_handles);

// 替换 AOTInductorModel 的常量映射。注意它不处理并发，因此确保在并发运行 AOTInductorModelRun 时正确处理顺序。
AOTIRuntimeError AOTInductorModelUpdateConstantsMap(
    AOTInductorModelHandle model_handle,
    AOTInductorConstantMapHandle constant_map_handle);

// 删除由 AOTInductorModelCreate 创建的 AOTInductorModel。
AOTIRuntimeError AOTInductorModelDelete(AOTInductorModelHandle model_handle);

// 获取 AOTInductorModel 的输出数量。
AOTIRuntimeError AOTInductorModelGetNumOutputs(
    AOTInductorModelHandle model_handle,
    size_t* ret_num_outputs);

// 获取 AOTInductorModelContainer 的调用规范。
AOTIRuntimeError AOTInductorModelContainerGetCallSpec(
    AOTInductorModelContainerHandle container_handle,
    const char** in_spec,
    const char** out_spec);

} // extern "C"
```