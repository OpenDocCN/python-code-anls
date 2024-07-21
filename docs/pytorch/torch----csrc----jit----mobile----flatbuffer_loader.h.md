# `.\pytorch\torch\csrc\jit\mobile\flatbuffer_loader.h`

```
/**
 * @file
 * @brief This header defines the public API for loading flatbuffer-serialized mobile modules.
 *        It does not include flatbuffer-defined types to prevent leakage to PyTorch clients.
 */

#pragma once

#include <istream>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <ATen/core/ivalue.h>
#include <c10/core/Device.h>
#include <c10/macros/Macros.h>
#include <c10/util/Optional.h>
#include <torch/csrc/jit/mobile/module.h>

namespace torch {
namespace jit {

/**
 * @brief Alignment boundary in bytes for non-copied data pointers used in parsing and initialization functions.
 *        This ensures proper alignment of certain types/structs within the Module.
 */
constexpr size_t kFlatbufferDataAlignmentBytes = 16;

/**
 * @brief Maps file names to their corresponding file contents.
 */
using ExtraFilesMap = std::unordered_map<std::string, std::string>;

/**
 * @brief Parses a mobile::Module from raw bytes and initializes it.
 *        This function combines steps 2 and 3: deserialization and module initialization.
 *
 * @param data Pointer to the raw data buffer containing the serialized module.
 * @param size Size of the data buffer in bytes.
 * @param device Optional parameter specifying the target device for the module.
 * @param extra_files Optional map of extra files needed during module initialization.
 * @param should_copy_tensor_memory Flag indicating whether tensor memory should be copied or referenced.
 * @return Initialized mobile::Module object.
 */
TORCH_API mobile::Module parse_and_initialize_mobile_module(
    void* data,
    size_t size,
    std::optional<at::Device> device = c10::nullopt,
    ExtraFilesMap* extra_files = nullptr,
    bool should_copy_tensor_memory = false);

/**
 * @brief Parses a mobile::Module from raw bytes and initializes it.
 *        This function combines steps 2 and 3: deserialization and module initialization.
 *        The Module holds a shared_ptr to `data`, ensuring its lifetime.
 *
 * @param data Shared pointer to the raw data buffer containing the serialized module.
 * @param size Size of the data buffer in bytes.
 * @param device Optional parameter specifying the target device for the module.
 * @param extra_files Optional map of extra files needed during module initialization.
 * @return Initialized mobile::Module object.
 */
TORCH_API mobile::Module parse_and_initialize_mobile_module(
    std::shared_ptr<char> data,
    size_t size,
    std::optional<at::Device> device = c10::nullopt,
    ExtraFilesMap* extra_files = nullptr);

// Additional function declaration would go here if needed.

} // namespace jit
} // namespace torch
// 解析并初始化移动端模块，同时提取 JIT 源文件和常量。可用于构建 jit::Module。
TORCH_API mobile::Module parse_and_initialize_mobile_module_for_jit(
    void* data,
    size_t size, // data 的大小，以字节为单位
    ExtraFilesMap& jit_sources, // JIT 源文件的映射
    std::vector<IValue>& jit_constants, // JIT 常量的向量
    std::optional<at::Device> device = c10::nullopt, // 设备选项，默认为空
    ExtraFilesMap* extra_files = nullptr); // 额外文件的映射，可选

// 从文件路径加载 mobile::Module。
//
// 这个函数执行上述描述的步骤 1+2+3。
//
// 我们需要这个函数作为便利，因为 Python API 需要包装这个函数。C++ 客户端应该使用其中一个
// parse_and_initialize_mobile_module() 的版本，这样它们可以更直接地管理原始数据。
TORCH_API mobile::Module load_mobile_module_from_file(
    const std::string& filename, // 文件路径名
    std::optional<at::Device> device = c10::nullopt, // 设备选项，默认为空
    ExtraFilesMap* extra_files = nullptr); // 额外文件的映射，可选

// 从输入流中获取字节码版本号。
TORCH_API uint64_t get_bytecode_version(std::istream& in);

// 从文件名获取字节码版本号。
TORCH_API uint64_t get_bytecode_version(const std::string& filename);

// 从字节内容获取字节码版本号。
TORCH_API uint64_t get_bytecode_version_from_bytes(char* flatbuffer_content);

// 从 FlatBuffer 内容中获取移动端模块信息。
TORCH_API mobile::ModuleInfo get_module_info_from_flatbuffer(
    char* flatbuffer_content);

// 以下方法效率较低，因为它需要将流完全读取到缓冲区中
// 从带复制的流中加载 mobile::Module。
TORCH_API mobile::Module load_mobile_module_from_stream_with_copy(
    std::istream& in,
    std::optional<at::Device> device = c10::nullopt, // 设备选项，默认为空
    ExtraFilesMap* extra_files = nullptr); // 额外文件的映射，可选

// 解析不包含对象的 FlatBuffer。
TORCH_API mobile::Module parse_flatbuffer_no_object(
    std::shared_ptr<char> data,
    size_t size,
    std::optional<at::Device> device); // 设备选项，可选

// 解析并初始化移动端模块。
TORCH_API mobile::Module parse_and_initialize_mobile_module(
    void* data,
    size_t, // 数据的大小，未使用的参数
    std::optional<at::Device>, // 设备选项，可选
    ExtraFilesMap* extra_files,
    bool should_copy_tensor_memory); // 是否复制张量内存的标志

// 空操作，TODO：删除
TORCH_API bool register_flatbuffer_loader();

} // namespace jit
} // namespace torch
```