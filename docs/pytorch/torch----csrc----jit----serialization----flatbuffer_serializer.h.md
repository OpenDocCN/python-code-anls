# `.\pytorch\torch\csrc\jit\serialization\flatbuffer_serializer.h`

```py
/**
 * @file
 * @brief Defines the public API for serializing mobile modules to flatbuffer.
 * Note that this header must not include or depend on flatbuffer-defined
 * types, to avoid leaking those details to PyTorch clients.
 */

#pragma once

#include <functional>  // 包含函数对象相关的头文件
#include <memory>      // 包含智能指针相关的头文件
#include <string>      // 包含字符串相关的头文件
#include <unordered_map>  // 包含无序映射相关的头文件
#include <vector>      // 包含向量相关的头文件

#include <ATen/core/ivalue.h>  // 包含 ATen 库中 IValue 相关的头文件
#include <c10/macros/Macros.h>  // 包含 C10 库中宏定义相关的头文件
#include <torch/csrc/jit/mobile/module.h>  // 包含 Torch 移动模块相关的头文件

/**
 * @brief Maps file names to file contents.
 */
namespace torch::jit {

/// Maps file names to file contents.
using ExtraFilesMap = std::unordered_map<std::string, std::string>;

/**
 * @brief Represents a span of data. Typically owned by a UniqueDetachedBuffer.
 */
class TORCH_API DetachedBuffer final {
 public:
  /**
   * @brief Creates a new DetachedBuffer with an optional data owner.
   *        This interface is provided to let users create objects of this type for testing.
   */
  DetachedBuffer(void* data, size_t size, void* internal_data_owner = nullptr)
      : data_(data), size_(size), data_owner_(internal_data_owner) {}

  /// Returns a pointer to the data.
  C10_NODISCARD void* data() {
    return data_;
  }
  /// Returns a pointer to the data.
  C10_NODISCARD const void* data() const {
    return data_;
  }
  /// Returns the size of the data, in bytes.
  C10_NODISCARD size_t size() const {
    return size_;
  }

  /// Wrapper type that typically owns data_owner_.
  using UniqueDetachedBuffer =
      std::unique_ptr<DetachedBuffer, std::function<void(DetachedBuffer*)>>;

 private:
  /**
   * @brief Deletes the owner, if present, and the buf itself.
   *        Note: we could have provided a movable type with a destructor that did
   *        this work, but the unique wrapper was easier in practice.
   */
  static void destroy(DetachedBuffer* buf);

  /// Provides access to destroy() for implementation and testing.
  friend struct DetachedBufferFriend;
  friend struct DetachedBufferTestingFriend;

  /// Pointer to the data. Not owned by this class.
  void* data_;
  /// The size of `data_`, in bytes.
  size_t size_;
  /// Opaque pointer to the underlying owner of `data_`. This class
  /// (DetachedBuffer) does not own the owner or the data. It will typically be
  /// owned by a UniqueDetachedBuffer that knows how to delete the owner along
  /// with this class.
  void* data_owner_;
};

/**
 * @brief Saves a mobile module to a file.
 * @param module The mobile module to save.
 * @param filename The name of the file to save to.
 * @param extra_files Optional map of extra files to include in the save.
 * @param jit_sources Optional map of JIT sources to include in the save.
 * @param jit_constants Optional vector of JIT constants to include in the save.
 */
TORCH_API void save_mobile_module(
    const mobile::Module& module,
    const std::string& filename,
    const ExtraFilesMap& extra_files = ExtraFilesMap(),
    const ExtraFilesMap& jit_sources = ExtraFilesMap(),
    const std::vector<IValue>& jit_constants = {});

/**
 * @brief Saves a mobile module to a DetachedBuffer.
 * @param module The mobile module to save.
 * @param extra_files Optional map of extra files to include in the save.
 * @param jit_sources Optional map of JIT sources to include in the save.
 * @param jit_constants Optional vector of JIT constants to include in the save.
 * @return UniqueDetachedBuffer containing the serialized module data.
 */
TORCH_API DetachedBuffer::UniqueDetachedBuffer save_mobile_module_to_bytes(
    const mobile::Module& module,
    const ExtraFilesMap& extra_files = ExtraFilesMap(),
    const ExtraFilesMap& jit_sources = ExtraFilesMap(),
    const std::vector<IValue>& jit_constants = {});

/**
 * @brief Saves a mobile module using a custom writer function.
 * @param module The mobile module to save.
 * @param writer_func The function to write data using a buffer pointer and size.
 */
TORCH_API void save_mobile_module_to_func(
    const mobile::Module& module,
    const std::function<size_t(const void*, size_t)>& writer_func);

// TODO(qihan): delete
# 调用 Torch API 中的函数，用于注册 FlatBuffer 的序列化器
TORCH_API bool register_flatbuffer_serializer();

# 定义命名空间 torch::jit，用于封装 Torch JIT 模块的相关功能
} // namespace torch::jit
```