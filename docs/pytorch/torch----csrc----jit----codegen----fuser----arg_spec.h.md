# `.\pytorch\torch\csrc\jit\codegen\fuser\arg_spec.h`

```py
#pragma once
#include <ATen/ATen.h>
#include <ATen/core/functional.h> // fmap
#include <c10/util/hash.h>
#include <torch/csrc/Export.h>
#include <torch/csrc/jit/codegen/fuser/tensor_desc.h>

#include <cstdint>
#include <vector>

namespace torch {
namespace jit {
namespace fuser {

// 描述一个核函数的运行时参数
// ArgSpec 也被用作查找实例化的核函数的键，因此必须是可哈希的
// 注意：包含设备信息是因为核函数是按设备编译的
struct TORCH_API ArgSpec {
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  // 根据输入张量列表和设备创建 ArgSpec 对象
  ArgSpec(at::TensorList inputs, const int _device)
      : descs_{c10::fmap<TensorDesc>(inputs)},  // 使用 fmap 将输入张量列表映射为 TensorDesc 向量
        hash_code_{c10::get_hash(_device, inputs.size(), descs_)},  // 计算哈希码
        device_{_device} {}  // 初始化设备信息

  // 哈希函数，用于计算对象的哈希值
  static size_t hash(const ArgSpec& spec) {
    return spec.hash_code_;
  }

  // 比较操作符，用于检查两个 ArgSpec 对象是否相等
  bool operator==(const ArgSpec& other) const {
    return (descs_ == other.descs_ && device_ == other.device_);
  }

  // 不相等比较操作符，用于检查两个 ArgSpec 对象是否不相等
  bool operator!=(const ArgSpec& spec) const {
    return !(*this == spec);
  }

  // 返回对象的哈希码
  size_t hashCode() const {
    return hash_code_;
  }

  // 返回描述符向量
  const std::vector<TensorDesc>& descs() const {
    return descs_;
  }

  // 返回设备信息
  int device() const {
    return device_;
  }

 private:
  std::vector<TensorDesc> descs_;  // 描述符向量
  size_t hash_code_;  // 哈希码
  int device_;  // 设备信息
};

} // namespace fuser
} // namespace jit
} // namespace torch
```