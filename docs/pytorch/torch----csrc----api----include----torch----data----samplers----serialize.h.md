# `.\pytorch\torch\csrc\api\include\torch\data\samplers\serialize.h`

```
#pragma once

#include <torch/data/samplers/base.h>  // 包含 Torch 库中的数据采样器基类头文件
#include <torch/serialize/archive.h>   // 包含 Torch 序列化存档相关头文件

namespace torch {
namespace data {
namespace samplers {
/// 将 `Sampler` 序列化到 `OutputArchive` 中。
template <typename BatchRequest>
serialize::OutputArchive& operator<<(
    serialize::OutputArchive& archive,
    const Sampler<BatchRequest>& sampler) {
  sampler.save(archive);  // 调用 Sampler 的 save 方法将其序列化保存到 archive 中
  return archive;  // 返回序列化后的 OutputArchive
}

/// 从 `InputArchive` 中反序列化一个 `Sampler`。
template <typename BatchRequest>
serialize::InputArchive& operator>>(
    serialize::InputArchive& archive,
    Sampler<BatchRequest>& sampler) {
  sampler.load(archive);  // 调用 Sampler 的 load 方法从 archive 中反序列化并加载到 sampler 中
  return archive;  // 返回反序列化后的 InputArchive
}
} // namespace samplers
} // namespace data
} // namespace torch
```