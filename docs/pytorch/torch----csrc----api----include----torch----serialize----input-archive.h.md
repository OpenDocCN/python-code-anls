# `.\pytorch\torch\csrc\api\include\torch\serialize\input-archive.h`

```py
#pragma once

#include <c10/core/Device.h>
#include <c10/util/Optional.h>
#include <torch/csrc/Export.h>
#include <torch/csrc/jit/api/module.h>
#include <torch/types.h>

#include <iosfwd>
#include <memory>
#include <string>
#include <utility>

namespace at {
class Tensor;
} // namespace at

namespace torch {

// 使用 at 命名空间中的 Tensor 类型
using at::Tensor;

namespace jit {

// 使用 jit 命名空间中的 Module 结构体
struct Module;

} // namespace jit
} // namespace torch

namespace torch {
namespace serialize {

/// A recursive representation of tensors that can be deserialized from a file
/// or stream. In most cases, users should not have to interact with this class,
/// and should instead use `torch::load`.
class InputArchive {
 public:

  // 构造函数，初始化一个 Module 对象
  InputArchive(std::string source_name, c10::optional<c10::Device> device = c10::nullopt);
  
  // 使用变参模板，读取多个参数类型的数据
  template <typename... Ts>
  void read(std::forward<Ts>(ts)...);

 private:

  // 存储反序列化后的 Module 对象
  jit::Module module_;

  // 序列化结构的层级前缀
  std::string hierarchy_prefix_;
};

} // namespace serialize
} // namespace torch
```