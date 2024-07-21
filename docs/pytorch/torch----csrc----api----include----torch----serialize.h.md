# `.\pytorch\torch\csrc\api\include\torch\serialize.h`

```py
#pragma once
// 预处理指令：确保本头文件只被编译一次

#include <c10/util/irange.h>
// 包含 c10 库中的 irange.h 头文件，提供了用于范围迭代的实用工具

#include <torch/csrc/Export.h>
// 包含 Torch 的导出宏定义头文件

#include <torch/serialize/archive.h>
// 包含 Torch 序列化档案相关的头文件

#include <torch/serialize/tensor.h>
// 包含 Torch 序列化张量相关的头文件

#include <utility>
// 包含 C++ 实用工具头文件，提供了一些泛型编程相关的实用工具

namespace torch {
// 命名空间 torch：包含了 Torch 框架的所有内容

/// Serializes the given `value`.
/// There must be an overload of `operator<<` between `serialize::OutputArchive`
/// and `Value` for this method to be well-formed. Currently, such an overload
/// is provided for (subclasses of):
///
/// - `torch::nn::Module`,
/// - `torch::optim::Optimizer`
/// - `torch::Tensor`
///
/// To perform the serialization, a `serialize::OutputArchive` is constructed,
/// and all arguments after the `value` are forwarded to its `save_to` method.
/// For example, you can pass a filename, or an `ostream`.
///
/// \rst
/// .. code-block:: cpp
///
///   torch::nn::Linear model(3, 4);
///   torch::save(model, "model.pt");
///
///   torch::optim::SGD sgd(model->parameters(), 0.9); // 0.9 is learning rate
///   std::ostringstream stream;
///   // Note that the same stream cannot be used in multiple torch::save(...)
///   // invocations, otherwise the header will be corrupted.
///   torch::save(sgd, stream);
///
///   auto tensor = torch::ones({3, 4});
///   torch::save(tensor, "my_tensor.pt");
/// \endrst
template <typename Value, typename... SaveToArgs>
void save(const Value& value, SaveToArgs&&... args) {
  // 构造一个 OutputArchive 对象，使用共享的 jit::CompilationUnit
  serialize::OutputArchive archive(std::make_shared<jit::CompilationUnit>());
  // 将 value 序列化到 archive 中
  archive << value;
  // 将后续参数转发给 archive 的 save_to 方法
  archive.save_to(std::forward<SaveToArgs>(args)...);
}

/// Serializes the given `tensor_vec` of type `std::vector<torch::Tensor>`.
///
/// To perform the serialization, a `serialize::OutputArchive` is constructed,
/// and all arguments after the `tensor_vec` are forwarded to its `save_to`
/// method. For example, you can pass a filename, or an `ostream`.
///
/// \rst
/// .. code-block:: cpp
///
///   std::vector<torch::Tensor> tensor_vec = { torch::randn({1, 2}),
///   torch::randn({3, 4}) }; torch::save(tensor_vec, "my_tensor_vec.pt");
///
///   std::vector<torch::Tensor> tensor_vec = { torch::randn({5, 6}),
///   torch::randn({7, 8}) }; std::ostringstream stream;
///   // Note that the same stream cannot be used in multiple torch::save(...)
///   // invocations, otherwise the header will be corrupted.
///   torch::save(tensor_vec, stream);
/// \endrst
template <typename... SaveToArgs>
void save(const std::vector<torch::Tensor>& tensor_vec, SaveToArgs&&... args) {
  // 构造一个 OutputArchive 对象，使用共享的 jit::CompilationUnit
  serialize::OutputArchive archive(std::make_shared<jit::CompilationUnit>());
  // 遍历 tensor_vec 中的每一个 torch::Tensor
  for (const auto i : c10::irange(tensor_vec.size())) {
    auto& value = tensor_vec[i];
    // 将当前 tensor 序列化到 archive 中，键为索引 i 转换为字符串
    archive.write(std::to_string(i), value);
  }
  // 将后续参数转发给 archive 的 save_to 方法
  archive.save_to(std::forward<SaveToArgs>(args)...);
}

TORCH_API std::vector<char> pickle_save(const torch::IValue& ivalue);
// 使用 Torch API 对象 ivalue 进行 pickle 序列化，并返回结果的字符向量

TORCH_API torch::IValue pickle_load(const std::vector<char>& data);
// 使用 Torch API 对 data 字符向量进行 pickle 反序列化，并返回结果的 Torch IValue 对象

/// Deserializes the given `value`.
/// There must be an overload of `operator>>` between `serialize::InputArchive`
/// Deserializes a value of type `Value` using the `serialize::InputArchive`.
///
/// To perform deserialization, an `serialize::InputArchive` is created.
/// All arguments following `value` are forwarded to `archive.load_from()` for initialization,
/// allowing flexibility such as passing filenames or input streams.
///
/// \rst
/// Example usage in C++:
///
/// .. code-block:: cpp
///
///   torch::nn::Linear model(3, 4);
///   torch::load(model, "model.pt");
///
///   torch::optim::SGD sgd(model->parameters(), 0.9); // 0.9 is learning rate
///   std::istringstream stream("...");
///   torch::load(sgd, stream);
///
///   auto tensor = torch::ones({3, 4});
///   torch::load(tensor, "my_tensor.pt");
/// \endrst
template <typename Value, typename... LoadFromArgs>
void load(Value& value, LoadFromArgs&&... args) {
  serialize::InputArchive archive;
  archive.load_from(std::forward<LoadFromArgs>(args)...);
  archive >> value;
}

/// Deserializes a `std::vector` of `torch::Tensor` objects.
///
/// To perform deserialization, an `serialize::InputArchive` is created.
/// All arguments following `tensor_vec` are forwarded to `archive.load_from()` for initialization,
/// enabling usage with filenames or input streams.
///
/// \rst
/// Example usage in C++:
///
/// .. code-block:: cpp
///
///   std::vector<torch::Tensor> tensor_vec;
///   torch::load(tensor_vec, "my_tensor_vec.pt");
///
///   std::vector<torch::Tensor> tensor_vec;
///   std::istringstream stream("...");
///   torch::load(tensor_vec, stream);
/// \endrst
template <typename... LoadFromArgs>
void load(std::vector<torch::Tensor>& tensor_vec, LoadFromArgs&&... args) {
  serialize::InputArchive archive;
  archive.load_from(std::forward<LoadFromArgs>(args)...);

  // NOTE: The number of elements in the serialized `std::vector<torch::Tensor>`
  // is not known ahead of time, so we need a while-loop to increment the index,
  // and use `archive.try_read(...)` to check whether we have reached the end of
  // the serialized `std::vector<torch::Tensor>`.
  size_t index = 0;
  torch::Tensor value;
  while (archive.try_read(std::to_string(index), value)) {
    tensor_vec.push_back(std::move(value));
    value = torch::Tensor();  // Reset `value` after moving it to `tensor_vec`
    index++;
  }
}
```