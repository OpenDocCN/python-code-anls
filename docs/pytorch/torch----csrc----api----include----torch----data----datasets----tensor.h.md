# `.\pytorch\torch\csrc\api\include\torch\data\datasets\tensor.h`

```py
#pragma once

/// `#pragma once` 是一种预处理指令，用于确保头文件只被编译一次，避免多重包含问题。


#include <torch/data/datasets/base.h>
#include <torch/data/example.h>
#include <torch/types.h>

#include <cstddef>
#include <vector>

/// 包含了几个头文件，这些头文件定义了在 Torch C++ 库中使用的数据集和张量类型的基本功能。


namespace torch {
namespace data {
namespace datasets {

/// 命名空间 `torch::data::datasets`，用于封装和组织与数据集相关的类和函数。


/// A dataset of tensors.
/// Stores a single tensor internally, which is then indexed inside `get()`.
struct TensorDataset : public Dataset<TensorDataset, TensorExample> {

/// `TensorDataset` 结构体表示一个张量数据集，继承自 `Dataset<TensorDataset, TensorExample>`，用于处理张量数据集。


/// Creates a `TensorDataset` from a vector of tensors.
explicit TensorDataset(const std::vector<Tensor>& tensors)
    : TensorDataset(torch::stack(tensors)) {}

/// 构造函数，从张量向量创建 `TensorDataset` 对象，使用 `torch::stack(tensors)` 将多个张量堆叠成一个张量。


explicit TensorDataset(torch::Tensor tensor) : tensor(std::move(tensor)) {}

/// 另一个构造函数，直接使用给定的张量初始化 `TensorDataset` 对象的 `tensor` 成员变量。


/// Returns a single `TensorExample`.
TensorExample get(size_t index) override {
  return tensor[index];
}

/// `get` 方法重载自基类，用于根据索引返回一个 `TensorExample`，实际返回的是 `tensor` 成员变量中的第 `index` 个张量。


/// Returns the number of tensors in the dataset.
optional<size_t> size() const override {
  return tensor.size(0);
}

/// `size` 方法重载自基类，返回数据集中张量的数量，使用 `tensor.size(0)` 获取第一个维度的大小。


Tensor tensor;

/// `Tensor` 类型的成员变量 `tensor`，用于存储数据集中的张量数据。


}; // struct TensorDataset

} // namespace datasets
} // namespace data
} // namespace torch
```