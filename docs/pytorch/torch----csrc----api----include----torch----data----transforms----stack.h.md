# `.\pytorch\torch\csrc\api\include\torch\data\transforms\stack.h`

```py
#pragma once
// 预处理指令，确保本文件只被编译一次

#include <torch/data/example.h>
// 引入 Torch 数据处理模块的示例定义

#include <torch/data/transforms/collate.h>
// 引入 Torch 数据处理模块的数据收集转换功能

#include <torch/types.h>
// 引入 Torch 的数据类型定义

#include <utility>
// 引入 C++ 标准库的 utility 头文件，用于支持 std::move 等功能

#include <vector>
// 引入 C++ 标准库的 vector 头文件，用于支持 std::vector 容器

namespace torch {
namespace data {
namespace transforms {

template <typename T = Example<>>
struct Stack;
// 定义一个模板结构体 Stack，用于数据堆叠操作

/// A `Collation` for `Example<Tensor, Tensor>` types that stacks all data
/// tensors into one tensor, and all target (label) tensors into one tensor.
// 用于 Example<Tensor, Tensor> 类型的数据收集，将所有数据张量堆叠成一个张量，
// 所有目标（标签）张量也堆叠成一个张量

template <>
struct Stack<Example<>> : public Collation<Example<>> {
  Example<> apply_batch(std::vector<Example<>> examples) override {
    std::vector<torch::Tensor> data, targets;
    // 创建两个空的 Torch 张量向量 data 和 targets
    data.reserve(examples.size());
    // 预留足够的空间以容纳 examples 容器的元素数量
    targets.reserve(examples.size());
    // 预留足够的空间以容纳 examples 容器的元素数量
    for (auto& example : examples) {
      // 遍历 examples 容器中的每个元素
      data.push_back(std::move(example.data));
      // 将 example.data 移动到 data 向量中
      targets.push_back(std::move(example.target));
      // 将 example.target 移动到 targets 向量中
    }
    return {torch::stack(data), torch::stack(targets)};
    // 返回堆叠后的数据和目标张量的 Example 对象
  }
};

/// A `Collation` for `Example<Tensor, NoTarget>` types that stacks all data
/// tensors into one tensor.
// 用于 Example<Tensor, NoTarget> 类型的数据收集，将所有数据张量堆叠成一个张量

template <>
struct Stack<TensorExample>
    : public Collation<Example<Tensor, example::NoTarget>> {
  TensorExample apply_batch(std::vector<TensorExample> examples) override {
    std::vector<torch::Tensor> data;
    // 创建一个空的 Torch 张量向量 data
    data.reserve(examples.size());
    // 预留足够的空间以容纳 examples 容器的元素数量
    for (auto& example : examples) {
      // 遍历 examples 容器中的每个元素
      data.push_back(std::move(example.data));
      // 将 example.data 移动到 data 向量中
    }
    return torch::stack(data);
    // 返回堆叠后的数据张量的 TensorExample 对象
  }
};

} // namespace transforms
} // namespace data
} // namespace torch
// 结束 torch 命名空间和头文件
```