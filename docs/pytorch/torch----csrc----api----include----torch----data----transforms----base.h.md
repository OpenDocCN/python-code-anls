# `.\pytorch\torch\csrc\api\include\torch\data\transforms\base.h`

```py
#pragma once
// 预处理指令，确保本文件只被编译一次

#include <torch/types.h>
// 包含 Torch 库的类型定义头文件

#include <utility>
// 包含 C++ 标准库中的 utility 头文件，用于 std::move 等操作

#include <vector>
// 包含 C++ 标准库中的 vector 头文件，用于 std::vector 容器

namespace torch {
namespace data {
namespace transforms {

/// A transformation of a batch to a new batch.
/// 批处理转换类，将一个批次转换为另一个批次。
template <typename InputBatch, typename OutputBatch>
class BatchTransform {
 public:
  using InputBatchType = InputBatch;
  using OutputBatchType = OutputBatch;

  virtual ~BatchTransform() = default;
  // 虚析构函数，确保派生类可以安全地销毁

  /// Applies the transformation to the given `input_batch`.
  /// 将转换应用于给定的 `input_batch`。
  virtual OutputBatch apply_batch(InputBatch input_batch) = 0;
  // 纯虚函数，子类需实现将转换应用于批次的具体逻辑
};

/// A transformation of individual input examples to individual output examples.
/// 单个输入例子到单个输出例子的转换。
///
/// Just like a `Dataset` is a `BatchDataset`, a `Transform` is a
/// `BatchTransform` that can operate on the level of individual examples rather
/// than entire batches. The batch-level transform is implemented (by default)
/// in terms of the example-level transform, though this can be customized.
/// 类似于 `Dataset` 是 `BatchDataset`，`Transform` 是一个可以在个体例子级别操作的 `BatchTransform`。
/// 批次级别的转换默认情况下是基于个体例子级别的转换实现的，尽管这可以进行定制。
template <typename Input, typename Output>
class Transform
    : public BatchTransform<std::vector<Input>, std::vector<Output>> {
 public:
  using InputType = Input;
  using OutputType = Output;

  /// Applies the transformation to the given `input`.
  /// 将转换应用于给定的 `input`。
  virtual OutputType apply(InputType input) = 0;
  // 纯虚函数，子类需实现将转换应用于单个输入例子的具体逻辑

  /// Applies the `transformation` over the entire `input_batch`.
  /// 对整个 `input_batch` 应用转换。
  std::vector<Output> apply_batch(std::vector<Input> input_batch) override {
    // 初始化输出批次向量
    std::vector<Output> output_batch;
    // 预留足够的空间以容纳输入批次的大小
    output_batch.reserve(input_batch.size());
    // 遍历输入批次中的每个输入例子
    for (auto&& input : input_batch) {
      // 对每个输入例子应用转换，并将结果添加到输出批次中
      output_batch.push_back(apply(std::move(input)));
    }
    // 返回转换后的输出批次
    return output_batch;
  }
};
} // namespace transforms
} // namespace data
} // namespace torch
```