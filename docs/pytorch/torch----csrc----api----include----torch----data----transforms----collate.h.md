# `.\pytorch\torch\csrc\api\include\torch\data\transforms\collate.h`

```py
#pragma once

#include <torch/data/example.h>
#include <torch/data/transforms/lambda.h>

#include <vector>

namespace torch {
namespace data {
namespace transforms {

/// `Collation` 是一个转换操作，将一个批次（batch）缩减为单个值。
/// 其结果是一个 `BatchDataset`，其 `BatchType` 为单个值的类型。
template <typename T, typename BatchType = std::vector<T>>
using Collation = BatchTransform<BatchType, T>;

/// `Collate` 允许传递一个自定义函数来将一个批次（batch）缩减/合并为单个值。
/// 这实际上是 `Collation` 的lambda版本，您可以子类化并重写 `operator()` 来达到相同的效果。
///
/// \rst
/// .. code-block:: cpp
///   using namespace torch::data;
///
///   auto dataset = datasets::MNIST("path/to/mnist")
///     .map(transforms::Collate<Example<>>([](std::vector<Example<>> e) {
///       return std::move(e.front());
///     }));
/// \endrst
template <typename T, typename BatchType = std::vector<T>>
using Collate = BatchLambda<BatchType, T>;
} // namespace transforms
} // namespace data
} // namespace torch
```