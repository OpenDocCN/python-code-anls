# `.\pytorch\torch\csrc\api\include\torch\data\transforms\tensor.h`

```py
#pragma once

#include <torch/data/example.h>
#include <torch/data/transforms/base.h>
#include <torch/types.h>

#include <functional>
#include <utility>

namespace torch {
namespace data {
namespace transforms {

/// 专门为典型的 `Example<Tensor, Tensor>` 组合而设计的 `Transform`。它暴露了一个 `operator()` 接口挂钩（供子类使用），
/// 并在输入的 `Example` 对象上调用此函数。
template <typename Target = Tensor>
class TensorTransform
    : public Transform<Example<Tensor, Target>, Example<Tensor, Target>> {
 public:
  using E = Example<Tensor, Target>;
  using typename Transform<E, E>::InputType;
  using typename Transform<E, E>::OutputType;

  /// 将单个输入张量转换为输出张量。
  virtual Tensor operator()(Tensor input) = 0;

  /// 实现 `Transform::apply`，调用 `operator()`。
  OutputType apply(InputType input) override {
    input.data = (*this)(std::move(input.data));
    return input;
  }
};

/// 专门为典型的 `Example<Tensor, Tensor>` 输入类型而设计的 `Lambda`。
template <typename Target = Tensor>
class TensorLambda : public TensorTransform<Target> {
 public:
  using FunctionType = std::function<Tensor(Tensor)>;

  /// 从给定的 `function` 创建一个 `TensorLambda`。
  explicit TensorLambda(FunctionType function)
      : function_(std::move(function)) {}

  /// 将用户提供的函数应用于输入张量。
  Tensor operator()(Tensor input) override {
    return function_(std::move(input));
  }

 private:
  FunctionType function_;
};

/// 通过减去提供的均值并除以给定的标准差来标准化输入张量。
template <typename Target = Tensor>
struct Normalize : public TensorTransform<Target> {
  /// 构造一个 `Normalize` 变换。均值和标准差可以是可以广播到输入张量的任何值（如单个标量）。
  Normalize(ArrayRef<double> mean, ArrayRef<double> stddev)
      : mean(torch::tensor(mean, torch::kFloat32)
                 .unsqueeze(/*dim=*/1)
                 .unsqueeze(/*dim=*/2)),
        stddev(torch::tensor(stddev, torch::kFloat32)
                   .unsqueeze(/*dim=*/1)
                   .unsqueeze(/*dim=*/2)) {}

  torch::Tensor operator()(Tensor input) override {
    return input.sub(mean).div(stddev);
  }

  torch::Tensor mean, stddev;
};
} // namespace transforms
} // namespace data
} // namespace torch


注释已经添加完毕，代码块中包含了对每行代码的解释性注释，说明了其作用和功能。
```