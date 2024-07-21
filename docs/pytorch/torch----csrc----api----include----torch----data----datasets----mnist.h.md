# `.\pytorch\torch\csrc\api\include\torch\data\datasets\mnist.h`

```py
#pragma once
// 预处理指令，确保头文件只被编译一次

#include <torch/data/datasets/base.h>
// 包含 Torch 数据集基类的头文件

#include <torch/data/example.h>
// 包含 Torch 示例类的头文件

#include <torch/types.h>
// 包含 Torch 类型定义的头文件

#include <torch/csrc/Export.h>
// 包含 Torch 导出符号定义的头文件

#include <cstddef>
// 包含标准库的头文件，定义了 size_t 类型

#include <string>
// 包含标准库的头文件，定义了 string 类型

namespace torch {
namespace data {
namespace datasets {
/// The MNIST dataset.
/// MNIST 数据集的实现
class TORCH_API MNIST : public Dataset<MNIST> {
  // 继承自 Dataset<MNIST> 的 MNIST 类定义

 public:
  /// The mode in which the dataset is loaded.
  /// 数据集加载的模式枚举
  enum class Mode { kTrain, kTest };

  /// Loads the MNIST dataset from the `root` path.
  /// 从 `root` 路径加载 MNIST 数据集
  ///
  /// The supplied `root` path should contain the *content* of the unzipped
  /// MNIST dataset, available from http://yann.lecun.com/exdb/mnist.
  explicit MNIST(const std::string& root, Mode mode = Mode::kTrain);
  // 构造函数，从指定路径 `root` 加载 MNIST 数据集

  /// Returns the `Example` at the given `index`.
  /// 返回给定索引处的 `Example` 示例
  Example<> get(size_t index) override;

  /// Returns the size of the dataset.
  /// 返回数据集的大小
  optional<size_t> size() const override;

  /// Returns true if this is the training subset of MNIST.
  /// 如果这是 MNIST 的训练子集，则返回 true
  // NOLINTNEXTLINE(bugprone-exception-escape)
  bool is_train() const noexcept;

  /// Returns all images stacked into a single tensor.
  /// 返回所有图像堆叠成的单个张量
  const Tensor& images() const;

  /// Returns all targets stacked into a single tensor.
  /// 返回所有目标值堆叠成的单个张量
  const Tensor& targets() const;

 private:
  Tensor images_, targets_;
  // 私有成员变量 images_ 和 targets_，存储图像和目标值的张量
};
} // namespace datasets
} // namespace data
} // namespace torch
```