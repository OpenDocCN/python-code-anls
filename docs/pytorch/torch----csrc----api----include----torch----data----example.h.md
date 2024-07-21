# `.\pytorch\torch\csrc\api\include\torch\data\example.h`

```
#pragma once

#include <torch/types.h>  // 包含 Torch 的类型定义

namespace torch {
namespace data {

/// An `Example` from a dataset.
///
/// A dataset consists of data and an associated target (label).
template <typename Data = at::Tensor, typename Target = at::Tensor>
struct Example {
  using DataType = Data;    // 使用别名 DataType 表示 Data 类型
  using TargetType = Target;  // 使用别名 TargetType 表示 Target 类型

  Example() = default;  // 默认构造函数
  Example(Data data, Target target)
      : data(std::move(data)), target(std::move(target)) {}  // 构造函数，接受 data 和 target

  Data data;    // 数据成员，存储数据
  Target target;    // 数据成员，存储目标
};

namespace example {
using NoTarget = void;    // example 命名空间中定义 NoTarget 别名为 void
} // namespace example

/// A specialization for `Example` that does not have a target.
///
/// This class exists so that code can be written for a templated `Example`
/// type, and work both for labeled and unlabeled datasets.
template <typename Data>
struct Example<Data, example::NoTarget> {
  using DataType = Data;    // 使用别名 DataType 表示 Data 类型
  using TargetType = example::NoTarget;    // 使用别名 TargetType 表示 NoTarget 类型

  Example() = default;    // 默认构造函数
  /* implicit */ Example(Data data) : data(std::move(data)) {}    // 隐式转换构造函数，接受 data

  // When a DataLoader returns an Example like this, that example should be
  // implicitly convertible to the underlying data type.

  operator Data&() {    // 转换运算符，返回可修改的 data 引用
    return data;
  }
  operator const Data&() const {    // 转换运算符，返回常量的 data 引用
    return data;
  }

  Data data;    // 数据成员，存储数据
};

using TensorExample = Example<at::Tensor, example::NoTarget>;    // 使用 Example 模板实例化 TensorExample
} // namespace data
} // namespace torch
```