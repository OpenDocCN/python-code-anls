# `.\pytorch\torch\csrc\api\include\torch\data\transforms\lambda.h`

```py
#pragma once

#include <torch/data/transforms/base.h>  // 引入基础头文件

#include <functional>  // 引入函数对象相关的头文件
#include <utility>     // 引入 std::move 使用的头文件
#include <vector>      // 引入向量容器的头文件

namespace torch {
namespace data {
namespace transforms {

/// 一个 `BatchTransform` 类，用于将用户提供的函数对象应用于批处理数据。
template <typename Input, typename Output = Input>
class BatchLambda : public BatchTransform<Input, Output> {
 public:
  using typename BatchTransform<Input, Output>::InputBatchType;   // 批处理输入类型别名
  using typename BatchTransform<Input, Output>::OutputBatchType;  // 批处理输出类型别名
  using FunctionType = std::function<OutputBatchType(InputBatchType)>;  // 函数对象类型定义

  /// 从给定的 `function` 对象构造 `BatchLambda`。
  explicit BatchLambda(FunctionType function)
      : function_(std::move(function)) {}  // 构造函数，接收一个函数对象并移动到成员变量

  /// 将用户提供的函数对象应用于 `input_batch` 批处理数据。
  OutputBatchType apply_batch(InputBatchType input_batch) override {
    return function_(std::move(input_batch));  // 调用函数对象处理输入数据并返回结果
  }

 private:
  FunctionType function_;  // 存储用户提供的函数对象
};

// 一个 `Transform` 类，用于将用户提供的函数对象应用于单个样本。
template <typename Input, typename Output = Input>
class Lambda : public Transform<Input, Output> {
 public:
  using typename Transform<Input, Output>::InputType;   // 输入类型别名
  using typename Transform<Input, Output>::OutputType;  // 输出类型别名
  using FunctionType = std::function<Output(Input)>;    // 函数对象类型定义

  /// 从给定的 `function` 对象构造 `Lambda`。
  explicit Lambda(FunctionType function) : function_(std::move(function)) {}  // 构造函数，接收一个函数对象并移动到成员变量

  /// 将用户提供的函数对象应用于 `input` 单个样本数据。
  OutputType apply(InputType input) override {
    return function_(std::move(input));  // 调用函数对象处理输入数据并返回结果
  }

 private:
  FunctionType function_;  // 存储用户提供的函数对象
};

} // namespace transforms
} // namespace data
} // namespace torch
```