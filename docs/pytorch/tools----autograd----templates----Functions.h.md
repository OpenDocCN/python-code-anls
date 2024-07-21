# `.\pytorch\tools\autograd\templates\Functions.h`

```
#pragma once

// ${generated_comment}  // 插入由代码生成工具生成的注释，通常包含有关生成代码的信息

#include <ATen/ATen.h>  // 包含 PyTorch 核心头文件
#include <ATen/core/functional.h>  // 包含 PyTorch 核心功能头文件
#include <ATen/TensorGeometry.h>  // 包含 PyTorch 张量几何特性头文件

#include "torch/csrc/autograd/function.h"  // 包含自动微分功能定义头文件
#include "torch/csrc/autograd/variable.h"  // 包含自动微分变量定义头文件
#include "torch/csrc/autograd/saved_variable.h"  // 包含保存的自动微分变量定义头文件
#include <torch/csrc/Export.h>  // 包含导出定义头文件

#include <c10/core/SymIntArrayRef.h>  // 包含 C10 核心符号整数数组引用头文件

namespace torch { namespace autograd { namespace generated {

using at::Scalar;  // 使用 PyTorch 标量类型
using at::Tensor;  // 使用 PyTorch 张量类型
using at::IntArrayRef;  // 使用 PyTorch 整数数组引用类型
using at::ArrayRef;  // 使用 PyTorch 数组引用类型
using at::Type;  // 使用 PyTorch 类型类型
using at::TensorGeometry;  // 使用 PyTorch 张量几何特性类型
using at::ScalarType;  // 使用 PyTorch 标量类型类型
using std::optional;  // 使用标准库的可选类型
using c10::fmap;  // 使用 C10 的映射操作

inline std::vector<Tensor> unpack_list(at::ArrayRef<SavedVariable> xs, std::shared_ptr<Node> saved_for = nullptr) {
  // 解包 SavedVariable 的数组引用 xs，返回一个 Tensor 的向量
  // 注意：必须在 lambda 表达式中显式进行类型转换，否则模板推导会导致返回 Variable 而不是可转换的 Tensor
  return fmap(xs, [&saved_for](const SavedVariable& x) {
    // 使用 saved_for 参数对 SavedVariable 进行解包，并转换为 Tensor 类型
    return static_cast<Tensor>(x.unpack(saved_for));
  });
}

inline c10::List<std::optional<Tensor>> unpack_opt_list(at::ArrayRef<SavedVariable> xs, std::shared_ptr<Node> saved_for = nullptr) {
  torch::List<std::optional<Tensor>> result;  // 创建一个 Tensor 可选值列表
  result.reserve(xs.size());  // 预留 xs 大小的空间
  for (const SavedVariable& v : xs) {
    auto var = v.unpack(saved_for);  // 解包 SavedVariable 并存储到 var 中
    // 如果 var 有定义，则将其作为 std::optional<Tensor> 添加到 result 中；否则添加 c10::nullopt
    result.push_back(var.defined() ? std::optional<Tensor>(var) : c10::nullopt);
  }
  return result;  // 返回结果列表
}

using torch::autograd::TypeAndSize;  // 使用自动微分类型和大小

${autograd_function_declarations}  // 插入自动生成的自动微分函数声明

}}} // namespace torch::autograd::generated  // 声明结束
```