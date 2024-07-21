# `.\pytorch\aten\src\ATen\InferSize.h`

```
#pragma once

#include <ATen/DimVector.h>
#include <c10/core/ScalarType.h>
#include <c10/core/SymIntArrayRef.h>
#include <c10/util/DimVector.h>
#include <c10/util/Optional.h>
#include <sstream>
#include <vector>

namespace at {

// 推断具有大小为 -1 的维度的大小（如果存在）。还检查新形状是否与元素数量兼容。
//
// 用于处理 std::vector<int64_t> 和 DimVector 的模板，见下文
//
template <typename InputArrayRef, typename NumelType, typename ResultVec>
inline void infer_size_impl(
    InputArrayRef shape,
    NumelType numel,
    ResultVec& res) {
  NumelType newsize = 1;
  // N.B. 这是一个索引，而不是符号维度！
  auto infer_dim = std::optional<int64_t>();
  for (int64_t dim = 0, ndim = shape.size(); dim != ndim; dim++) {
    if (shape[dim] == -1) {
      if (infer_dim) {
        throw std::runtime_error("only one dimension can be inferred");
      }
      infer_dim = dim;
    } else if (shape[dim] >= 0) {
      newsize *= shape[dim];
    } else {
      AT_ERROR("invalid shape dimension ", shape[dim]);
    }
  }

  // 检查推断的维度或者大小是否与给定的元素数量兼容
  if (TORCH_GUARD_SIZE_OBLIVIOUS(sym_eq(numel, newsize)) ||
      (infer_dim && newsize > 0 && numel % newsize == 0)) {
    if (infer_dim) {
      // 在这里我们有选择维度大小的自由度；遵循 NumPy 语义并返回。
      // 然而，需要一个良好的错误消息，因为用户经常使用 `view` 来展平和展开维度，
      // 否则会因为下面的情况而感到困惑：
      //   empty_tensor.view( 0, 0)
      // 可行但
      //   empty_tensor.view(-1, 0)
      // 不行。
      TORCH_CHECK(
          newsize != 0,
          "cannot reshape tensor of 0 elements into shape ",
          shape,
          " because the unspecified dimension size -1 can be any "
          "value and is ambiguous");
      res[*infer_dim] = numel / newsize;
    }
    return;
  }

  // 如果形状不合法，抛出运行时错误并附带详细信息
  std::ostringstream ss;
  ss << "shape '" << shape << "' is invalid for input of size " << numel;
  throw std::runtime_error(ss.str());
}

// 推断形状，返回一个 int64_t 类型的向量
inline std::vector<int64_t> infer_size(IntArrayRef shape, int64_t numel) {
  auto res = shape.vec();
  infer_size_impl(shape, numel, res);
  return res;
}

// 推断形状，返回一个 DimVector 类型的向量
inline at::DimVector infer_size_dv(IntArrayRef shape, int64_t numel) {
  auto res = at::DimVector(shape);
  infer_size_impl(shape, numel, res);
  return res;
}

// 推断形状，返回一个 SymDimVector 类型的向量
inline at::SymDimVector infer_size_dv(
    c10::SymIntArrayRef shape,
    c10::SymInt numel) {
  auto res = at::SymDimVector(shape);
  infer_size_impl<c10::SymIntArrayRef, c10::SymInt, at::SymDimVector>(
      shape, std::move(numel), res);
  return res;
}

} // namespace at


这段代码是一个 C++ 的命名空间 `at` 下的函数和模板定义，用于推断张量的形状大小。具体功能包括：

1. `infer_size_impl` 函数：推断具有大小为 -1 的维度的大小，检查新形状是否与元素数量兼容。
2. `infer_size`, `infer_size_dv` 和 `infer_size_dv` 函数：分别用于推断形状并返回不同类型的向量（`std::vector<int64_t>`、`at::DimVector` 和 `at::SymDimVector`）。

每个函数和模板都有详细的注释，解释其功能和用途。
```