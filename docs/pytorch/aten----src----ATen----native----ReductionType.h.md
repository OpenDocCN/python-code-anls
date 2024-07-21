# `.\pytorch\aten\src\ATen\native\ReductionType.h`

```py
#pragma once
// 预处理指令，确保头文件只包含一次

#include <c10/core/Scalar.h>
// 包含 C++ Tensor 类型的标量头文件

namespace at::native {

enum class ReductionType {MAX, MEAN, MIN, SUM, PROD};
// 定义枚举类型 ReductionType，表示各种归约操作类型

inline ReductionType get_reduction_enum(const c10::string_view& reduce) {
  // 定义内联函数，根据字符串 reduce 返回对应的 ReductionType 枚举值
  if (reduce == "max" || reduce == "amax") {
    return ReductionType::MAX;
  } else if (reduce == "mean") {
    return ReductionType::MEAN;
  } else if (reduce == "min" || reduce == "amin") {
    return ReductionType::MIN;
  } else if (reduce == "sum") {
    return ReductionType::SUM;
  } else if (reduce == "prod") {
    return ReductionType::PROD;
  } else {
    TORCH_CHECK(false, "reduce argument must be either sum, prod, mean, amax or amin, got ", reduce);
    // 如果 reduce 不匹配任何预定义的操作类型，抛出错误信息
  }
}

// used for `scatter_reduce`, old options for BC.
// 用于 scatter_reduce，旧的选项用于向后兼容。

inline ReductionType get_operator_enum(const c10::string_view reduce, bool use_new_options) {
  // 定义内联函数，根据字符串 reduce 和布尔值 use_new_options 返回对应的 ReductionType 枚举值
  if (use_new_options) {
    return get_reduction_enum(reduce);
    // 如果 use_new_options 为 true，则调用 get_reduction_enum 返回结果
  } else {
    if (reduce == "add") {
      return ReductionType::SUM;
      // 如果 reduce 是 "add"，返回 SUM 枚举值
    } else if (reduce == "multiply") {
      return ReductionType::PROD;
      // 如果 reduce 是 "multiply"，返回 PROD 枚举值
    } else {
      TORCH_CHECK(false, "reduce argument must be either add or multiply.")
      // 如果 reduce 不是 "add" 或 "multiply"，抛出错误信息
    }
  }
}

} // at::native
// 命名空间 at::native 结束
```