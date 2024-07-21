# `.\pytorch\aten\src\ATen\native\Constraints.cpp`

```
// 包含 C++ 标准库头文件，用于导入 <limits> 库
#include <limits>
// 定义宏，用于设置 TORCH_ASSERT_ONLY_METHOD_OPERATORS
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
// 导入 ATen 库中的 Tensor 类定义
#include <ATen/core/Tensor.h>
// 导入 ATen 库中的 Device 类定义
#include <c10/core/Device.h>
// 导入 ATen 库中的 Layout 类定义
#include <c10/core/Layout.h>
// 导入 ATen 库中的 MemoryFormat 类定义
#include <c10/core/MemoryFormat.h>
// 导入 ATen 库中的 Scalar 类定义
#include <c10/core/Scalar.h>
// 导入 ATen 库中的 ScalarType 枚举定义
#include <c10/core/ScalarType.h>
// 导入 ATen 库中的 Optional 类定义
#include <c10/util/Optional.h>

// 根据是否定义了 AT_PER_OPERATOR_HEADERS 来选择导入不同的头文件
#ifndef AT_PER_OPERATOR_HEADERS
// 导入 ATen 库中的函数和原生函数声明
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
// 导入特定函数声明头文件
#include <ATen/ops/_functional_sym_constrain_range_native.h>
#include <ATen/ops/_make_dep_token_native.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/sym_constrain_range_native.h>
#include <ATen/ops/sym_constrain_range_for_size_native.h>
#include <ATen/ops/_functional_sym_constrain_range_for_size_native.h>
#endif

// ATen 库的命名空间
namespace at::native {

// 函数定义：sym_constrain_range
void sym_constrain_range(
    const Scalar& size,
    std::optional<int64_t> min,
    std::optional<int64_t> max) {

    // 初始化最小值和最大值，若未指定则取默认值
    int64_t min_val = min.has_value() ? min.value() : std::numeric_limits<int64_t>::min();
    int64_t max_val = max.has_value() ? max.value() : std::numeric_limits<int64_t>::max();
    // 将 Scalar 转换为 int64_t 类型
    int64_t size_as_int = size.toLong();

    // 检查最大值是否大于等于最小值
    TORCH_CHECK(
      max_val >= min_val,
      "Max must be greater than or equal to min. Got min=",
      min_val,
      " max=",
      max_val
    );

    // 检查 size_as_int 是否在 min_val 和 max_val 范围内
    TORCH_CHECK(
      min_val <= size_as_int && size_as_int <= max_val,
      "Invalid value range for ",
      size_as_int,
      " between [",
      min_val,
      ", ",
      max_val,
      "]."
    );
}

// 函数定义：_functional_sym_constrain_range
Tensor _functional_sym_constrain_range(
    const Scalar& size,
    std::optional<int64_t> min,
    std::optional<int64_t> max,
    const Tensor& dep_token) {
  // 调用 sym_constrain_range 函数进行范围约束检查
  sym_constrain_range(size, min, max);
  // 返回 dep_token 的克隆副本
  return dep_token.clone();
}

// 函数定义：sym_constrain_range_for_size
void sym_constrain_range_for_size(const Scalar& size, std::optional<int64_t> min, std::optional<int64_t> max) {
  // 初始化最小值，若未指定则取默认值
  int64_t min_val = min.has_value() ? min.value() : 0;
  // 若最大值指定且小于等于 2，则抛出错误
  if (max.has_value() && max.value() <= 2) {
    TORCH_CHECK(false, "Max value to constrain_range_for_size must be greater than 2. got: ", max.value());
  }
  // 调用 sym_constrain_range 函数进行范围约束检查
  sym_constrain_range(size, min_val, max);
}

// 函数定义：_functional_sym_constrain_range_for_size
Tensor _functional_sym_constrain_range_for_size(
  const Scalar& size,
  std::optional<int64_t> min,
  std::optional<int64_t> max,
  const Tensor& dep_token) {
  // 调用 sym_constrain_range_for_size 函数进行范围约束检查
  sym_constrain_range_for_size(size, min, max);
  // 返回 dep_token 的克隆副本
  return dep_token.clone();
}

// 函数定义：_make_dep_token_cpu
Tensor _make_dep_token_cpu(
    std::optional<ScalarType> dtype_opt,
    std::optional<Layout> layout_opt,
    std::optional<Device> device_opt,
    std::optional<bool> pin_memory_opt,
    std::optional<c10::MemoryFormat> memory_format_opt) {
  // 创建一个空 Tensor
  return at::empty(
      {}, dtype_opt, layout_opt, device_opt, pin_memory_opt, memory_format_opt);
}

} // namespace at::native


这段代码是一个 C++ 的命名空间 `at::native`，定义了一些涉及张量操作的函数和范围约束的功能。
```