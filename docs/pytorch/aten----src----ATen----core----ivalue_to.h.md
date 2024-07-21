# `.\pytorch\aten\src\ATen\core\ivalue_to.h`

```
#pragma once
// 预处理指令，确保本头文件只被编译一次

#include <string>
// 引入标准字符串库

namespace at {
class Tensor;
} // namespace at
// 在 at 命名空间中声明 Tensor 类

namespace c10 {
struct IValue;
namespace detail {
// 在 c10 命名空间中的 detail 命名空间中声明以下内容

// 确定 `IValue::to() const &` 的返回类型。当可能时是常引用，否则是拷贝。
// 该结构用于 List 类中，并放在独立的头文件中以便于使用。
template<typename T>
struct ivalue_to_const_ref_overload_return {
  using type = T;
};

// 特化模板，确定 at::Tensor 的返回类型为常引用
template<>
struct ivalue_to_const_ref_overload_return<at::Tensor> {
  using type = const at::Tensor&;
};

// 特化模板，确定 std::string 的返回类型为常引用
template<>
struct ivalue_to_const_ref_overload_return<std::string> {
  using type = const std::string&;
};

// 特化模板，确定 IValue 的返回类型为常引用
template<>
struct ivalue_to_const_ref_overload_return<IValue> {
  using type = const IValue&;
};

} // namespace detail
} // namespace c10
```