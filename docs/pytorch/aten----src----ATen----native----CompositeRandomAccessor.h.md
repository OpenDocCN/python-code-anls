# `.\pytorch\aten\src\ATen\native\CompositeRandomAccessor.h`

```py
#pragma once
// 预处理指令，确保头文件只包含一次

#include <ATen/native/CompositeRandomAccessorCommon.h>
// 包含 CompositeRandomAccessorCommon.h 文件，其中包含了一些与复合随机访问器相关的内容

namespace at::native {

struct TupleInfoCPU {
  // 定义一个结构体 TupleInfoCPU

  template <typename ...Types>
  using tuple = std::tuple<Types...>;
  // 定义一个别名模板 tuple，表示一个包含任意类型模板参数的 std::tuple

  template <typename ...Types>
  static constexpr auto tie(Types&... args) noexcept {
    return std::tie(args...);
  }
  // 定义一个静态成员函数 tie，返回一个 std::tuple 对象，包含传入参数的引用

};

template <typename KeyAccessor, typename ValueAccessor>
using CompositeRandomAccessorCPU =
  CompositeRandomAccessor<KeyAccessor, ValueAccessor, TupleInfoCPU>;
// 定义一个别名模板 CompositeRandomAccessorCPU，用于 CPU 上的复合随机访问器，使用 KeyAccessor 和 ValueAccessor 类型参数，以及 TupleInfoCPU

template <typename Values, typename References>
void swap(
  references_holder<Values, References> rh1,
  references_holder<Values, References> rh2
) {
  return std::swap(rh1.data(), rh2.data());
}
// 定义一个模板函数 swap，用于交换两个 references_holder 对象的数据

template <int N, typename Values, typename References>
auto get(references_holder<Values, References> rh) -> decltype(std::get<N>(rh.data())) {
  return std::get<N>(rh.data());
}
// 定义一个模板函数 get，返回 references_holder 中第 N 个元素的数据

} // namespace at::native
// 命名空间结束
```