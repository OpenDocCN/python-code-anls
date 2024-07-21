# `.\pytorch\aten\src\ATen\native\cuda\CompositeRandomAccessor.h`

```
#pragma once
// 预处理命令，确保头文件只被包含一次

#include <ATen/native/CompositeRandomAccessorCommon.h>
// 包含 ATen 库中的 CompositeRandomAccessorCommon.h 头文件

#include <thrust/tuple.h>
// 包含 thrust 库中的 tuple 头文件

namespace at { namespace native {
// 进入 at 和 native 命名空间

struct TupleInfoCPU {
  template <typename ...Types>
  using tuple = thrust::tuple<Types...>;
  // 定义 TupleInfoCPU 结构体，包含一个用于模板类型的 tuple 别名，使用 thrust::tuple 实现

  template <typename ...Types>
  static constexpr auto tie(Types&... args) noexcept {
    return thrust::tie(args...);
  }
  // 定义静态成员函数 tie，返回一个用于参数包的 tuple，通过 thrust::tie 实现
};

template <typename KeyAccessor, typename ValueAccessor>
using CompositeRandomAccessorCPU =
  CompositeRandomAccessor<KeyAccessor, ValueAccessor, TupleInfoCPU>;
// 定义 CompositeRandomAccessorCPU 类型别名，使用 CompositeRandomAccessor 模板类，
// 其中 KeyAccessor、ValueAccessor 作为模板参数，TupleInfoCPU 作为第三个模板参数

template <typename Values, typename References>
void swap(
  references_holder<Values, References> rh1,
  references_holder<Values, References> rh2
) {
  return thrust::swap(rh1.data(), rh2.data());
}
// 定义 swap 函数，用于交换两个 references_holder 的数据，调用 thrust::swap 实现

template <int N, typename Values, typename References>
auto get(references_holder<Values, References> rh) -> decltype(thrust::get<N>(rh.data())) {
  return thrust::get<N>(rh.data());
}
// 定义 get 函数模板，获取 references_holder 的第 N 个元素，返回值类型由 thrust::get 推导得出

}} // namespace at::native
// 结束 at 和 native 命名空间
```